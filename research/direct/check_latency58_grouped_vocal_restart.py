"""Rehearse grouped Adam/EMA boundaries and serialized restart on CPU only."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def fingerprint(model, optimizer, ema):
    moments = {str(index) + "/" + key: value for index, state in optimizer.state_dict()["state"].items()
               for key, value in state.items()}
    return {"model": state_sha256(model.state_dict()), "adam": state_sha256(moments),
            "adam_groups": copy.deepcopy(optimizer.state_dict()["param_groups"]),
            "ema": state_sha256(ema.parameters), "ema_updates": ema.updates,
            "ema_raw_owner": ema.raw_state_sha256}


class RequestedStop(RuntimeError):
    pass


def exercise(model, optimizer, ema, inputs, *, step, stop_after=None):
    import torch
    from research.direct.latency58_grouped_vocal_step import grouped_update
    before = fingerprint(model, optimizer, ema)
    groups, gradients, optimizer_calls, stop = [], {}, [], False

    def check_continue():
        if stop:
            raise RequestedStop("Requested stop before committing grouped update")

    def after_group(group, row):
        nonlocal stop
        require(fingerprint(model, optimizer, ema) == before,
                "Weights, Adam or EMA advanced before both groups completed")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
                    for p in model.parameters()), "A group did not reach all 40 parameters")
        if group == "ordinary":
            gradients.update({name: p.grad.detach().clone() for name, p in model.named_parameters()})
        else:
            require(all(float((p.grad - gradients[name]).norm()) > 0 for name, p in model.named_parameters()),
                    "Auxiliary group did not add to every parameter gradient")
            gradients.clear()
        groups.append(group)
        stop = group == stop_after
        print(json.dumps({"event": "grouped_restart_progress", "step": step,
                          "group": group, "stop_after": stop_after, "weighted_loss": row["weighted_loss"]}), flush=True)

    hook = optimizer.register_step_post_hook(lambda *args: optimizer_calls.append(1))
    try:
        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clips, \
                patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zeroes:
            try:
                result = grouped_update(model, optimizer, ema, *inputs, step=step, warmup_samples=512,
                                        check_continue=check_continue, after_group=after_group)
            except RequestedStop:
                require(stop_after is not None and fingerprint(model, optimizer, ema) == before
                        and not optimizer_calls and clips.call_count == 0 and zeroes.call_count == 1,
                        "Interrupted accumulation changed an optimizer endpoint")
                require(groups == (["ordinary"] if stop_after == "ordinary" else ["ordinary", "auxiliary"]),
                        "Wrong interruption boundary")
                return {"stop_after": stop_after, "groups_completed": groups,
                        "weights_adam_ema_unchanged": True, "optimizer_calls": 0, "clip_calls": 0}
            require(stop_after is None and groups == ["ordinary", "auxiliary"] and optimizer_calls == [1]
                    and clips.call_count == zeroes.call_count == 1 and ema.updates == before["ema_updates"] + 1,
                    "Grouped update was not committed exactly once")
            require(result["raw_model_state_sha256"] != before["model"]
                    and result["ema_parameters_sha256"] != before["ema"]
                    and len(result["groups"]["ordinary"]["microbatches"]) == 4
                    and len(result["groups"]["auxiliary"]["microbatches"]) == 1,
                    "Grouped update or its microbatch geometry differs")
            result["observed_optimizer_calls"] = len(optimizer_calls)
            result["observed_clip_calls"] = clips.call_count
            result["observed_zero_grad_calls"] = zeroes.call_count
            result["both_group_gradients_reach_all_40_parameters"] = True
            result["endpoint_unchanged_at_both_group_boundaries"] = True
            return result
    finally:
        hook.remove()


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_live, audit_resume
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_ema_checkpoint import make_payloads, audit_payloads, policy as ema_policy
    from research.direct.latency58_grouped_vocal_auxiliary import VERSION, policy
    from research.direct.latency58_grouped_vocal_step import grouped_update
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    from research.direct.check_latency58_branch_long_context_data import audio_sha
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    previous = PHASE / "grouped-vocal-auxiliary-gradients-001"
    old, result, execution = (read(previous / name) for name in ("plan.json", "result.json", "execution.json"))
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and result["plan_sha256"] == sha(previous / "plan.json")
            and execution["actual_exit_code"] == execution["actual_enclosing_exit_code"] == 0
            and execution["source_bindings_unchanged"] and not execution["timed_out"],
            "Whole-objective model-gradient qualification incomplete")
    bindings = dict(old["source_bindings"])
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_grouped_vocal_step.py",
             PHASE / "branch-gru-int8-post-ci-storage-001.json"]
    paths.extend(previous / name for name in ("plan.json", "result.json", "execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_branch_ema.py", "latency58_branch_ema_checkpoint.py", "latency58_branch_memory_checkpoint.py",
        "train_latency58_branch_sdr_ema.py", "run_latency58_deployed_vocal_views.py"))
    bindings.update({str(path): sha(path) for path in paths})
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "grouped-vocal-auxiliary-restart-001"
    require(not out.exists(), "Preserve earlier restart rehearsals")
    budget_plan = read(PHASE / "branch-gru-int8-post-ci-storage-001.json")
    plan = {"schema": "latency58-grouped-vocal-cpu-update-restart-v1", "source_bindings": bindings,
        "fixture_checkpoint": old["fixture_checkpoint"], "fixture_model_state_sha256": old["fixture_model_state_sha256"],
        "policy": policy(), "warmup_samples": 512, "scored_samples": 44160,
        "ordinary_microbatch": 4, "auxiliary_microbatch": 2, "precision": "CPU FP32",
        "fixture": "Synthetic short context for complete optimizer/restart integration only; full recorded context qualified separately",
        "seed": 202610309, "optimizer": {"name": "Adam", "lr": 6e-5, "foreach": False, "clip_norm": 5.},
        "ema": ema_policy(.995), "serialized_after_step": 2, "replayed_step": 3,
        "future_parent_selected": False, "production_recipe_selected": False, "gpu_used": False,
        "checkpoint_files_written": False, "quality_measured": False, "budget_before": budget_snapshot(budget_plan)}
    out.mkdir(); write(out / "plan.json", plan)
    began = time.monotonic()
    parent, payload = load_model(plan["fixture_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    require(parent_sha == plan["fixture_model_state_sha256"], "Retained model fixture changed")
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.training_precision = "fp32"
    model.provenance = {**model.provenance, "branch_memory_previous_provenance": copy.deepcopy(model.provenance),
        "branch_memory_parent_model_state_sha256": parent_sha,
        "branch_memory_current_stage_corpus": "Synthetic CPU restart fixture; no recorded examples",
        "branch_memory_current_stage_augmentation": "Fixed synthetic batch and two derived source views",
        "branch_memory_current_stage_training_context": {"warmup_samples": 512, "scored_samples": 44160}}
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    ema = BranchParameterEMA(model, decay=.995, base_state_sha256=parent_sha)
    audit_live(model, optimizer, 0, frozen)
    rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(plan["seed"])
    truth = .02 * torch.randn(16, 4, 2, 512 + 44160, generator=generator)
    # Uneven activity among ordinary microbatches exercises whole-group denominators.
    truth[:3, 2] = 0
    truth[4:6, 1] = 0
    truth[8:9, 3] = 0
    inputs = (truth.sum(1), truth)
    input_sha = audio_sha(*inputs)
    initial = fingerprint(model, optimizer, ema)
    try:
        grouped_update(model, optimizer, ema, *inputs, step=2, warmup_samples=512)
    except RuntimeError as error:
        require(str(error) == "Grouped update must follow the contiguous EMA endpoint",
                "Unexpected noncontiguous update rejection")
    else:
        raise RuntimeError("Noncontiguous update accepted")
    require(fingerprint(model, optimizer, ema) == initial and all(p.grad is None for p in model.parameters()),
            "Invalid step changed the initial endpoint or gradients")
    interrupted = [exercise(model, optimizer, ema, inputs, step=1, stop_after=group)
                   for group in ("ordinary", "auxiliary")]
    updates = []
    for step in (1, 2):
        updates.append(exercise(model, optimizer, ema, inputs, step=step))
        audit_live(model, optimizer, step, frozen)
    fixture = {"schema": "latency58-synthetic-grouped-restart-fixture-v1",
        "config": {"steps": 2, "batch_size": 16, "data_start": 0},
        "parent_checkpoint": plan["fixture_checkpoint"], "parent_model_state_sha256": parent_sha,
        "parent_training_updates": payload["provenance"]["training_updates"],
        "fixed_buffers_sha256": state_sha256(frozen), "ema": plan["ema"],
        "objective_version": VERSION, "precision_policy": "CPU FP32 synthetic restart fixture",
        "fixture_data_sha256": input_sha, "ordinary_data_cursor_counts_auxiliary_examples": False,
        "qualification_plan_sha256": sha(out / "plan.json")}
    write(out / "fixture-plan.json", fixture)
    fixture_sha = sha(out / "fixture-plan.json")
    raw, resume, averaged, metadata = make_payloads(model, optimizer, ema, 2, fixture, fixture_sha)
    require(resume["next_sample_index"] == 32, "Auxiliary examples incorrectly advanced the ordinary data cursor")
    serialized, loaded = {}, {}
    for name, value in (("raw", raw), ("optimizer", resume), ("averaged", averaged), ("ema_metadata", metadata)):
        with io.BytesIO() as stream:
            torch.save(value, stream)
            view = stream.getbuffer()
            serialized[name] = {"bytes": len(view), "sha256": hashlib.sha256(view).hexdigest()}
            del view
            stream.seek(0)
            loaded[name] = torch.load(stream, map_location="cpu", weights_only=True)
    restored, recovered, averaged_model, restored_ema = audit_payloads(
        loaded["raw"], loaded["optimizer"], loaded["averaged"], loaded["ema_metadata"], fixture, fixture_sha)
    require("weight_averaging" not in loaded["raw"]["provenance"]
            and loaded["raw"]["provenance"]["initial_parent_weight_averaging"] == parent.provenance["weight_averaging"],
            "Inherited average confused with current raw weights")
    try:
        audit_resume(averaged_model, loaded["averaged"], recovered, fixture, fixture_sha)
    except RuntimeError as error:
        require(str(error) == "Saved optimizer belongs to another endpoint", "Unexpected EMA/Adam owner rejection")
    else:
        raise RuntimeError("Raw Adam accepted for averaged weights")
    restored.train().requires_grad_(True); restored.training_precision = "fp32"
    restarted = torch.optim.Adam(restored.parameters(), lr=6e-5, foreach=False)
    restarted.load_state_dict(recovered["optimizer"])
    audit_live(restored, restarted, 2, frozen)
    require(fingerprint(model, optimizer, ema) == fingerprint(restored, restarted, restored_ema),
            "Serialized endpoint differs before resumed update")
    del raw, resume, averaged, metadata, loaded, recovered, averaged_model
    gc.collect()
    third = exercise(model, optimizer, ema, inputs, step=3)
    replay = exercise(restored, restarted, restored_ema, inputs, step=3)
    audit_live(model, optimizer, 3, frozen)
    audit_live(restored, restarted, 3, frozen)
    require(third == replay and fingerprint(model, optimizer, ema) == fingerprint(restored, restarted, restored_ema),
            "Resumed grouped update differs from the continuous trajectory")
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        require(all(torch.equal(optimizer.state[first][key], restarted.state[second][key])
                    for key in ("step", "exp_avg", "exp_avg_sq")), "Resumed Adam moments differ")
    require(state_sha256(parent.state_dict()) == parent_sha and audio_sha(*inputs) == input_sha
            and torch.equal(torch.get_rng_state(), rng) and not torch.cuda.is_initialized(),
            "CPU fixture changed retained weights, input data, global RNG or initialized CUDA")
    verify_inputs(plan)
    result = {"status": "pass", "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "fixture_checkpoint": plan["fixture_checkpoint"], "fixture_input_sha256": input_sha,
        "noncontiguous_step_rejected_before_gradients": True, "interrupted_accumulations": interrupted,
        "updates": [*updates, third], "replayed_third_update": replay, "all_40_adam_states_checked": True,
        "serialized_in_memory": serialized, "saved_ordinary_cursor": 32,
        "resumed_raw_parameters_adam_moments_ema_and_update_accounting_bit_exact": True,
        "raw_adam_rejected_for_averaged_weights": True, "parent_data_rng_unchanged": True,
        "gpu_used": False, "checkpoint_files_written": False, "quality_measured": False,
        "future_parent_selected": False, "production_recipe_selected": False,
        "scope": plan["fixture"], "elapsed_seconds": time.monotonic() - began,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_after": budget_snapshot(budget_plan), "completed_utc": datetime.now(timezone.utc).isoformat()}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "peak_rss_bytes", "gpu_used")}), flush=True)


if __name__ == "__main__":
    main()
