"""CPU qualification of canonical whole-group derivatives and guarded restart."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.check_latency58_grouped_vocal_restart import fingerprint, RequestedStop

def exercise(model, optimizer, ema, inputs, *, step, stop_after=None):
    import torch
    from research.direct.latency58_grouped_vocal_canonical import grouped_update
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
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_grouped_vocal_canonical import policy
    from research.direct.check_latency58_grouped_vocal_canonical_gpu import compare_group_gradients, check_restart, check_gpu
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    out = PHASE / "grouped-vocal-canonical-cpu-001"
    require(not out.exists(), "Preserve canonical CPU qualification")
    failed = PHASE / "branch-grouped-vocal-007"
    diagnostic = PHASE / "grouped-vocal-gradient-diagnostic-001"
    parent_plan = read(failed / "plan.json")
    evidence, root_execution, diagnostic_inputs = (read(diagnostic / name) for name in
        ("child-result.json", "root-execution.json", "inputs.json"))
    require(root_execution["actual_exit_code"] == 0 and root_execution["source_bindings_unchanged"]
            and root_execution["actual_tool_chunk_id"] == "45e036"
            and root_execution["child_result_sha256"] == sha(diagnostic / "child-result.json")
            and evidence["status"] == "diagnostic_complete" and evidence["parent_weights_unchanged"]
            and not evidence["original_resource_pass"]
            and all(r["bitwise_equal"] for r in evidence["all_parameter_comparisons"]["replay_vs_direct"].values()),
            "Original diagnostic is incomplete")
    paths = [Path(__file__).resolve(), failed / "plan.json"]
    paths.extend(diagnostic / name for name in ("inputs.json", "child-result.json", "result.json", "execution.json", "root-execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in
        ("latency58_grouped_vocal_canonical.py", "check_latency58_grouped_vocal_canonical_gpu.py"))
    bindings = {**diagnostic_inputs["source_bindings"], **{str(p): sha(p) for p in paths}}
    plan = {"schema": "latency58-canonical-grouped-cpu-qualification-v1", "source_bindings": bindings,
        "fixture_checkpoint": parent_plan["parent_checkpoint"], "fixture_model_state_sha256": parent_plan["initialized_model_state_sha256"],
        "parent_plan": str(failed / "plan.json"), "accumulation_policy": policy(), "precision": "CPU FP32",
        "warmup_samples": 512, "scored_samples": 44160, "synthetic_seed": 202610309,
        "original_tolerances_changed": False, "gpu_used": False,
        "storage_budget": parent_plan["storage_budget"], "budget_before": budget_snapshot(parent_plan["storage_budget"])}
    verify_inputs(plan)
    out.mkdir(); write(out / "plan.json", plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    model, _ = load_model(plan["fixture_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"] and not torch.cuda.is_initialized(),
            "CPU selected parent differs")
    model.train().requires_grad_(True); model.training_precision = "fp32"
    rng = torch.get_rng_state().clone()
    began = time.monotonic()
    try:
        check_gpu(model, parent_plan)
    except RuntimeError as error:
        require(str(error) == "Require the selected BF16 parent before updates" and not torch.cuda.is_initialized(),
                "Unexpected GPU-entry rejection")
    else:
        raise RuntimeError("CPU model accepted at GPU resource entry")
    generator = torch.Generator().manual_seed(plan["synthetic_seed"])
    truth = .02 * torch.randn(16, 4, 2, 512 + 44160, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    def progress(phase, group, offset):
        print(json.dumps({"event": "canonical_cpu_progress", "phase": phase, "group": group, "offset": offset}), flush=True)
    gradients = compare_group_gradients(model, truth.sum(1), truth, warmup_samples=512, progress=progress)
    write(out / "gradients.json", gradients)
    print(json.dumps({"event": "canonical_cpu_gradients_pass", "parameters": len(gradients["all_40_gradients"])}), flush=True)
    del truth
    restart = check_restart(model, parent_plan, progress=progress)
    write(out / "restart.json", restart)
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "CPU parent or RNG changed")
    verify_inputs(plan)
    result = {"schema": plan["schema"], "status": "pass", "source_bindings_unchanged": True,
        "plan_sha256": sha(out / "plan.json"), "accumulation_policy": policy(),
        "gradients_sha256": sha(out / "gradients.json"), "restart_sha256": sha(out / "restart.json"),
        "all_40_parameter_gradients_match_independent_whole_objective": True,
        "interruption_before_commit_preserves_raw_adam_ema": True,
        "third_update_raw_adam_ema_and_accounting_bit_exact": restart["third_update_raw_adam_ema_and_accounting_bit_exact"],
        "parent_and_rng_unchanged": True, "gpu_entry_rejected_before_cuda_initialization": True,
        "original_tolerances_changed": False, "gpu_used": False, "checkpoint_files_written": False,
        "quality_measured": False, "gpu_execution_qualified": False, "elapsed_seconds": time.monotonic() - began,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "budget_after": budget_snapshot(plan["storage_budget"])}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "elapsed_seconds": result["elapsed_seconds"], "gpu_used": False}), flush=True)


if __name__ == "__main__":
    main()
