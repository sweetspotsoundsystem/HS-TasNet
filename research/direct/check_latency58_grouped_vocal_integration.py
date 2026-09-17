"""CPU integration checks for the prospective grouped trainer and GPU checker."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def check_plan_boundary(template, budget, data, out):
    from research.direct.train_latency58_grouped_vocal import validate_recipe, applied_policy
    from research.direct.latency58_grouped_vocal_auxiliary import VERSION
    fixture = {**template, "schema": "latency58-grouped-vocal-training-plan-v1", "objective_version": VERSION,
        "config": {**template["config"], "auxiliary_microbatch_size": 2},
        "output_directory": str(out / "not_a_prepared_training_plan"), "grouped_vocal_loss": applied_policy(),
        "qualified_data_prefix": data["batches"], "storage_budget": {**budget, "counted_roots": list(budget["counted_roots"])}}
    validate_recipe(fixture)
    cases = []

    def rejects(label, change):
        candidate = copy.deepcopy(fixture)
        change(candidate)
        try:
            validate_recipe(candidate)
        except RuntimeError as error:
            cases.append({"case": label, "rejection": str(error)})
        else:
            raise RuntimeError("Invalid candidate recipe accepted: " + label)

    rejects("ordinary_only_schema", lambda p: p.update(schema=template["schema"]))
    rejects("changed_auxiliary_weight", lambda p: p["grouped_vocal_loss"].update(auxiliary_weight=.2))
    rejects("ordinary_batch_replaced_by_auxiliary", lambda p: p["config"].update(batch_size=14))
    rejects("unqualified_auxiliary_microbatch", lambda p: p["config"].update(auxiliary_microbatch_size=1))
    rejects("shorter_training_context", lambda p: p.update(warmup_samples=512))
    rejects("missing_auxiliary_history_binding", lambda p: p["qualified_data_prefix"][0].pop("auxiliary_full_context_sha256"))
    rejects("nonhex_auxiliary_history_binding", lambda p: p["qualified_data_prefix"][0].update(auxiliary_full_context_sha256="x" * 64))
    rejects("negative_activity_count", lambda p: p["qualified_data_prefix"][0].update(auxiliary_active_counts=[-1, 2, 2, 2], auxiliary_absent_counts=[5, 2, 2, 2]))
    rejects("wrong_ordinary_data_cursor", lambda p: p["qualified_data_prefix"][1].update(first_index=0))
    rejects("zero_warmup_schedule", lambda p: p["config"].update(warmup=0))
    rejects("missing_checkpoint_reserve", lambda p: p["storage_budget"].update(live_training_save_reservation_bytes=0))
    rejects("external_git_omitted_from_accounting", lambda p: p["storage_budget"].update(external_git_common_directory=p["counted_roots"][0]))
    require(not Path(fixture["output_directory"]).exists(), "Plan boundary checks created a training output")
    return {"status": "pass", "structural_fixture_accepted_without_launch": True, "rejected_cases": cases,
            "scope": "Pure recipe validation; no GPU plan, selected parent, launch specification or training directory created"}


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.check_latency58_grouped_vocal_gpu import compare_group_gradients, check_restart, check_gpu
    from research.direct.run_latency58_grouped_vocal import require_resource_result
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    from research.direct.check_latency58_branch_long_context_data import audio_sha
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    previous = PHASE / "grouped-vocal-auxiliary-restart-001"
    old, previous_result, execution = (read(previous / name) for name in ("plan.json", "result.json", "execution.json"))
    require(previous_result["status"] == "pass" and previous_result["source_bindings_unchanged"]
            and previous_result["plan_sha256"] == sha(previous / "plan.json")
            and execution["actual_exit_code"] == execution["actual_enclosing_exit_code"] == 0
            and execution["source_bindings_unchanged"] and not execution["timed_out"],
            "Grouped step/restart CPU qualification is incomplete")
    bindings = dict(old["source_bindings"])
    paths = [Path(__file__).resolve()]
    paths.extend(ROOT / "research/direct" / name for name in (
        "check_latency58_grouped_vocal_gpu.py", "train_latency58_grouped_vocal.py", "run_latency58_grouped_vocal.py",
        "check_latency58_grouped_vocal_restart.py", "latency58_grouped_vocal_step.py"))
    paths.extend(previous / name for name in ("plan.json", "result.json", "execution.json"))
    existing = PHASE / "branch-long-context-006"
    paths.extend(existing / name for name in (
        "plan.json", "resource-run/result.json", "resource-run/branch-long-context-gpu-parity.json"))
    paths.append(PHASE / "grouped-vocal-auxiliary-data-001/result.json")
    bindings.update({str(path): sha(path) for path in paths})
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "grouped-vocal-trainer-integration-cpu-001"
    require(not out.exists(), "Preserve earlier integration qualification")
    budget = read(PHASE / "branch-gru-int8-post-ci-storage-001.json")
    plan = {"schema": "latency58-grouped-trainer-integration-cpu-v1", "source_bindings": bindings,
        "fixture_checkpoint": old["fixture_checkpoint"], "fixture_model_state_sha256": old["fixture_model_state_sha256"],
        "warmup_samples": 512, "scored_samples": 44160, "ordinary_microbatch": 4, "auxiliary_microbatch": 2,
        "precision": "CPU FP32", "seed": 202610309,
        "scope": "Execute the new resource checker's gradient and serialized-restart functions on short CPU context; GPU execution remains unqualified",
        "gpu_used": False, "future_parent_selected": False, "production_recipe_selected": False,
        "budget_before": budget_snapshot(budget)}
    out.mkdir(); write(out / "plan.json", plan)
    began = time.monotonic()
    template = read(existing / "plan.json")
    data = read(PHASE / "grouped-vocal-auxiliary-data-001/result.json")
    boundaries = check_plan_boundary(template, budget, data, out)
    write(out / "plan-boundary-checks.json", boundaries)
    try:
        require_resource_result(read(existing / "resource-run/result.json"),
            read(existing / "resource-run/branch-long-context-gpu-parity.json"), template, sha(existing / "plan.json"))
    except (RuntimeError, KeyError) as error:
        legacy_rejection = {"status": "pass", "rejection": str(error), "actual_ordinary_only_gpu_result_rejected": True}
    else:
        raise RuntimeError("Ordinary-only GPU evidence was accepted for grouped training")
    model, payload = load_model(plan["fixture_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"], "Retained fixture changed")
    model.train().requires_grad_(True); model.training_precision = "fp32"
    try:
        check_gpu(model, template)
    except RuntimeError as error:
        require(str(error) == "Require the selected BF16 parent before updates", "Unexpected CPU entry rejection")
    else:
        raise RuntimeError("GPU qualification accepted a CPU model")
    rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(plan["seed"])
    truth = .02 * torch.randn(16, 4, 2, plan["warmup_samples"] + plan["scored_samples"], generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    mixture = truth.sum(1)
    inputs_sha = audio_sha(mixture, truth)

    def progress(phase, group, offset):
        print(json.dumps({"event": "grouped_integration_progress", "phase": phase, "group": group, "offset": offset}), flush=True)

    gradients = compare_group_gradients(model, mixture, truth, warmup_samples=plan["warmup_samples"], progress=progress)
    write(out / "grouped-gradients.json", gradients)
    require(audio_sha(mixture, truth) == inputs_sha, "Gradient checker changed input data")
    del mixture, truth
    gc.collect()
    restart_plan = {"parent_checkpoint": plan["fixture_checkpoint"], "parent_training_updates": payload["provenance"]["training_updates"],
                    "ema": template["ema"]}
    restart = check_restart(model, restart_plan, progress=progress)
    write(out / "grouped-restart.json", restart)
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "CPU integration changed retained weights, RNG or initialized CUDA")
    verify_inputs(plan)
    result = {"status": "pass", "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "plan_boundary_checks": boundaries, "legacy_resource_rejection": legacy_rejection,
        "gpu_entry_rejected_cpu_before_cuda_initialization": True,
        "grouped_gradients": {"path": str(out / "grouped-gradients.json"), "sha256": sha(out / "grouped-gradients.json")},
        "grouped_restart": {"path": str(out / "grouped-restart.json"), "sha256": sha(out / "grouped-restart.json")},
        "retained_parent_and_rng_unchanged": True, "gpu_used": False, "gpu_execution_qualified": False,
        "future_parent_selected": False, "production_recipe_selected": False, "quality_measured": False,
        "elapsed_seconds": time.monotonic() - began, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_after": budget_snapshot(budget), "completed_utc": datetime.now(timezone.utc).isoformat()}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "peak_rss_bytes", "gpu_used")}), flush=True)


if __name__ == "__main__":
    main()
