"""Prepare a matched 250-update learning-rate comparison from the accepted model."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, state_sha256, verify_inputs


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation required")
    control_path = PHASE / "cleanup-successor-prep-002/training-plan.json"
    control = read(control_path)
    verify_inputs(control)
    evidence = {}
    observations = {}
    for prefix, stage_name in (("cleanup-successor", "cleanup-successor-to-000250-002"),
                               ("cleanup-followup", "cleanup-followup-to-000250-001")):
        stage = PHASE / stage_name
        execution, audit_execution, audit = [read(stage / name) for name in ("execution.json", "audit-execution.json", "audit.json")]
        monitor = read(execution["monitor_result"])
        summary = PHASE / (prefix + "-250-summary-001/result.json")
        result = read(summary)
        require(execution["actual_exit_code"] == audit_execution["actual_exit_code"] == monitor["child_exit_code"] == 0
                and execution["source_bindings_unchanged"] and audit_execution["source_bindings_unchanged"]
                and not audit_execution["timed_out"] and audit["status"] == result["status"] == "pass"
                and audit["model_state_sha256"] == result["model_state_sha256"]
                and result["source_bindings_unchanged"] and audit["step"] == 250
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["post_exit_quiet_completed"], "Prior experiment did not close healthily")
        for p in (summary, stage / "execution.json", stage / "audit-execution.json", stage / "audit.json", Path(execution["monitor_result"])):
            evidence[str(p)] = sha(p)
        observations[prefix] = {"model_state_sha256": result["model_state_sha256"],
                               "full_sdr_db": result["full_mixture_aggregate"]["full_sdr_db"],
                               "low_sdr_db": result["full_mixture_aggregate"]["low_sdr_db"],
                               "bleed_sir_db": result["full_mixture_aggregate"]["bleed_sir_db"]}
    # Reserve half a GB for all known model payloads and objects outside counted roots.
    outside = 146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects")
    counted = sum(disk_bytes(Path(p)) for p in control["counted_roots"])
    require(outside < 500_000_000 and counted + 850_000_000 < 79_500_000_000, "Combined artifact cap exceeded")
    prep = PHASE / "cleanup-lr3e6-prep-001"
    require(not prep.exists(), "Preserve existing preparation")
    prep.mkdir()
    config = {**control["config"], "lr": 3e-6, "min_lr": 3e-7}
    decision = {"schema": "latency58-cleanup-lr3e6-decision-v1", "status": "train_lr3e6_independent_of_deployment",
                "config": config, "training_parent": control["parent"], "maximum_production_updates": 250,
                "resource_updates": 2, "quality_endpoints": [250], "qualification_blocks_training": False,
                "release_work_blocks_training": False, "optimizer_initialization": "fresh_adam",
                "matched_control_training_plan": binding(control_path), "observations": observations,
                "hypothesis": "The two lower-learning-rate continuation endpoints do not improve the accepted panel score. Test whether a lower learning-rate schedule better preserves mixture separation while learning cleanup from the accepted checkpoint. This arm is declared before seeing the higher-rate endpoint, completing a three-rate comparison against the existing middle-rate control. All 4000 crop addresses, augmentation RNG, teacher targets, model initialization, loss and context match the completed lower-rate successor; only peak and floor learning rate change.",
                "evaluation": "Unchanged full14, per-stem/bands/interference/absence, Actions60, probes, vocal-only and instrumental views. Compare accepted parent, working baseline and matched lower-rate control. Require the complete input/RNG/teacher journal to match; first gradients must be identical. No deployment selection or unobserved confirmation claim.",
                "storage": {"counted_bytes": counted, "outside_counted_roots_bytes": outside, "reserve_bytes": 850_000_000,
                            "counted_stop_bytes": 79_500_000_000, "combined_cap_bytes": 80_000_000_000},
                "source_bindings": evidence}
    write(prep / "decision.json", decision)
    plan = copy.deepcopy(control)
    for k in ("resource_plan", "full_resource", "full_resource_execution"):
        plan.pop(k, None)
    plan.update(schema="latency58-cleanup-lr3e6-training-v1", resource_only=True, config=config,
                run_dir=str(PHASE / "cleanup-lr3e6-resource-run-001"), comparison_variable="matched_learning_rate_schedule",
                stop_counted_bytes=79_500_000_000, preparation_decision=binding(prep / "decision.json"),
                matched_control_training_plan=binding(control_path))
    plan["source_bindings"].update(evidence)
    paths = [control_path, prep / "decision.json", Path(__file__).resolve(),
             *(ROOT / "research/direct").glob('*cleanup_lr3e6*.py'),
             *(Path(control["run_dir"]) / 'checkpoints/step-000250').glob('*')]
    plan["source_bindings"].update({str(p):sha(p) for p in paths if p.is_file()})
    from research.direct.latency58_cleanup_lr3e6_checkpoint import validate_recipe, load_parent
    verify_inputs(plan)
    validate_recipe(plan)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(plan)
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"]
            and not torch.cuda.is_initialized(), "Parent identity or CPU preparation differs")
    write(prep / "resource-plan.json", plan)
    print(json.dumps({"status":"prepared", "plan":binding(prep / "resource-plan.json"), "storage":decision["storage"]}), flush=True)


if __name__ == '__main__':
    main()
