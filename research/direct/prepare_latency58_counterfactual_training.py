"""Freeze one 250-update selective trial after both GPU rehearsals pair exactly."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_counterfactual_checkpoint import validate_recipe, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Training preparation plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-counterfactual-training-preparation-plan-v1", "Different preparation")
    for item in (plan["pairing"], plan["pairing_execution"], plan["resource_plan"], plan["resource_execution"]):
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound production prerequisite")
    paired, paired_execution = read(plan["pairing"]["path"]), read(plan["pairing_execution"]["path"])
    require(paired["schema"] == "latency58-counterfactual-resource-pairing-v1" and paired["status"] == "pass"
            and paired["source_bindings_unchanged"] and paired["updates_per_mode"] == 2
            and paired["existing_control_replay_exact"] and paired["all_inputs_and_teacher_targets_exact"]
            and paired["final_rng_states_exact"] and paired["selected_loss_changes_updates"]
            and paired_execution["actual_exit_code"] == 0 and not paired_execution["timed_out"]
            and paired_execution["source_bindings_unchanged"]
            and paired_execution["plan_sha256"] == paired["plan_sha256"], "GPU pairing is incomplete")
    verify_inputs(paired)
    resource_plan = read(plan["resource_plan"]["path"])
    validate_recipe(resource_plan)
    verify_inputs(resource_plan)
    protocol = resource_plan["matched_protocol"]
    require(plan["source_bindings"].get(protocol["path"]) == protocol["sha256"] == sha(protocol["path"]),
            "Unbound new protocol")
    matched = read(protocol["path"])
    require(matched["schema"] == "latency58-counterfactual-protocol-v1"
            and matched["production_teacher_modes"] == ["ordinary_only"] and matched["maximum_production_updates"] == 250
            and matched["quality_endpoints"] == [250] and not matched["automatic_continuation"],
            "Different bounded production schedule")
    resource_binding = paired["resource_results"]["ordinary_only"]
    require(paired["training_plans"]["ordinary_only"] == plan["resource_plan"]
            and plan["source_bindings"].get(resource_binding["path"]) == resource_binding["sha256"] == sha(resource_binding["path"]),
            "Different paired recipe or resource result")
    resource, execution = read(resource_binding["path"]), read(plan["resource_execution"]["path"])
    monitor = read(execution["monitor_result"])
    require(resource_plan["resource_only"] and resource_plan["teacher_mode"] == "ordinary_only"
            and resource["status"] == "pass" and resource["source_bindings_unchanged"]
            and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == resource["plan_sha256"] == plan["resource_plan"]["sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Selective rehearsal did not close successfully")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "training-preparation.json").exists(), "Preserve preparation")
    before = require_space(resource_plan, 380_000_000)
    candidate = copy.deepcopy(resource_plan)
    candidate.update(resource_only=False, run_dir=str(PHASE / "counterfactual-teacher-ordinary-only-b16-micro4-lr3e5-250"),
                     resource_plan=plan["resource_plan"], full_resource=resource_binding,
                     full_resource_execution=plan["resource_execution"], resource_pairing=plan["pairing"],
                     resource_pairing_execution=plan["pairing_execution"],
                     source_bindings={**resource_plan["source_bindings"], **paired["source_bindings"],
                                      **plan["source_bindings"], str(args.plan): args.plan_sha256})
    require(not Path(candidate["run_dir"]).exists(), "Preserve previous production trial")
    validate_recipe(candidate)
    verify_inputs(candidate)
    path = out / "ordinary_only-training-plan.json"
    write(path, candidate)
    write(out / "training-preparation.json", {"schema": "latency58-counterfactual-training-preparation-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": candidate["source_bindings"], "source_bindings_unchanged": True,
          "training_plan": {"path": str(path), "sha256": sha(path)}, "maximum_new_production_updates": 250,
          "quality_endpoints": [250], "automatic_continuation": False, "training_updates_executed": 0,
          "checkpoint_and_quality_reserve_bytes": 380_000_000, "quality_selected": False,
          "counted_bytes_before": before, "counted_bytes_after": require_space(candidate, 0)})
    print({"status": "pass", "training_plan": str(path), "training_plan_sha256": sha(path)}, flush=True)


if __name__ == "__main__":
    main()
