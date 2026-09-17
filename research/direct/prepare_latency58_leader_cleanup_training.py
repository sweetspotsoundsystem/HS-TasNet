"""Freeze the original 250-update leader-cleanup schedule after its monitored rehearsal."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_leader_cleanup_checkpoint import validate_recipe, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Production preparation changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-training-preparation-plan-v1", "Different preparation")
    comparison, execution = (read(plan[k]["path"]) for k in ("comparison", "comparison_execution"))
    require(comparison["schema"] == "latency58-leader-cleanup-resource-comparison-v1"
            and comparison["status"] == "pass" and comparison["source_bindings_unchanged"]
            and comparison["operational_recipe_exact"] and comparison["all_inputs_and_teacher_targets_exact"]
            and comparison["final_rng_states_exact"] and comparison["updates_per_parent"] == 2
            and not comparison["matched_loss_effect_from_leader_claimed"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == comparison["plan_sha256"],
            "Parent-transfer comparison did not pass")
    verify_inputs(comparison)
    for item in (plan["comparison"], plan["comparison_execution"], comparison["training_plan"],
                 comparison["resource"], comparison["execution"]):
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound production prerequisite")
    resource_plan = read(comparison["training_plan"]["path"])
    validate_recipe(resource_plan)
    verify_inputs(resource_plan)
    resource, resource_execution = (read(comparison[k]["path"]) for k in ("resource", "execution"))
    monitor = read(resource_execution["monitor_result"])
    require(resource_plan["resource_only"] and resource["status"] == "pass" and resource["training_updates_executed"] == 2
            and resource["plan_sha256"] == resource_execution["plan_sha256"] == comparison["training_plan"]["sha256"]
            and resource_execution["actual_exit_code"] == 0 and resource_execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"], "Rehearsal did not close")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "training-preparation.json").exists(), "Preserve preparation")
    before = require_space(resource_plan, 400_000_000)
    candidate = copy.deepcopy(resource_plan)
    candidate.update(resource_only=False, run_dir=str(PHASE / "leader-cleanup-b16-micro4-lr3e5-250"),
                     resource_plan=comparison["training_plan"], full_resource=comparison["resource"],
                     full_resource_execution=comparison["execution"], resource_comparison=plan["comparison"],
                     resource_comparison_execution=plan["comparison_execution"],
                     source_bindings={**resource_plan["source_bindings"], **comparison["source_bindings"],
                                      **plan["source_bindings"], str(args.plan.resolve()): args.plan_sha256})
    require(not Path(candidate["run_dir"]).exists(), "Preserve previous trial")
    validate_recipe(candidate)
    verify_inputs(candidate)
    path = out / "training-plan.json"
    write(path, candidate)
    after = require_space(candidate, 400_000_000)
    write(out / "training-preparation.json", {"schema": "latency58-leader-cleanup-training-preparation-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": candidate["source_bindings"], "source_bindings_unchanged": True,
          "training_plan": {"path": str(path), "sha256": sha(path)}, "maximum_new_production_updates": 250,
          "quality_endpoints": [250], "automatic_continuation": False, "training_updates_executed": 0,
          "checkpoint_and_quality_reserve_bytes": 400_000_000, "quality_selected": False,
          "counted_bytes_before": before, "counted_bytes_after": after})
    print({"status": "pass", "training_plan": str(path), "training_plan_sha256": sha(path)}, flush=True)


if __name__ == "__main__":
    main()
