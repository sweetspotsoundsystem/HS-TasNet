"""Freeze the three original 250-update pilots after actual matched GPU rehearsals."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_vocal_focus_checkpoint import require_space, validate_recipe


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Preparation plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-vocal-focus-training-preparation-plan-v1", "Unexpected preparation")
    for item in (plan["protocol"], plan["pairing"], plan["pairing_execution"], plan["retirement"]):
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound training preparation prerequisite")
    protocol = read(plan["protocol"]["path"])
    verify_inputs(protocol)
    require(protocol["schema"] == "latency58-vocal-focus-protocol-v1"
            and protocol["arms"] == ["original", "focused", "focused_mixer"]
            and protocol["maximum_production_updates"] == 750 and protocol["quality_endpoints"] == [250]
            and not protocol["automatic_continuation"], "Original pilot scope changed")
    pairing, paired_execution = read(plan["pairing"]["path"]), read(plan["pairing_execution"]["path"])
    require(pairing["schema"] == "latency58-vocal-focus-resource-pairing-v1" and pairing["status"] == "pass"
            and pairing["source_bindings_unchanged"] and pairing["updates_per_arm"] == 2
            and pairing["microbatches_per_arm"] == 8 and pairing["examples_per_arm"] == 32
            and pairing["all_three_pristine_and_original_draws_exact"]
            and pairing["focused_inputs_and_teacher_targets_exact"] and pairing["initial_inherited_gradients_exact"]
            and pairing["first_update_four_microbatch_losses_exact"]
            and paired_execution["actual_exit_code"] == 0 and not paired_execution["timed_out"]
            and paired_execution["source_bindings_unchanged"]
            and paired_execution["plan_sha256"] == pairing["plan_sha256"], "Matched GPU rehearsal did not pass")
    verify_inputs(pairing)
    retirement = read(plan["retirement"]["path"])
    require(retirement["schema"] == "latency58-closed-history-optimizer-retirement-v1"
            and retirement["status"] == "complete" and retirement["source_bindings_unchanged"]
            and retirement["active_training_inputs_unchanged"] and retirement["freed_bytes"] > 400_000_000
            and all(not Path(p).exists() for p in retirement["retired_paths"])
            and all(sha(p) == s for p, s in retirement["protected_files"].items()), "History retirement is incomplete")
    before = require_space(protocol, 650_000_000)
    prep = Path(plan["protocol"]["path"]).parent
    require(prep.is_relative_to(PHASE) and not (prep / "training-preparation.json").exists(), "Preserve preparation")
    binds = {**plan["source_bindings"], **pairing["source_bindings"], str(args.plan): args.plan_sha256,
             str(Path(__file__).resolve()): sha(__file__)}
    prepared = []
    for arm in protocol["arms"]:
        resource_plan_path = prep / (arm + "-resource-plan.json")
        resource_plan = read(resource_plan_path)
        validate_recipe(resource_plan)
        verify_inputs(resource_plan)
        stage = PHASE / ("vocal-focus-" + arm.replace("_", "-") + "-resource-001")
        resource_path = Path(resource_plan["run_dir"]) / "resource.json"
        resource, execution = read(resource_path), read(stage / "execution.json")
        monitor = read(execution["monitor_result"])
        require(resource_plan["resource_only"] and resource_plan["arm"] == arm
                and resource_plan["matched_protocol"] == plan["protocol"]
                and pairing["resource_results"][arm] == binding(resource_path)
                and resource["status"] == "pass" and resource["source_bindings_unchanged"]
                and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
                and resource["config"] == protocol["config"] == resource_plan["config"]
                and resource["initial_model_state_sha256"] == resource_plan["initialized_model_state_sha256"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and execution["plan_sha256"] == resource["plan_sha256"] == sha(resource_plan_path)
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                "Resource recipe or actual execution differs")
        training = copy.deepcopy(resource_plan)
        training.update(resource_only=False,
                        run_dir=str(PHASE / ("vocal-focus-" + arm.replace("_", "-") + "-b16-micro4-lr3e5-250")),
                        resource_plan=binding(resource_plan_path), full_resource=binding(resource_path),
                        full_resource_execution=binding(stage / "execution.json"),
                        resource_pairing=copy.deepcopy(plan["pairing"]),
                        resource_pairing_execution=copy.deepcopy(plan["pairing_execution"]),
                        source_bindings={**resource_plan["source_bindings"], **binds})
        for path in (resource_plan_path, resource_path, stage / "execution.json", Path(execution["monitor_result"])):
            training["source_bindings"][str(path)] = sha(path)
        path = prep / (arm + "-training-plan.json")
        require(not path.exists() and not Path(training["run_dir"]).exists(), "Preserve existing pilot plan or weights")
        validate_recipe(training)
        verify_inputs(training)
        prepared.append((path, training))
    verify_inputs(plan)
    for path, training in prepared:
        write(path, training)
        print({"plan": str(path), "sha256": sha(path), "arm": training["arm"], "updates": 250}, flush=True)
    write(prep / "training-preparation.json", {
        "schema": "latency58-vocal-focus-training-preparation-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "protocol": plan["protocol"],
        "source_bindings": binds, "source_bindings_unchanged": True,
        "plans": {training["arm"]: binding(path) for path, training in prepared},
        "pilot_updates_per_arm": 250, "maximum_production_updates": 750,
        "automatic_continuation": False, "training_updates_executed": 0,
        "counted_bytes_before": before, "counted_bytes_after": require_space(protocol, 0),
        "combined_reservation_bytes": 650_000_000, "quality_selected": False,
        "reservation_requires": "Retire each terminal pilot Adam only after its independent audit and full quality review close the arm; retain all inference generations."})


if __name__ == "__main__":
    main()
