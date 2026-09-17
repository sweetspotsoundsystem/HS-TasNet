"""Freeze resource or production plans for the two matched student histories."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("resource", "training"), required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    prep = PHASE / "sdr-history-prep-001"
    protocol_path = prep / "protocol.json"
    require(sha(protocol_path) == args.protocol_sha256, "Frozen matched protocol changed")
    protocol = read(protocol_path)
    verify_inputs(protocol)
    parent_plan = read(protocol["parent_plan"]["path"])
    require(sha(protocol["parent_plan"]["path"]) == protocol["parent_plan"]["sha256"], "Parent plan changed")
    verify_inputs(parent_plan)
    old_path = Path(parent_plan["parent"]["training_plan"]["path"])
    old = read(old_path)
    functional_dir = PHASE / "sdr-history-functional-001"
    functional = read(functional_dir / "result.json")
    executed = read(functional_dir / "functional-execution.json")
    require(functional["status"] == "pass" and functional["source_bindings_unchanged"]
            and functional["all_21_parameter_gradients_match"]
            and functional["model_state_sha256"] == protocol["parent_model_state_sha256"]
            and functional["teacher_model_state_sha256"] == protocol["teacher_model_state_sha256"]
            and executed["actual_exit_code"] == 0 and not executed["timed_out"]
            and executed["source_bindings_unchanged"]
            and executed["plan_sha256"] == functional["plan_sha256"] == sha(functional_dir / "plan.json"),
            "Exact-parent functional check did not pass")
    verify_inputs(functional)
    require_space(protocol, protocol["combined_additional_artifact_reservation_bytes"])
    sources = ("latency58_drum_accum_parent.py", "latency58_matched_history.py",
               "check_latency58_matched_history.py", "latency58_sdr_history_checkpoint.py",
               "train_latency58_sdr_history.py", "run_latency58_sdr_history_stage.py",
               "audit_latency58_sdr_history.py", "prepare_latency58_sdr_history.py")
    binds = {**protocol["source_bindings"], **functional["source_bindings"]}
    paths = [protocol_path, *(ROOT / "research/direct" / name for name in sources),
             *(functional_dir / name for name in ("plan.json", "result.json", "functional-execution.json"))]
    data_dir = PHASE / "sdr-long-context-data-functional-001"
    data_plan, data_result, data_execution = (read(data_dir / name) for name in ("plan.json", "result.json", "execution.json"))
    require(data_result["status"] == "pass" and data_result["source_bindings_unchanged"]
            and data_execution["actual_exit_code"] == 0 and not data_execution["timed_out"]
            and data_execution["source_bindings_unchanged"]
            and data_execution["plan_sha256"] == data_result["plan_sha256"] == sha(data_dir / "plan.json"),
            "Long-crop data qualification did not pass")
    verify_inputs(data_plan)
    binds.update(data_plan["source_bindings"])
    paths += [data_dir / name for name in ("plan.json", "result.json", "execution.json")]
    for path in paths:
        binds[str(path)] = sha(path)
    base = {key: copy.deepcopy(old[key]) for key in (
        "environment", "torch_version", "precision_policy", "helper_source", "watchdog_source",
        "manifest_sha256", "geometry", "teacher_kind")}
    for key in ("config", "teacher_history_samples", "scored_samples", "teacher", "teacher_weight",
                "teacher_model_state_sha256", "drum_weight", "objective_version", "accumulation_version",
                "history_version", "microbatch_size", "accumulation_steps", "carry_state",
                "optimizer_initialization", "quality_endpoints", "continuation_rules", "counted_roots", "stop_counted_bytes"):
        base[key] = copy.deepcopy(protocol[key])
    base.update(schema="latency58-sdr-history-training-v1", resource_only=args.phase == "resource",
                parent_prefix="sdr-drum-accum-1000", parent=copy.deepcopy(parent_plan["parent"]),
                source_bindings=binds, matched_protocol=binding(protocol_path),
                functional_proofs=copy.deepcopy(old["functional_proofs"]) + [
                    {"result": str(data_dir / "result.json"), "execution": str(data_dir / "execution.json")}],
                accumulation_functional=binding(functional_dir / "result.json"),
                accumulation_functional_execution=binding(functional_dir / "functional-execution.json"))
    resources = {}
    if args.phase == "training":
        for arm in ("short", "long"):
            resource_plan_path = prep / (arm + "-resource-plan.json")
            resource_plan = read(resource_plan_path)
            verify_inputs(resource_plan)
            stage_dir = PHASE / ("sdr-history-" + arm + "-resource-001")
            resource_path = Path(resource_plan["run_dir"]) / "resource.json"
            resource, execution = read(resource_path), read(stage_dir / "execution.json")
            monitor = read(execution["monitor_result"])
            require(resource["status"] == "pass" and resource["source_bindings_unchanged"]
                    and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
                    and resource["initial_model_state_sha256"] == protocol["parent_model_state_sha256"]
                    and resource["teacher_unchanged"] and resource["fixed_buffers_unchanged"]
                    and resource["all_parameter_gradients_present"]
                    and resource["plan_sha256"] == execution["plan_sha256"] == sha(resource_plan_path)
                    and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                    and monitor["status"] == monitor["supervisor_health"] == "pass"
                    and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                    "A full GPU history rehearsal did not pass")
            resources[arm] = resource
            for path in (resource_plan_path, resource_path, Path(execution["monitor_result"]),
                         *(stage_dir / name for name in ("stage.json", "watchdog-spec.json", "command.json",
                                                       "execution.json", "resource-qualification.json"))):
                binds[str(path)] = sha(path)
        matches = []
        for left, right in zip(resources["short"]["matching_production_updates"],
                               resources["long"]["matching_production_updates"], strict=True):
            keys = ("step", "lr", "first_sample_index", "next_sample_index", "augmented_batch_sha256",
                    "teacher_targets_sha256", "deranged_examples", "teacher_history_samples", "scored_samples")
            require(all(left[k] == right[k] for k in keys), "Resource arms differ in data, augmentation or teacher targets")
            for a, b in zip(left["microbatches"], right["microbatches"], strict=True):
                require(all(a[k] == b[k] for k in ("micro_index", "first_sample_index", "next_sample_index", "batch_size",
                                                 "augmented_batch_sha256", "teacher_targets_sha256", "deranged_examples")),
                        "Resource microbatches are not matched")
            matches.append({k: left[k] for k in keys})
        match_path = prep / "resource-match.json"
        require(not match_path.exists(), "Preserve resource match")
        write(match_path, {"schema": "latency58-matched-history-resource-match-v1", "status": "pass",
                           "protocol_sha256": args.protocol_sha256, "source_bindings": copy.deepcopy(binds),
                           "source_bindings_unchanged": True, "matched_updates": matches,
                           "matched_microbatches": 8, "all_augmentation_and_teacher_targets_exact": True,
                           "short_final_state_sha256": resources["short"]["final_model_state_sha256"],
                           "long_final_state_sha256": resources["long"]["final_model_state_sha256"],
                           "training_updates_per_arm": 2, "production_updates_executed": 0,
                           "quality_selected": False})
        binds[str(match_path)] = sha(match_path)
    prepared = []
    for arm, warm in (("short", 88064), ("long", 352256)):
        plan = copy.deepcopy(base)
        plan["source_bindings"] = copy.deepcopy(binds)
        plan["warmup_samples"] = warm
        plan["run_dir"] = str(PHASE / ("sdr-history-" + arm + ("-resource-run-001" if args.phase == "resource" else "-b16-micro4-lr3e5-500")))
        if args.phase == "training":
            plan.update(full_resource=binding(PHASE / ("sdr-history-" + arm + "-resource-run-001/resource.json")),
                        full_resource_execution=binding(PHASE / ("sdr-history-" + arm + "-resource-001/execution.json")),
                        resource_plan=binding(prep / (arm + "-resource-plan.json")),
                        matched_resource=binding(prep / "resource-match.json"))
        path = prep / (arm + "-" + args.phase + "-plan.json")
        require(not path.exists() and not Path(plan["run_dir"]).exists(), "Preserve existing arm or plan")
        verify_inputs(plan)
        prepared.append((path, plan))
    for path, plan in prepared:
        write(path, plan)
        print({"plan": str(path), "sha256": sha(path), "history_samples": plan["warmup_samples"],
               "resource_only": plan["resource_only"], "bindings": len(plan["source_bindings"])}, flush=True)


if __name__ == "__main__":
    main()
