"""Preserve the telemetry-stopped attempt and retry its exact qualified training recipe."""
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    old = PHASE / "branch-pitch-ema-001"
    original_path = old / "plan.json"
    original = read(original_path)
    verify_inputs(original)
    stopped = read(old / "root-execution.json")
    execution = read(old / "production-stage/execution.json")
    failed_path = Path(execution["monitor_result"])
    failed = read(failed_path)
    require(stopped["actual_exit_code"] == execution["actual_exit_code"] == 1
            and stopped["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and not stopped["checkpoint_written"] and not (old / "production-run/checkpoint").exists()
            and failed["child_exit_code"] == -9 and failed["status"] == "stopped_by_watchdog"
            and failed["reason"] == "RuntimeError('windows_system telemetry timed out')"
            and not Path("/proc", str(failed["child_pid"])).exists(), "Require the stopped attempt and unchanged inputs")
    recovery = PHASE / "branch-pitch-ema-telemetry-recovery-002"
    recovery_execution = read(recovery / "execution.json")
    recovery_root = read(recovery / "root-execution.json")
    recovered_monitor_path = Path(recovery_execution["monitor_result"])
    recovered_monitor = read(recovered_monitor_path)
    recovered_child = read(recovery / "child-result.json")
    require(recovery_execution["actual_exit_code"] == recovery_root["actual_exit_code"] == 0
            and recovery_execution["source_bindings_unchanged"] and recovery_root["source_bindings_unchanged"]
            and recovered_monitor["status"] == recovered_monitor["supervisor_health"] == "pass"
            and recovered_monitor["child_exit_code"] == 0 and recovered_monitor["post_exit_quiet_completed"]
            and recovered_child["status"] == "pass" and recovered_child["event_continuity_passed"]
            and recovered_child["source_bindings_unchanged"] and not recovered_child["gpu_workload_started"],
            "Require completed monitored event-continuity recovery before retry preparation")
    paths = [Path(__file__).resolve(), original_path, old / "root-execution.json",
             old / "production-stage/execution.json", old / "production-run/metrics.jsonl", failed_path,
             old / "production-prefix-independent-check.json", old / "production-prefix-independent-check-execution.json",
             ROOT / "research/direct/report_latency58_branch_pitch_ema_retry.py",
             ROOT / "research/direct/observe_latency58_branch_pitch_ema_retry_prefix.py", recovered_monitor_path]
    paths.extend(recovery / name for name in ("execution.json", "root-execution.json", "child-result.json",
                 "event-continuity.json", "inputs.json", "watchdog-spec.json", "command.json"))
    recovery_inputs = read(recovery / "inputs.json")
    verify_inputs(recovery_inputs)
    for name in ("execution.json", "root-execution.json"):
        path = PHASE / "branch-pitch-ema-preparation-stage-001" / name
        previous = read(path)
        require(previous["actual_exit_code"] == 0 and previous["source_bindings_unchanged"]
                and not previous.get("timed_out", False), "Original CPU qualification did not close")
        paths.append(path)
    old_prefix = read(old / "production-prefix-independent-check.json")
    require(old_prefix["status"] == "pass" and old_prefix["normalized_fields_equal"],
            "Stopped attempt did not reproduce the qualified two-update trajectory")
    out = PHASE / "branch-pitch-ema-002"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve all earlier training attempts")
    counted = require_space(original, 650_000_000)
    out.mkdir()
    plan = {**original, "name": out.name, "output_directory": str(out),
            "retry_of": str(original_path), "retry_reason": "Windows system telemetry query timed out; no checkpoint saved",
            "previous_execution_for_resource": str(recovery / "execution.json"),
            "source_bindings": {**original["source_bindings"], **recovery_inputs["source_bindings"],
                                **{str(path): sha(path) for path in paths}}}
    require(plan["config"] == original["config"] and plan["ema"] == original["ema"]
            and plan["parent_checkpoint"] == original["parent_checkpoint"]
            and plan["qualified_data_prefix"] == original["qualified_data_prefix"],
            "Retry changed the qualified learning or data recipe")
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "original_plan": str(original_path),
          "original_plan_sha256": sha(original_path), "qualification_reused_without_model_or_recipe_changes": True,
          "cpu_qualification_repeated": False, "gpu_rehearsal_required": True,
          "parent_full_sdr_db": plan["parent_full_sdr_db"], "starting_from_saved_parent": True,
          "stopped_updates_not_added_to_lineage": True, "source_bindings_unchanged": True,
          "forecast_including_outside_and_run": counted + 650_000_000 + plan["outside_roots_reservation_bytes"],
          "artifact_cap_bytes": 90_000_000_000, "monitored_telemetry_recovery": str(recovery / "execution.json")})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"status": "pass", "plan": str(out / "plan.json"), "plan_sha256": sha(out / "plan.json"),
                      "learning_recipe_unchanged": True, "original_attempt_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
