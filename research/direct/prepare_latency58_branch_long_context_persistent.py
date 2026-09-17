"""Preserve the two-second B4 recipe while reusing a qualified, deadline-bound Windows event worker."""
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    old = PHASE / "branch-long-context-005"
    original_path = old / "plan.json"
    original = read(original_path)
    verify_inputs(original)
    execution = read(old / "resource-stage/execution.json")
    closed = read(old / "resource-stage/root-execution.json")
    failed_path = Path(execution["monitor_result"])
    failed = read(failed_path)
    require(execution["actual_exit_code"] == closed["actual_exit_code"] == failed["child_exit_code"] == 1
            and execution["source_bindings_unchanged"] and closed["source_bindings_unchanged"]
            and closed["training_updates"] == 0 and not closed["checkpoint_written"]
            and (old / "resource-run/metrics.jsonl").stat().st_size == 0
            and not (old / "production-run").exists()
            and failed["reason"] == "RuntimeError('windows_system telemetry timed out')"
            and not Path("/proc", str(failed["child_pid"])).exists(), "Require the closed resource telemetry failure")
    qualification = PHASE / "branch-long-context-persistent-monitor-001"
    qualified = read(qualification / "result.json")
    qualify_root = read(qualification / "root-execution.json")
    verify_inputs(qualified)
    require(qualified["status"] == "pass" and qualified["source_bindings_unchanged"]
            and qualify_root["actual_exit_code"] == 0 and qualify_root["source_bindings_unchanged"]
            and qualified["original_and_persistent_event_payloads_match"]
            and qualified["sentinel_coverage_fault_detection_and_gpu_control_loop_unchanged"]
            and qualified["runtime_deadline_seconds"] == 10
            and len(qualified["fault_cases"]) == 3 and all(case["status"] == "pass" for case in qualified["fault_cases"])
            and all(not row["SameInstanceStillRunning"] for row in qualified["windows_worker_closure_confirmed"]),
            "Persistent event transport is not qualified")
    recovery = PHASE / "branch-long-context-telemetry-recovery-006"
    recovered_execution = read(recovery / "execution.json")
    recovered_root = read(recovery / "root-execution.json")
    recovered_monitor_path = Path(recovered_execution["monitor_result"])
    recovered_monitor = read(recovered_monitor_path)
    recovered_child = read(recovery / "child-result.json")
    recovery_inputs = read(recovery / "inputs.json")
    verify_inputs(recovery_inputs)
    require(recovered_execution["actual_exit_code"] == recovered_root["actual_exit_code"] == 0
            and recovered_execution["source_bindings_unchanged"] and recovered_root["source_bindings_unchanged"]
            and recovered_monitor["status"] == recovered_monitor["supervisor_health"] == "pass"
            and recovered_monitor["child_exit_code"] == 0 and recovered_monitor["post_exit_quiet_completed"]
            and recovered_monitor["event_worker_close"]["closed"] and not recovered_monitor["event_worker_close"]["forced"]
            and recovered_monitor["event_worker_close"]["actual_exit_code"] == 0
            and recovered_child["status"] == "pass" and recovered_child["event_continuity_passed"]
            and recovered_child["source_bindings_unchanged"] and not recovered_child["gpu_workload_started"]
            and recovery_inputs["previous_event_record_id"] == failed["last_event_record_id"],
            "Require closed traced idle monitoring and complete event continuity")
    watchdog = Path(recovery_inputs["watchdog_source"])
    require(watchdog.name == "watch_gpu_process_persistent.py"
            and watchdog.with_name("watch_gpu_process_trace.py") == Path(original["watchdog_source"]),
            "Persistent monitor lineage differs")
    paths = [Path(__file__).resolve(), original_path, old / "resource-stage/root-execution.json",
             old / "resource-stage/execution.json", old / "resource-run/metrics.jsonl", failed_path,
             old / "resource-run/branch-long-context-gpu-parity.json", watchdog, recovered_monitor_path,
             qualification / "result.json", qualification / "root-execution.json",
             ROOT / "research/direct/observe_latency58_branch_long_context_persistent_prefix.py",
             ROOT / "research/direct/report_latency58_branch_long_context_persistent.py"]
    paths.extend(recovery / name for name in ("execution.json", "root-execution.json", "child-result.json",
                 "event-continuity.json", "inputs.json", "watchdog-spec.json", "command.json"))
    for name in ("execution.json", "root-execution.json"):
        path = PHASE / "branch-long-context-preparation-stage-005" / name
        previous = read(path)
        require(previous["actual_exit_code"] == 0 and previous["source_bindings_unchanged"]
                and not previous.get("timed_out", False), "Original preparation did not close")
        paths.append(path)
    out = PHASE / "branch-long-context-006"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve all previous attempts")
    counted = require_space(original, 650_000_000)
    plan = {**original, "name": out.name, "output_directory": str(out), "retry_of": str(original_path),
            "retry_reason": "Queries timed out without an entry marker; reuse a qualified PowerShell worker while retaining per-query deadlines, complete event coverage and all GPU stop conditions",
            "event_query_transport": {"kind": "persistent_owned_powershell", "query_deadline_seconds": 10,
                "helper_sha256": sha(ROOT / "research/direct/latency58_windows_event_transport.py"),
                "automatic_restart_after_error": False, "normal_worker_closure_required": True},
            "watchdog_source": str(watchdog), "previous_execution_for_resource": str(recovery / "execution.json"),
            "source_bindings": {**original["source_bindings"], **qualified["source_bindings"],
                                **recovery_inputs["source_bindings"], **{str(path): sha(path) for path in paths}}}
    require(plan["config"] == original["config"] and plan["ema"] == original["ema"]
            and plan["logical_batch_loss"] == original["logical_batch_loss"]
            and plan["qualified_data_prefix"] == original["qualified_data_prefix"]
            and plan["parent_checkpoint"] == original["parent_checkpoint"]
            and plan["config"]["batch_size"] == 16 and plan["config"]["microbatch_size"] == plan["accumulation_steps"] == 4
            and plan["warmup_samples"] == 88064 and plan["scored_samples"] == 88320, "Learning recipe changed")
    verify_inputs(plan)
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "original_plan_sha256": sha(original_path),
          "learning_and_data_recipe_unchanged": True, "trainer_and_gpu_checker_sources_unchanged": True,
          "event_coverage_gpu_control_loop_and_fault_detection_unchanged": True, "all_supervision_limits_unchanged": True,
          "event_query_transport": plan["event_query_transport"],
          "full_gpu_resource_rehearsal_required": True, "source_bindings_unchanged": True,
          "parent_full_sdr_db": plan["parent_full_sdr_db"], "failed_attempt_optimizer_updates": 0,
          "forecast_including_outside_and_run": counted + 650_000_000 + plan["outside_roots_reservation_bytes"],
          "artifact_cap_bytes": 90_000_000_000,
          "limitation": "Persistent transport removes repeated process startup but does not prove the cause of past stalls, future host stability or model quality. The full GPU rehearsal remains required."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"status": "pass", "plan": str(out / "plan.json"), "plan_sha256": sha(out / "plan.json"),
                      "learning_recipe_unchanged": True, "persistent_event_query_transport": True}), flush=True)


if __name__ == "__main__":
    main()
