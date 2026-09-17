"""Retry the qualified B4 trajectory after a closed Windows telemetry timeout."""
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    old = PHASE / "branch-long-context-003"
    original_path = old / "plan.json"
    original = read(original_path)
    verify_inputs(original)
    stopped = read(old / "root-execution.json")
    execution = read(old / "production-stage/execution.json")
    failed_path = Path(execution["monitor_result"])
    failed = read(failed_path)
    require(stopped["actual_exit_code"] == execution["actual_exit_code"] == failed["child_exit_code"] == 1
            and stopped["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and stopped["training_updates"] == 0 and not stopped["checkpoint_written"]
            and (old / "production-run/metrics.jsonl").stat().st_size == 0
            and not (old / "production-run/checkpoint").exists()
            and failed["status"] == "stopped_by_watchdog"
            and failed["reason"] == "RuntimeError('windows_system telemetry timed out')"
            and not Path("/proc", str(failed["child_pid"])).exists(),
            "Require the closed telemetry-stopped child before any optimizer update")
    resource = read(old / "resource-run/result.json")
    resource_execution = read(old / "resource-stage/execution.json")
    resource_root = read(old / "resource-stage/root-execution.json")
    resource_monitor = Path(resource_execution["monitor_result"])
    resource_terminal = read(resource_monitor)
    require(resource["status"] == "pass" and resource["updates"] == resource["ema_updates"] == 2
            and not resource["checkpoint_written"] and resource["source_bindings_unchanged"]
            and resource["plan_sha256"] == sha(original_path)
            and resource_execution["actual_exit_code"] == resource_root["actual_exit_code"] == 0
            and resource_execution["source_bindings_unchanged"] and resource_root["source_bindings_unchanged"]
            and resource_terminal["status"] == resource_terminal["supervisor_health"] == "pass"
            and resource_terminal["child_exit_code"] == 0 and resource_terminal["post_exit_quiet_completed"],
            "Original B4 resource qualification did not close successfully")
    recovery = PHASE / "branch-long-context-telemetry-recovery-004"
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
            and recovered_child["status"] == "pass" and recovered_child["event_continuity_passed"]
            and recovered_child["source_bindings_unchanged"] and not recovered_child["gpu_workload_started"]
            and recovery_inputs["previous_event_record_id"] == failed["last_event_record_id"],
            "Require completed monitored event continuity after the telemetry timeout")
    paths = [Path(__file__).resolve(), original_path, old / "root-execution.json", failed_path,
             old / "production-stage/execution.json", old / "production-run/metrics.jsonl",
             old / "root-command.json", old / "production-prefix-independent-check-execution.json",
             old / "resource-run/result.json", old / "resource-run/branch-long-context-gpu-parity.json",
             old / "resource-stage/execution.json", old / "resource-stage/root-execution.json", resource_monitor,
             ROOT / "research/direct/observe_latency58_branch_long_context_b4_retry_prefix.py",
             ROOT / "research/direct/report_latency58_branch_long_context_b4_retry.py", recovered_monitor_path]
    paths.extend(recovery / name for name in ("execution.json", "root-execution.json", "child-result.json",
                 "event-continuity.json", "inputs.json", "watchdog-spec.json", "command.json"))
    for name in ("execution.json", "root-execution.json"):
        path = PHASE / "branch-long-context-preparation-stage-003" / name
        previous = read(path)
        require(previous["actual_exit_code"] == 0 and previous["source_bindings_unchanged"]
                and not previous.get("timed_out", False), "Original B4 preparation did not close")
        paths.append(path)
    out = PHASE / "branch-long-context-004"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve all earlier attempts")
    counted = require_space(original, 650_000_000)
    plan = {**original, "name": out.name, "output_directory": str(out),
            "retry_of": str(original_path), "retry_reason": "Windows system telemetry query timed out before the first production optimizer update",
            "previous_execution_for_resource": str(recovery / "execution.json"),
            "independent_observation_start": "After two completed production updates; retain unchanged supervision during startup",
            "source_bindings": {**original["source_bindings"], **recovery_inputs["source_bindings"],
                                **{str(path): sha(path) for path in paths}}}
    require(plan["config"] == original["config"] and plan["ema"] == original["ema"]
            and plan["config"]["microbatch_size"] == plan["accumulation_steps"] == 4
            and plan["config"]["batch_size"] == 16 and plan["scored_samples"] == 88320
            and plan["parent_checkpoint"] == original["parent_checkpoint"]
            and plan["qualified_data_prefix"] == original["qualified_data_prefix"]
            and plan["logical_batch_loss"] == original["logical_batch_loss"], "Retry changed the qualified recipe")
    verify_inputs(plan)
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "original_plan": str(original_path),
          "original_plan_sha256": sha(original_path), "learning_and_data_recipe_unchanged": True,
          "trainer_and_resource_checker_sources_unchanged": True, "qualification_reused": True,
          "cpu_qualification_repeated": False, "gpu_rehearsal_required": True,
          "parent_full_sdr_db": plan["parent_full_sdr_db"], "failed_production_updates": 0,
          "source_bindings_unchanged": True, "supervision_limits_unchanged": True,
          "forecast_including_outside_and_run": counted + 650_000_000 + plan["outside_roots_reservation_bytes"],
          "artifact_cap_bytes": 90_000_000_000, "monitored_telemetry_recovery": str(recovery / "execution.json"),
          "limitation": "A completed recovery checks current continuity, not future host stability. Deferring the independent source audit reduces simultaneous startup work but does not establish the cause of the timeout."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"status": "pass", "plan": str(out / "plan.json"), "plan_sha256": sha(out / "plan.json"),
                      "learning_recipe_unchanged": True, "original_attempt_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
