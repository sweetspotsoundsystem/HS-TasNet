"""Authenticate the allocated-log supervisor and retain all closure requirements."""
from pathlib import Path
from research.direct.run_latency58_quality import ROOT, read, require, sha
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.recover_latency58_nvml_guard_idle import require_monitor_qualification as require_original_qualification
from research.direct.run_latency58_weighted_vocal import require_monitor_closed as require_original_monitor_closed

WATCHDOG = ROOT / "research/direct/watch_latency58_four_second.py"
WATCHDOG_SHA = "5c8d010a361ab14fc768c2dc56021d56b7c92c8219fead6a21f77a657420f217"
CONTROL = ARTIFACT_ROOT / "monitor-path-cpu-001"


def require_monitor_qualification():
    bindings = require_original_qualification()
    plan, result, execution = (read(CONTROL / name) for name in ("plan.json", "result.json", "root-execution.json"))
    require(sha(WATCHDOG) == WATCHDOG_SHA and result["status"] == "pass" and result["source_bindings_unchanged"]
            and type(execution["actual_exit_code"]) is int and execution["actual_exit_code"] == 0
            and execution["timed_out"] is False and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == result["plan_sha256"] == sha(CONTROL / "plan.json")
            and execution["result_sha256"] == sha(CONTROL / "result.json")
            and result["path_only_source_proof"]["adapted"]["sha256"] == WATCHDOG_SHA
            and result["path_only_source_proof"]["entire_module_ast_identical_except_two_path_expressions"]
            and result["path_only_source_proof"]["global_lock_directory_and_all_transport_bindings_identical"]
            and len(result["cases"]) == 22 and all(row["status"] == "pass" for row in result["cases"])
            and len(result["rejected_output_paths"]) == 3
            and all(row["rejected_before_workers_or_child"] for row in result["rejected_output_paths"])
            and result["all_actual_cpu_children_reaped"] and result["host_telemetry_simulated"]
            and not result["gpu_workload_started"], "Allocated monitor path qualification is incomplete")
    verify_inputs(plan)
    bindings.update(plan["source_bindings"])
    bindings.update({str(CONTROL / name): sha(CONTROL / name) for name in ("plan.json", "result.json", "root-execution.json")})
    return bindings


def require_monitor_closed(execution, monitor, *, final_step=None):
    require(execution['actual_exit_code'] == 0 and execution['source_bindings_unchanged'] and (monitor['status'] == monitor['supervisor_health'] == 'pass') and (monitor['child_exit_code'] == 0) and monitor['post_exit_quiet_completed'] and monitor['identities_unchanged'] and (monitor['source_sha256'] == monitor['source_sha256_after'] == WATCHDOG_SHA) and monitor['finalization_started'] and (monitor['finalization_timeout_seconds'] == 180), 'The preceding GPU supervisor did not complete successfully')
    for name in ('event_worker_close', 'gpu_worker_close'):
        value = monitor[name]
        require(value['closed'] and value['actual_exit_code'] == 0 and (not value['forced']), 'A telemetry worker did not close normally')
    require(monitor['gpu_worker_close']['identities_unchanged'], 'GPU telemetry identities changed')
    if final_step is not None:
        require(monitor['expected_final_step'] == monitor['latest_completed_step_seen'] == final_step, 'GPU monitor did not observe the requested endpoint')
