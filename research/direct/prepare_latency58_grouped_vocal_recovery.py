"""Prepare the same grouped pilot with qualified periodic recovery saves."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_grouped_vocal_recovery import validate_recipe
from research.direct.run_latency58_grouped_vocal_recovery import require_recovery_check
from research.direct.latency58_grouped_vocal_recovery import policy
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.run_latency58_paired_vocal_views import binding, merge_bindings

SOURCE = PHASE / "branch-grouped-vocal-009"
IDLE = PHASE / "branch-grouped-vocal-telemetry-idle-011"
CPU = PHASE / "grouped-vocal-recovery-cpu-001"
OUT = PHASE / "branch-grouped-vocal-010"


def prepare():
    require(Path.cwd() == ROOT and not OUT.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Prepare with CUDA hidden on CPU1 and preserve earlier attempts")
    source_path = SOURCE / "plan.json"
    source = read(source_path)
    failed_root, failed_execution = (read(SOURCE / name) for name in
        ("production-root-execution.json", "production-stage/execution.json"))
    failed_monitor_path = Path(failed_execution["monitor_result"])
    failed = read(failed_monitor_path)
    require(sha(source_path) == failed_root["plan_sha256"]
            == "4b0c107b30c9e57260121c7149d4068e5997277defe5a344023cfec40b3dac7c"
            and failed_root["actual_session_id"] == 63579 and failed_root["actual_tool_chunk_id"] == "80a6ec"
            and failed_root["actual_exit_code"] == failed_execution["actual_exit_code"] == failed["child_exit_code"] == 1
            and failed_root["source_bindings_unchanged"] and failed_execution["source_bindings_unchanged"]
            and failed_root["execution_sha256"] == sha(SOURCE / "production-stage/execution.json")
            and failed_root["monitor_result_sha256"] == sha(failed_monitor_path)
            and failed_root["retained_journal_sha256"] == sha(SOURCE / "production-run/metrics.jsonl")
            and failed["reason"] == "RuntimeError('nvidia_smi telemetry timed out')"
            and failed["latest_completed_step_seen"] == failed_root["completed_training_updates"] == 103
            and not failed_root["checkpoint_written"] and not failed["post_exit_quiet_completed"]
            and not Path("/proc", str(failed["child_pid"])).exists()
            and not (SOURCE / "production-run/checkpoint").exists()
            and not (SOURCE / "production-run/checkpoint.pending").exists(),
            "Original unsaved failure changed")
    idle_inputs, idle_result, idle_execution, idle_root = (read(IDLE / name) for name in
        ("inputs.json", "result.json", "execution.json", "root-execution.json"))
    idle_monitor_path = Path(idle_execution["monitor_result"])
    idle_monitor = read(idle_monitor_path)
    verify_inputs(idle_inputs)
    require(idle_root["actual_exit_code"] == idle_execution["actual_exit_code"] == idle_monitor["child_exit_code"] == 0
            and idle_root["actual_session_id"] == 13246 and idle_root["actual_tool_chunk_id"] == "e10414"
            and idle_root["execution_sha256"] == sha(IDLE / "execution.json")
            and idle_root["result_sha256"] == sha(IDLE / "result.json")
            and idle_root["monitor_result_sha256"] == sha(idle_monitor_path)
            and idle_execution["source_bindings_unchanged"] and idle_result["source_bindings_unchanged"]
            and idle_result["status"] == idle_monitor["status"] == idle_monitor["supervisor_health"] == "pass"
            and idle_monitor["post_exit_quiet_completed"] and idle_monitor["identities_unchanged"]
            and idle_monitor["finalization_started"] and 60 < idle_monitor["finalization_elapsed_seconds"] < 180
            and idle_monitor["event_worker_close"]["closed"]
            and idle_monitor["event_worker_close"]["actual_exit_code"] == 0
            and not idle_monitor["event_worker_close"]["forced"]
            and idle_inputs["previous_event_record_id"] == failed["last_event_record_id"] == 65004
            and not Path("/proc", str(idle_monitor["child_pid"])).exists()
            and not Path("/proc", str(idle_monitor["event_worker_close"]["linux_pid"])).exists(),
            "Host continuity and monitored idle did not finish successfully")
    cpu_plan, cpu_result, cpu_execution, cpu_root = (read(CPU / name) for name in
        ("plan.json", "result.json", "execution.json", "root-execution.json"))
    verify_inputs(cpu_plan)
    require_recovery_check(cpu_result)
    require(cpu_result["gpu_used"] is False and cpu_result["source_bindings_unchanged"]
            and cpu_result["plan_sha256"] == cpu_execution["plan_sha256"] == sha(CPU / "plan.json")
            and cpu_execution["actual_exit_code"] == cpu_root["actual_exit_code"] == 0
            and cpu_root["actual_session_id"] == 64727 and cpu_root["actual_tool_chunk_id"] == "70e696"
            and cpu_execution["source_bindings_unchanged"] and not cpu_execution["timed_out"]
            and cpu_root["execution_sha256"] == sha(CPU / "execution.json")
            and cpu_root["result_sha256"] == cpu_execution["result_sha256"] == sha(CPU / "result.json"),
            "Recovery CPU qualification is incomplete")
    paths = [Path(__file__).resolve(), source_path, failed_monitor_path, idle_monitor_path]
    paths.extend(SOURCE / name for name in ("production-root-command.json", "production-root-execution.json",
        "production-stage/execution.json", "production-run/metrics.jsonl"))
    paths.extend(IDLE / name for name in ("inputs.json", "result.json", "execution.json", "root-execution.json",
        "child-result.json", "event-continuity.json"))
    paths.extend(CPU / name for name in ("plan.json", "result.json", "execution.json", "root-execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in ("train_latency58_grouped_vocal_recovery.py",
        "run_latency58_grouped_vocal_recovery.py", "latency58_grouped_vocal_recovery.py",
        "check_latency58_grouped_vocal_recovery.py", "check_latency58_grouped_vocal_recovery_timing.py"))
    bindings = dict(source["source_bindings"])
    merge_bindings(bindings, idle_inputs["source_bindings"])
    merge_bindings(bindings, cpu_plan["source_bindings"])
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    before = budget_snapshot(source["storage_budget"])
    require(before["headroom_bytes"] > 1_200_000_000,
            "Reserve room for the rolling recovery, atomic replacement and final endpoint")
    outside = (before["external_git_common_bytes"] + source["storage_budget"]["other_outside_allowance_bytes"]
               + source["storage_budget"]["diagnostic_artifact_allowance_bytes"])
    plan = copy.deepcopy(source)
    plan.update(name=OUT.name, output_directory=str(OUT), source_bindings=bindings,
        retry_of=binding(source_path), retry_reason="Replay the unsaved 103-update attempt after successful real-host idle observation, adding complete atomic rolling recovery every 50 updates. Scientific schedule and host guards are unchanged.",
        prior_failed_training=binding(SOURCE / "production-root-execution.json"),
        recovered_host_idle=binding(IDLE / "root-execution.json"),
        recovery_cpu_qualification=binding(CPU / "root-execution.json"), recovery_checkpoint=policy(),
        previous_execution_for_resource=str(IDLE / "execution.json"),
        event_continuity_scope="Fresh resource scan starts at the successful idle monitor, whose child verified all host records from the failed training's last successful poll; both failed attempts remain recorded.",
        budget_before=before, outside_roots_reservation_bytes=outside,
        stop_counted_bytes=90_000_000_000 - outside, prior_unsaved_training_updates=103,
        replay_from_saved_selected_parent=True, resume_checkpoint=None)
    validate_recipe(plan)
    require(all(plan[key] == source[key] for key in ("config", "parent_checkpoint", "parent_model_state_sha256",
        "initialized_model_state_sha256", "ema", "objective_version", "grouped_vocal_loss", "accumulation_policy",
        "qualified_data_prefix", "inference_architecture", "supervision", "runtime_allowance")),
        "Recovery retry changed its scientific recipe or monitoring limits")
    OUT.mkdir(); write(OUT / "plan.json", plan)
    print(json.dumps({"status": "prepared", "plan_sha256": sha(OUT / "plan.json"),
        "recovery_interval_updates": policy()["interval_updates"], "gpu_workload_started": False,
        "budget_before": before}), flush=True)


if __name__ == "__main__":
    prepare()
