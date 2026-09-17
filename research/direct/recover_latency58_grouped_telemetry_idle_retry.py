"""Retry idle observation after separately recorded event-worker startup timeout.

Reuse the qualified idle child and shutdown checks without changing the frozen
watchdog or either training attempt. No GPU computation runs in this module.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct import recover_latency58_finalization_idle as idle
from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

SOURCE = PHASE / "branch-grouped-vocal-009"
OUT = PHASE / "branch-grouped-vocal-telemetry-idle-011"
PREVIOUS = PHASE / "branch-grouped-vocal-telemetry-idle-010"
# The reused functions resolve their output directory from their own module.
# Their code, heartbeat duration, telemetry deadline and shutdown checks stay
# byte-identical and are included in this experiment's source bindings.
idle.OUT = OUT


def prepare():
    require(Path.cwd() == ROOT and not OUT.exists()
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Prepare with CUDA hidden on CPU1 and preserve previous observations")
    prior_root, prior_execution = (read(PREVIOUS / name) for name in ("root-execution.json", "execution.json"))
    prior_monitor_path = Path(prior_execution["monitor_result"])
    prior_monitor = read(prior_monitor_path)
    diagnostic = read(PREVIOUS / "persistent-event-query-diagnostic.json")
    require(prior_root["actual_session_id"] == 24701 and prior_root["actual_tool_chunk_id"] == "afef81"
            and prior_root["actual_exit_code"] == prior_execution["actual_exit_code"] == 1
            and prior_root["execution_sha256"] == sha(PREVIOUS / "execution.json")
            and prior_root["monitor_result_sha256"] == sha(prior_monitor_path)
            and prior_execution["source_bindings_unchanged"]
            and prior_monitor["status"] == "blocked_preflight" and prior_monitor["child_pid"] is None
            and prior_monitor["reason"] == "RuntimeError('windows_system telemetry timed out')"
            and prior_monitor["event_worker_close"]["closed"]
            and not Path("/proc", str(prior_monitor["event_worker_close"]["linux_pid"])).exists()
            and diagnostic["status"] == "pass" and len(diagnostic["queries"]) == 6
            and diagnostic["event_worker_close"]["actual_exit_code"] == 0
            and not diagnostic["event_worker_close"]["forced"]
            and all(q["previous_event_record_id"] == q["last_event_record_id"] == 65004
                    and q["duration_seconds"] < 10 for q in diagnostic["queries"]),
            "Prior idle failure or subsequent read-only diagnosis is incomplete")
    verify_inputs(diagnostic)
    plan_path = SOURCE / "plan.json"
    execution_path = SOURCE / "production-stage/execution.json"
    root_path = SOURCE / "production-root-execution.json"
    journal_path = SOURCE / "production-run/metrics.jsonl"
    plan, execution, root = (read(p) for p in (plan_path, execution_path, root_path))
    failed_path = Path(execution["monitor_result"])
    failed = read(failed_path)
    rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
    require(sha(plan_path) == root["plan_sha256"]
            == "4b0c107b30c9e57260121c7149d4068e5997277defe5a344023cfec40b3dac7c"
            and root["actual_session_id"] == 63579 and root["actual_tool_chunk_id"] == "80a6ec"
            and root["actual_exit_code"] == execution["actual_exit_code"] == failed["child_exit_code"] == 1
            and root["status"] == "watchdog_gpu_telemetry_timeout_before_saved_endpoint"
            and root["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and root["root_command_sha256"] == sha(SOURCE / "production-root-command.json")
            and root["execution_sha256"] == sha(execution_path)
            and root["monitor_result_sha256"] == sha(failed_path)
            and root["retained_journal_sha256"] == sha(journal_path)
            and not root["checkpoint_written"] and not root["quiet_interval_completed"]
            and failed["status"] == "stopped_by_watchdog" and failed["supervisor_health"] == "failed"
            and failed["reason"] == "RuntimeError('nvidia_smi telemetry timed out')"
            and failed["latest_completed_step_seen"] == root["completed_training_updates"] == len(rows) == 103
            and [row["step"] for row in rows] == list(range(1, 104))
            and failed["last_event_record_id"] == 65004 and not failed["finalization_started"]
            and not failed["post_exit_quiet_completed"] and failed["identities_unchanged"]
            and failed["event_worker_close"]["closed"]
            and failed["event_worker_close"]["actual_exit_code"] == 0
            and not failed["event_worker_close"]["forced"]
            and not Path("/proc", str(failed["child_pid"])).exists()
            and not Path("/proc", str(failed["event_worker_close"]["linux_pid"])).exists()
            and not (SOURCE / "production-run/checkpoint").exists()
            and not (SOURCE / "production-run/checkpoint.pending").exists(),
            "Retained unsaved failure or process settlement differs")
    # Idle touches no model/data dependency. Training revalidates the full plan
    # before any later GPU work; this observation binds its direct inputs only.
    control, control_execution = (read(idle.CONTROL / (n + ".json")) for n in ("result", "execution"))
    verify_inputs(control)
    require(control["status"] == "pass" and control["source_bindings_unchanged"]
            and len(control["cases"]) == 13 and all(c["status"] == "pass" for c in control["cases"])
            and control["all_actual_cpu_children_reaped"] and control["host_telemetry_simulated"]
            and control_execution["actual_exit_code"] == 0 and control_execution["source_bindings_unchanged"]
            and control_execution["result_sha256"] == sha(idle.CONTROL / "result.json")
            and control_execution["plan_sha256"] == control["plan_sha256"] == sha(idle.CONTROL / "plan.json")
            and sha(idle.WATCHDOG) == idle.WATCHDOG_SHA, "Finalization monitor qualification differs")
    paths = [Path(__file__).resolve(), Path(idle.__file__).resolve(), plan_path, execution_path,
             root_path, SOURCE / "production-root-command.json", journal_path, failed_path, idle.WATCHDOG]
    paths.extend(idle.CONTROL / (n + ".json") for n in ("plan", "result", "execution"))
    for artifact in failed["artifacts"].values():
        require(sha(artifact["path"]) == artifact["sha256"], "Failed observation artifact changed")
        paths.append(Path(artifact["path"]))
    paths.extend(PREVIOUS / name for name in ("root-execution.json", "execution.json",
        "persistent-event-query-diagnostic.json", "post-preflight-event-query.json"))
    paths.append(prior_monitor_path)
    paths.append(ROOT / "research/direct/recover_latency58_grouped_telemetry_idle.py")
    bindings = dict(control["source_bindings"])
    for path, digest in {**control["source_bindings"], **{str(p): sha(p) for p in paths}}.items():
        require(path not in bindings or bindings[path] == digest, "Prerequisite source conflict")
        bindings[path] = digest
    inputs = {"schema": "latency58-grouped-telemetry-idle-v1", "source_bindings": bindings,
              "failed_monitor": str(failed_path), "previous_event_record_id": failed["last_event_record_id"],
              "watchdog_source": str(idle.WATCHDOG), "output_directory": str(OUT), "expected_final_step": 15,
              "deliberate_finalization_delay_seconds": 65, "storage_budget": plan["storage_budget"],
              "budget_before": budget_snapshot(plan["storage_budget"]), "gpu_workload_started": False,
              "original_training_monitor_successful": False, "host_stability_proven": False}
    verify_inputs(inputs)
    OUT.mkdir()
    write(OUT / "inputs.json", inputs)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "cwd": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            "progress_path": str(OUT / "metrics.jsonl"), "expected_final_step": 15,
            "argv": [PYTHON, "-u", "-m", "research.direct.recover_latency58_grouped_telemetry_idle_retry",
                     "--child-inputs-sha256", sha(OUT / "inputs.json")]}
    write(OUT / "watchdog-spec.json", spec)
    monitor_out = idle.WATCHDOG.parent / OUT.name
    require(not monitor_out.exists(), "Preserve previous monitor output")
    argv = [PYTHON, str(idle.WATCHDOG), "--launch-spec", str(OUT / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(OUT / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", "240", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60",
            "--finalization-timeout-seconds", "180", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(OUT / "command.json", {"argv": argv, "monitor_result": str(monitor_out / "result.json"),
                                "inputs_sha256": sha(OUT / "inputs.json")})
    print(json.dumps({"status": "prepared", "inputs_sha256": sha(OUT / "inputs.json"),
                      "command_sha256": sha(OUT / "command.json"), "gpu_workload_started": False,
                      "budget_before": inputs["budget_before"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--child-inputs-sha256")
    group.add_argument("--run-command-sha256")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    elif args.child_inputs_sha256:
        idle.child(args.child_inputs_sha256)
    else:
        idle.run(args.run_command_sha256)
