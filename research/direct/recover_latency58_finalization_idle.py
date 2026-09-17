"""Check real host continuity and bounded shutdown after the saved 006 teardown failure.

The owned child only writes idle heartbeats and deliberately waits 65 seconds
after its final heartbeat. It never imports a GPU computation library.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, load_source, continuity
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

SOURCE = PHASE / "branch-long-context-006"
CONTROL = PHASE / "finalization-monitor-check-001"
OUT = PHASE / "branch-long-context-finalization-idle-007"
WATCHDOG = PHASE.parent / "latency11/smoke/gpu-crash-followup/watch_gpu_process_finalization.py"
WATCHDOG_SHA = "58832a30ddd9c38e603273638834546eec4978623cf3d88c25990a6d08be840e"


def prepare():
    require(Path.cwd() == ROOT and not OUT.exists(), "Preserve prior idle observations")
    failed_execution, failed_root = (read(SOURCE / n) for n in ("production-stage/execution.json", "root-execution.json"))
    failed_path = Path(failed_execution["monitor_result"])
    failed = read(failed_path)
    require(failed_execution["actual_exit_code"] == failed_root["actual_exit_code"] == 1
            and failed_execution["source_bindings_unchanged"]
            and failed_root["actual_root_session"] == 27228 and failed_root["actual_tool_output_chunk"] == "1b6049"
            and failed["child_exit_code"] == -15 and failed["latest_completed_step_seen"] == 4000
            and failed["status"] == "stopped_by_watchdog" and failed["supervisor_health"] == "failed"
            and failed["reason"] == "RuntimeError('Owned child stopped reporting completed updates')"
            and not Path("/proc", str(failed["child_pid"])).exists(), "Original failure evidence differs")
    control, execution = (read(CONTROL / (n + ".json")) for n in ("result", "execution"))
    verify_inputs(control)
    require(control["status"] == "pass" and control["source_bindings_unchanged"]
            and len(control["cases"]) == 13 and all(c["status"] == "pass" for c in control["cases"])
            and control["all_actual_cpu_children_reaped"] and control["host_telemetry_simulated"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and execution["result_sha256"] == sha(CONTROL / "result.json")
            and execution["plan_sha256"] == control["plan_sha256"] == sha(CONTROL / "plan.json")
            and sha(WATCHDOG) == WATCHDOG_SHA, "Finalization monitor qualification is incomplete")
    budget_path = PHASE / "branch-gru-int8-post-ci-storage-001.json"
    paths = [Path(__file__).resolve(), SOURCE / "production-stage/execution.json", SOURCE / "root-execution.json",
             failed_path, WATCHDOG, budget_path]
    paths.extend(CONTROL / (n + ".json") for n in ("plan", "result", "execution"))
    bindings = {**control["source_bindings"], **{str(p): sha(p) for p in paths}}
    inputs = {"schema": "latency58-finalization-idle-v1", "source_bindings": bindings,
              "failed_monitor": str(failed_path), "previous_event_record_id": failed["last_event_record_id"],
              "watchdog_source": str(WATCHDOG), "output_directory": str(OUT), "expected_final_step": 15,
              "deliberate_finalization_delay_seconds": 65, "storage_budget": read(budget_path),
              "budget_before": budget_snapshot(read(budget_path)), "gpu_workload_started": False,
              "original_training_monitor_successful": False, "host_stability_proven": False}
    verify_inputs(inputs)
    OUT.mkdir()
    write(OUT / "inputs.json", inputs)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "cwd": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            "progress_path": str(OUT / "metrics.jsonl"), "expected_final_step": 15,
            "argv": [PYTHON, "-u", "-m", "research.direct.recover_latency58_finalization_idle",
                     "--child-inputs-sha256", sha(OUT / "inputs.json")]}
    write(OUT / "watchdog-spec.json", spec)
    monitor_out = WATCHDOG.parent / OUT.name
    require(not monitor_out.exists(), "Preserve previous monitor output")
    argv = [PYTHON, str(WATCHDOG), "--launch-spec", str(OUT / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(OUT / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", "240", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60",
            "--finalization-timeout-seconds", "180", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(OUT / "command.json", {"argv": argv, "monitor_result": str(monitor_out / "result.json"),
                                "inputs_sha256": sha(OUT / "inputs.json")})
    print(json.dumps({"status": "prepared", "inputs_sha256": sha(OUT / "inputs.json"),
                      "command_sha256": sha(OUT / "command.json"), "gpu_workload_started": False}), flush=True)


def child(expected_sha):
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and sha(OUT / "inputs.json") == expected_sha, "Idle inputs changed")
    inputs = read(OUT / "inputs.json")
    verify_inputs(inputs)
    failed = read(inputs["failed_monitor"])
    require(not Path("/proc", str(failed["child_pid"])).exists(), "Failed training child is still present")
    monitor = load_source("finalization_idle_monitor", WATCHDOG)
    _, evidence = continuity(inputs, monitor)
    write(OUT / "event-continuity.json", evidence)
    with (OUT / "metrics.jsonl").open("x", buffering=1) as journal:
        for step in range(1, inputs["expected_final_step"] + 1):
            time.sleep(1)
            journal.write(json.dumps({"step": step, "cpu_idle_only": True}) + "\n")
    began = time.monotonic()
    time.sleep(inputs["deliberate_finalization_delay_seconds"])
    elapsed = time.monotonic() - began
    verify_inputs(inputs)
    write(OUT / "child-result.json", {"status": "pass", "inputs_sha256": expected_sha,
          "source_bindings_unchanged": True, "event_continuity_passed": True,
          "previous_event_record_id": inputs["previous_event_record_id"], "idle_heartbeats": 15,
          "deliberate_finalization_delay_seconds": elapsed, "original_failed_child_absent": True,
          "gpu_workload_started": False, "host_stability_proven": False})
    print(json.dumps({"status": "pass", "cpu_idle_only": True, "finalization_delay_seconds": elapsed}), flush=True)


def run(expected_sha):
    require(Path.cwd() == ROOT and sha(OUT / "command.json") == expected_sha,
            "Prepared idle command changed")
    command, inputs = read(OUT / "command.json"), read(OUT / "inputs.json")
    require(command["inputs_sha256"] == sha(OUT / "inputs.json"), "Idle input binding changed")
    verify_inputs(inputs)
    began = time.monotonic()
    with (OUT / "console.log").open("x") as log:
        completed = subprocess.run(command["argv"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(p) == h for p, h in inputs["source_bindings"].items())
    write(OUT / "execution.json", {"actual_exit_code": completed.returncode,
          "elapsed_seconds": time.monotonic() - began, "source_bindings_unchanged": unchanged,
          "command_sha256": expected_sha, "inputs_sha256": command["inputs_sha256"],
          "monitor_result": command["monitor_result"], "gpu_workload_started": False})
    require(completed.returncode == 0 and unchanged, "Monitored idle finalization failed")
    terminal, idle = read(command["monitor_result"]), read(OUT / "child-result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"] and terminal["identities_unchanged"]
            and terminal["finalization_started"] and terminal["expected_final_step"] == 15
            and 60 < terminal["finalization_elapsed_seconds"] < 180
            and terminal["event_worker_close"]["actual_exit_code"] == 0
            and not terminal["event_worker_close"]["forced"] and idle["status"] == "pass"
            and idle["event_continuity_passed"] and idle["deliberate_finalization_delay_seconds"] >= 65,
            "Real idle finalization or event continuity did not complete")
    write(OUT / "result.json", {"status": "pass", "source_bindings_unchanged": True,
          "inputs_sha256": command["inputs_sha256"], "command_sha256": expected_sha,
          "monitor_result": {"path": command["monitor_result"], "sha256": sha(command["monitor_result"])},
          "finalization_elapsed_seconds": terminal["finalization_elapsed_seconds"],
          "continuous_real_host_monitoring_during_finalization": True,
          "original_training_monitor_successful": False, "gpu_workload_started": False,
          "host_stability_proven": False, "budget_after": budget_snapshot(inputs["storage_budget"])})
    print(json.dumps({"status": "pass", "finalization_elapsed_seconds": terminal["finalization_elapsed_seconds"],
                      "gpu_workload_started": False}), flush=True)


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
        child(args.child_inputs_sha256)
    else:
        run(args.run_command_sha256)
