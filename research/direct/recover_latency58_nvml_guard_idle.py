"""Check real host continuity and both persistent workers after the step-171 NVML timeout.

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

SOURCE = PHASE / "branch-grouped-vocal-011"
CONTROL = PHASE / "nvml-guard-monitor-check-001"
TRANSPORT_CONTROL = PHASE / "nvml-transport-check-001"
OUT = PHASE / "branch-grouped-vocal-nvml-guard-idle-001"
WATCHDOG = PHASE.parent / "latency11/smoke/gpu-crash-followup/watch_gpu_process_nvml_guard.py"
WATCHDOG_SHA = "8f58bfaa4bc349e083a5843385560f6b631098f190be7baa2343011260bb1499"


def require_monitor_qualification():
    bindings = {}
    for directory in (CONTROL, TRANSPORT_CONTROL):
        plan, result, execution, root = (read(directory / (n + ".json"))
                                         for n in ("plan", "result", "execution", "root-execution"))
        verify_inputs(plan)
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == root["actual_exit_code"] == 0
                and execution["source_bindings_unchanged"] and not execution["timed_out"]
                and root["source_bindings_unchanged"] and not root["timed_out"]
                and type(root["actual_session_id"]) is int and isinstance(root["actual_tool_chunk_id"], str)
                and execution["result_sha256"] == root["result_sha256"] == sha(directory / "result.json")
                and root["execution_sha256"] == sha(directory / "execution.json")
                and execution["plan_sha256"] == result["plan_sha256"] == sha(directory / "plan.json"),
                "NVML transport or monitor qualification is incomplete")
        if directory == CONTROL:
            require(len(result["cases"]) == 22 and all(c["status"] == "pass" for c in result["cases"])
                    and result["all_actual_cpu_children_reaped"] and result["host_telemetry_simulated"]
                    and sha(WATCHDOG) == WATCHDOG_SHA, "NVML monitor controls changed")
        bindings.update(plan["source_bindings"])
        bindings.update({str(directory / (n + ".json")): sha(directory / (n + ".json"))
                         for n in ("plan", "result", "execution", "root-execution")})
    for name, count, session, chunk in (
            ("nvml-guard-reader-cpu-001", 20, None, "0af208"),
            ("nvml-guard-worker-cpu-001", 6, 14591, "37e846")):
        directory = PHASE / name
        plan, result, execution, root = (read(directory / (n + ".json"))
            for n in ("plan", "result", "execution", "root-execution"))
        verify_inputs(plan)
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and len(result["cases"]) == count and all(c["status"] == "pass" for c in result["cases"])
                and execution["actual_exit_code"] == root["actual_exit_code"] == 0
                and not execution["timed_out"] and not root["timed_out"]
                and root["actual_session_id"] == session and root["actual_tool_chunk_id"] == chunk
                and execution["result_sha256"] == root["result_sha256"] == sha(directory / "result.json")
                and execution["plan_sha256"] == result["plan_sha256"] == root["plan_sha256"] == sha(directory / "plan.json")
                and root["execution_sha256"] == sha(directory / "execution.json")
                and execution["source_bindings_unchanged"] and root["source_bindings_unchanged"]
                and not root["gpu_queried"], "Guard reader or worker CPU qualification is incomplete")
        bindings.update(plan["source_bindings"])
        bindings.update({str(directory / (n + ".json")): sha(directory / (n + ".json"))
                        for n in ("plan", "result", "execution", "root-execution")})
    verify_inputs({"source_bindings": bindings})
    return bindings


def prepare():
    require(Path.cwd() == ROOT and not OUT.exists(), "Preserve prior idle observations")
    failed_execution, failed_root = (read(SOURCE / n) for n in ("production-stage/execution.json", "production-root-execution.json"))
    failed_path = Path(failed_execution["monitor_result"])
    failed = read(failed_path)
    require(failed_execution["actual_exit_code"] == failed_root["actual_exit_code"] == 1
            and failed_execution["source_bindings_unchanged"]
            and failed_root["actual_session_id"] == 54158 and failed_root["actual_tool_chunk_id"] == "a79ba7"
            and failed["child_exit_code"] == 1 and failed["latest_completed_step_seen"] == 171
            and failed["status"] == "stopped_by_watchdog" and failed["supervisor_health"] == "failed"
            and failed["reason"] == "RuntimeError('nvml telemetry timed out')"
            and failed["last_event_record_id"] == 65004
            and not Path("/proc", str(failed["child_pid"])).exists(), "Original failure evidence differs")
    for name in ("event_worker_close", "gpu_worker_close"):
        worker = failed[name]
        require(worker["closed"] and not worker["forced"] and worker["actual_exit_code"] == 0
                and not Path("/proc", str(worker["linux_pid"])).exists(), "Failed worker is still present")
    require(failed_root["execution_sha256"] == sha(SOURCE / "production-stage/execution.json")
            and failed_root["monitor_result_sha256"] == sha(failed_path), "Failed root bindings changed")
    bindings = require_monitor_qualification()
    budget_path = SOURCE / "plan.json"
    budget = read(budget_path)["storage_budget"]
    paths = [Path(__file__).resolve(), SOURCE / "production-stage/execution.json", SOURCE / "production-root-execution.json",
             failed_path, WATCHDOG, budget_path]
    bindings.update({str(p): sha(p) for p in paths})
    inputs = {"schema": "latency58-nvml-idle-v1", "source_bindings": bindings,
              "failed_monitor": str(failed_path), "previous_event_record_id": failed["last_event_record_id"],
              "watchdog_source": str(WATCHDOG), "output_directory": str(OUT), "expected_final_step": 15,
              "deliberate_finalization_delay_seconds": 65, "storage_budget": budget,
              "budget_before": budget_snapshot(budget), "gpu_workload_started": False,
              "original_training_monitor_successful": False, "host_stability_proven": False}
    verify_inputs(inputs)
    OUT.mkdir()
    write(OUT / "inputs.json", inputs)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "cwd": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            "progress_path": str(OUT / "metrics.jsonl"), "expected_final_step": 15,
            "argv": [PYTHON, "-u", "-m", "research.direct.recover_latency58_nvml_guard_idle",
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
    monitor = load_source("nvml_idle_monitor", WATCHDOG)
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
            and not terminal["event_worker_close"]["forced"]
            and terminal["gpu_worker_close"]["actual_exit_code"] == 0
            and terminal["gpu_worker_close"]["closed"] and terminal["gpu_worker_close"]["identities_unchanged"]
            and not terminal["gpu_worker_close"]["forced"] and idle["status"] == "pass"
            and idle["event_continuity_passed"] and idle["deliberate_finalization_delay_seconds"] >= 65,
            "Real idle finalization or event continuity did not complete")
    for pid in (terminal["child_pid"], terminal["event_worker_close"]["linux_pid"],
                terminal["gpu_worker_close"]["linux_pid"]):
        require(not Path("/proc", str(pid)).exists(), "Owned idle child or worker remains present")
    log_rows = [json.loads(line) for line in
        (Path(command["monitor_result"]).parent / "watchdog.jsonl").read_text().splitlines()]
    queries = [row for row in log_rows if row["event"] == "query" and row.get("kind") == "nvml"]
    require(len(queries) >= 30 and all(
        row["nvml"]["sampling_policy"] == "latency58-nvml-required-health-getters-v1"
        and len(row["nvml"]["api_timings"]) == 7
        and all(t["returncode"] == 0 for t in row["nvml"]["api_timings"])
        and row["nvml"]["memory_bytes"]["free"] >= 4096 * 2**20 for row in queries),
        "Real required-health sampling did not complete")
    write(OUT / "result.json", {"status": "pass", "source_bindings_unchanged": True,
          "real_guard_reader_queries": len(queries),
          "max_nvml_query_seconds": max(row["duration"] for row in queries),
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
