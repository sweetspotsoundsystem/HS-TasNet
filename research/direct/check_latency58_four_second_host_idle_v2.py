"""Exercise the unchanged real monitor before resuming the step-1900 save."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, continuity
from research.direct.latency58_four_second_monitor import WATCHDOG, require_monitor_qualification, require_monitor_closed
from research.direct.check_latency58_four_second_recovery_v2 import OUT, binding

DIRECTORY = OUT / "host-idle-002"
MONITOR = OUT.parent / "monitors/branch-four-second-015-host-idle-002"


def child(expected):
    from research.direct import watch_latency58_four_second as monitor
    require(sha(DIRECTORY / "inputs.json") == expected, "Idle inputs changed")
    inputs = read(DIRECTORY / "inputs.json")
    verify_inputs(inputs)
    _, evidence = continuity(inputs, monitor)
    write(DIRECTORY / "event-continuity.json", evidence)
    with (DIRECTORY / "metrics.jsonl").open("x", buffering=1) as journal:
        for step in range(1, 16):
            time.sleep(1)
            journal.write(json.dumps({"step": step, "cpu_idle_only": True}) + "\n")
    began = time.monotonic()
    time.sleep(65)
    verify_inputs(inputs)
    write(DIRECTORY / "child-result.json", {"status": "pass", "gpu_workload_started": False,
          "inputs_sha256": expected, "finalization_delay_seconds": time.monotonic() - began})


def run():
    audit_path = OUT / "post-interruption-host-audit-002.json"
    audit = read(audit_path)
    verify_inputs(audit)
    require(audit["status"] == "observed" and audit["event_gap_fully_covered"] and not audit["fault_records"],
            "Review host faults before establishing a new baseline")
    previous, current = audit["last_baseline"]["gpu"], audit["current_gpu"]
    require(all(previous[k] == current[k] for k in ("uuid", "driver_version", "memory.total")),
            "GPU identity changed since qualification")
    bindings = require_monitor_qualification()
    bindings.update({str(p): sha(p) for p in (audit_path, Path(__file__).resolve())})
    require(not DIRECTORY.exists() and not MONITOR.exists(), "Preserve idle evidence")
    DIRECTORY.mkdir()
    inputs = {"source_bindings": bindings, "previous_event_record_id": audit["last_event_record_id"]}
    write(DIRECTORY / "inputs.json", inputs)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "cwd": str(ROOT),
        "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "expected_final_step": 15, "progress_path": str(DIRECTORY / "metrics.jsonl"),
        "argv": [PYTHON, "-u", "-m", "research.direct.check_latency58_four_second_host_idle_v2",
                 "--child-inputs-sha256", sha(DIRECTORY / "inputs.json")]}
    write(DIRECTORY / "watchdog-spec.json", spec)
    argv = [PYTHON, str(WATCHDOG), "--launch-spec", str(DIRECTORY / "watchdog-spec.json"),
        "--launch-spec-sha256", sha(DIRECTORY / "watchdog-spec.json"), "--output-dir", str(MONITOR),
        "--max-runtime-seconds", "240", "--poll-seconds", "2", "--query-timeout-seconds", "10",
        "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60",
        "--finalization-timeout-seconds", "180", "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10",
        "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(DIRECTORY / "command.json", {"argv": argv})
    began = time.monotonic()
    with (DIRECTORY / "console.log").open("x") as log:
        code = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    unchanged = all(sha(p) == digest for p, digest in bindings.items())
    execution = {"actual_exit_code": code, "source_bindings_unchanged": unchanged,
                 "elapsed_seconds": time.monotonic() - began, "monitor_result": str(MONITOR / "result.json")}
    write(DIRECTORY / "execution.json", execution)
    terminal = read(MONITOR / "result.json")
    require_monitor_closed(execution, terminal, final_step=15)
    require(60 < terminal["finalization_elapsed_seconds"] < 180
            and read(DIRECTORY / "child-result.json")["finalization_delay_seconds"] >= 65,
            "Idle monitor did not span the deliberate finalization delay")
    for pid in (terminal["child_pid"], terminal["event_worker_close"]["linux_pid"],
                terminal["gpu_worker_close"]["linux_pid"]):
        require(not Path(f"/proc/{pid}").exists(), "Owned idle worker remains present")
    result = {"status": "pass", "monitor_result": binding(MONITOR / "result.json"),
              "execution": binding(DIRECTORY / "execution.json"), "source_bindings": bindings,
              "gpu_workload_started": False, "historical_monitor_successful": False}
    write(DIRECTORY / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-inputs-sha256")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Use controlled checkout")
    child(args.child_inputs_sha256) if args.child_inputs_sha256 else run()
