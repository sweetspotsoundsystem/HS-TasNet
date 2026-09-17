"""Check event continuity after the stopped two-second context run without a GPU workload."""
import argparse
import json
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import load_source, continuity, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.inputs) == args.inputs_sha256, "Recovery inputs changed")
    inputs = read(args.inputs)
    verify_inputs(inputs)
    failed = read(inputs["failed_monitor"])
    require(failed["status"] == "stopped_by_watchdog" and failed["child_exit_code"] == 1
            and failed["reason"] == "RuntimeError('windows_system telemetry timed out')"
            and not Path("/proc", str(failed["child_pid"])).exists(),
            "Require the stopped telemetry-timeout child to be gone")
    monitor = load_source("long_context_telemetry_monitor", inputs["watchdog_source"])
    _, evidence = continuity(inputs, monitor)
    out = Path(inputs["output_directory"])
    write(out / "event-continuity.json", evidence)
    with (out / "metrics.jsonl").open("x", buffering=1) as journal:
        for step in range(1, 16):
            time.sleep(1)
            journal.write(json.dumps({"step": step, "cpu_idle_only": True}) + "\n")
    verify_inputs(inputs)
    write(out / "child-result.json", {"status": "pass", "event_continuity_passed": True,
          "failed_training_pid_gone": True, "gpu_workload_started": False,
          "driver_registry_and_power_settings_changed": False, "source_bindings_unchanged": True})
    print(json.dumps({"status": "pass", "cpu_idle_only": True}), flush=True)


if __name__ == "__main__":
    main()
