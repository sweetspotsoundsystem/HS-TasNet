"""Check that stderr timing markers preserve the frozen event query and fault checks."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    failed_root = PHASE / "branch-long-context-004"
    failed_path = Path(read(failed_root / "resource-stage/execution.json")["monitor_result"])
    failed = read(failed_path)
    require(failed["reason"] == "RuntimeError('windows_system telemetry timed out')"
            and failed["child_exit_code"] == 1 and not Path("/proc", str(failed["child_pid"])).exists(),
            "Require the closed failed child")
    original_path = failed_path.parent.parent / "watch_gpu_process.py"
    trace_path = original_path.with_name("watch_gpu_process_trace.py")
    original, trace = original_path.read_text(), trace_path.read_text()
    markers = [line for line in trace.split("\n") if "[Console]::Error.WriteLine('telemetry_phase:" in line]
    require(len(markers) == 5 and "\n".join(line for line in trace.split("\n") if line not in markers) == original,
            "Instrumentation changed more than five stderr marker lines")
    out = PHASE / "branch-long-context-telemetry-trace-001"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve earlier trace qualification")
    out.mkdir()
    bindings = {str(path): sha(path) for path in (Path(__file__).resolve(), original_path, trace_path, failed_path,
                 failed_root / "resource-stage/execution.json", failed_root / "resource-stage/root-execution.json")}
    write(out / "inputs.json", {"source_bindings": bindings, "previous_event_record_id": failed["last_event_record_id"],
          "timeout_seconds": 10, "gpu_workload_started": False, "os_driver_registry_power_changes": False})
    observations = []
    for name, path in (("original", original_path), ("trace", trace_path), ("trace_repeat", trace_path)):
        module = load(name, path)
        program = module.event_query(failed["last_event_record_id"], verify_sentinel=True)
        argv = [module.POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
                base64.b64encode(program.encode("utf-16le")).decode()]
        began = time.monotonic()
        child = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        require(child.returncode == 0, "Event query did not complete")
        payload = json.loads(child.stdout)
        newest, rows = module.validate_events(payload, failed["last_event_record_id"])
        require(payload["SentinelVerified"] and not any(module.reset_event(row) for row in rows),
                "Sentinel verification or fresh fault coverage failed")
        expected_phases = ["start", "before_initial_read", "after_initial_read", "coverage_loaded", "records_encoded"]
        phases = [line.split(":", 2)[1] for line in child.stderr.splitlines() if line.startswith("telemetry_phase:")]
        require(phases == ([] if name == "original" else expected_phases), "Trace markers are missing or reordered")
        observations.append({"name": name, "actual_exit_code": child.returncode,
                             "elapsed_seconds": time.monotonic() - began, "payload": payload,
                             "stderr": child.stderr, "newest_event_record_id": newest, "new_fault_records": []})
    normalized = [{k: v for k, v in row["payload"].items() if k != "CheckedUtc"} for row in observations]
    require(all(row == normalized[0] for row in normalized[1:]), "Repeated event payloads differ")
    require(all(sha(path) == digest for path, digest in bindings.items()), "Trace qualification inputs changed")
    result = {"status": "pass", "source_bindings_unchanged": True, "source_bindings": bindings,
              "only_five_stderr_lines_added": True, "original_and_traced_event_payloads_match": True,
              "sentinel_and_coverage_and_fault_detection_unchanged": True, "query_deadline_seconds": 10,
              "gpu_workload_started": False, "observations": observations}
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_bindings", "observations")}), flush=True)


if __name__ == "__main__":
    main()
