"""Qualify persistent event-query transport, unchanged safety logic and failure handling on CPU."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.latency58_windows_event_transport import PersistentWindowsQuery


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_function(source, name):
    return next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    old = PHASE / "branch-long-context-005"
    failed_path = Path(read(old / "resource-stage/execution.json")["monitor_result"])
    failed = read(failed_path)
    require(failed["child_exit_code"] == 1 and not Path("/proc", str(failed["child_pid"])).exists(),
            "Require the failed GPU child to be closed")
    trace_path = failed_path.parent.parent / "watch_gpu_process_trace.py"
    persistent_path = trace_path.with_name("watch_gpu_process_persistent.py")
    trace_source, persistent_source = trace_path.read_text(), persistent_path.read_text()
    preserved = ("event_query", "validate_events", "reset_event", "parse_gpu", "gpu_alert")
    require(all(dump(tree_function(trace_source, name)) == dump(tree_function(persistent_source, name))
                for name in preserved), "Event coverage or GPU fault-detection functions changed")
    main_functions = [tree_function(source, "main") for source in (trace_source, persistent_source)]
    controls = [next(node for node in function.body if isinstance(node, ast.Try)) for function in main_functions]
    require([dump(node) for node in controls[0].body] == [dump(node) for node in controls[1].body]
            and [dump(node) for node in controls[0].handlers] == [dump(node) for node in controls[1].handlers]
            and [dump(node) for node in controls[0].finalbody[:2]] == [dump(node) for node in controls[1].finalbody[:2]],
            "GPU launch, supervision loop, failure classification or owned-child stop changed")
    arguments = [[dump(node) for node in ast.walk(function) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
                 for function in main_functions]
    require(arguments[0] == arguments[1], "Supervisor CLI limits changed")
    trace, persistent = load("trace_monitor", trace_path), load("persistent_monitor", persistent_path)
    require(sha(persistent.TRANSPORT_SOURCE) == persistent.TRANSPORT_SHA256, "Pinned transport identity changed")
    out = PHASE / "branch-long-context-persistent-monitor-001"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve earlier transport qualifications")
    out.mkdir()
    paths = (Path(__file__).resolve(), trace_path, persistent_path, persistent.TRANSPORT_SOURCE, failed_path,
             old / "resource-stage/execution.json", old / "resource-stage/root-execution.json")
    bindings = {str(path): sha(path) for path in paths}
    write(out / "inputs.json", {"source_bindings": bindings, "gpu_workload_started": False,
          "runtime_query_deadline_seconds": 10, "previous_event_record_id": failed["last_event_record_id"]})
    program = trace.event_query(failed["last_event_record_id"], verify_sentinel=True)
    argv = [trace.POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
            base64.b64encode(program.encode("utf-16le")).decode()]
    began = time.monotonic()
    reference = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    require(reference.returncode == 0, "Original event query failed")
    reference_elapsed = time.monotonic() - began
    expected = json.loads(reference.stdout)
    newest, records = trace.validate_events(expected, failed["last_event_record_id"])
    require(expected["SentinelVerified"] and not any(trace.reset_event(row) for row in records),
            "Original event continuity or sentinel failed")
    normalized = {k: v for k, v in expected.items() if k != "CheckedUtc"}
    rows, closures, identities = [], [], []
    worker = persistent.load_event_transport()
    try:
        for index in range(3):
            row = worker.query(program, 10)
            payload = json.loads(row["stdout"])
            mark, records = persistent.validate_events(payload, failed["last_event_record_id"])
            require(mark == newest and payload["SentinelVerified"]
                    and not any(persistent.reset_event(record) for record in records)
                    and {k: v for k, v in payload.items() if k != "CheckedUtc"} == normalized
                    and row["request_id"] == index + 1 and row["worker_still_running"],
                    "Persistent event response differs from the original complete scan")
            rows.append(row)
        require(len({(row["worker_pid"], row["worker_started"]) for row in rows}) == 1,
                "Persistent transport started a process for each query")
    finally:
        closures.append({"case": "normal_event_queries", **worker.close()})
    require(closures[-1]["actual_exit_code"] == 0 and not closures[-1]["forced"], "Normal worker failed to close")
    identities.append((worker.worker_pid, worker.worker_started))
    fault_cases = []
    worker = PersistentWindowsQuery(trace.POWERSHELL)
    try:
        worker.query("'ready'", 10)
        identity = (worker.process.pid, worker.worker_pid, worker.worker_started)
        began = time.monotonic()
        try:
            worker.query("[Console]::Error.WriteLine('bounded_timeout_entered'); Start-Sleep -Milliseconds 1000; 'late'", .2)
        except subprocess.TimeoutExpired as error:
            duration = time.monotonic() - began
            require(duration < .8 and b"bounded_timeout_entered" in error.stderr,
                    "Persistent IO failed to enforce its deadline or expose partial output")
        else:
            raise RuntimeError("Late response was incorrectly accepted")
        try:
            worker.query("'must_not_run'", 1)
        except RuntimeError:
            require(identity == (worker.process.pid, worker.worker_pid, worker.worker_started) and worker.broken,
                    "Failed worker restarted automatically")
        else:
            raise RuntimeError("Failed query transport was reused")
        fault_cases.append({"case": "timeout", "status": "pass", "deadline_seconds": .2,
                            "observed_seconds": duration, "partial_stderr_received": True,
                            "late_response_rejected_and_automatic_restart_forbidden": True})
    finally:
        closures.append({"case": "after_timeout", **worker.close()})
    require(closures[-1]["actual_exit_code"] == 0 and not closures[-1]["forced"], "Timed-out CPU probe did not close")
    identities.append((worker.worker_pid, worker.worker_started))
    worker = PersistentWindowsQuery(trace.POWERSHELL)
    try:
        worker.query("'ready'", 10)
        forged = json.dumps({"kind": "response", "session": worker.session, "id": 999,
                            "pid": worker.worker_pid, "stdout": "{}"})
        try:
            worker.query("[Console]::Out.WriteLine('" + forged + "'); 'following_response'", 10)
        except RuntimeError:
            require(worker.broken, "Stale request identity did not poison the transport")
        else:
            raise RuntimeError("Wrong request identity was accepted")
        fault_cases.append({"case": "stale_response_identity", "status": "pass", "response_rejected": True})
    finally:
        closures.append({"case": "after_stale_response", **worker.close()})
    require(closures[-1]["actual_exit_code"] == 0 and not closures[-1]["forced"], "Stale-response fixture did not close")
    identities.append((worker.worker_pid, worker.worker_started))
    worker = PersistentWindowsQuery(trace.POWERSHELL)
    try:
        worker.query("'ready'", 10)
        worker.process.terminate()
        worker.process.wait(timeout=5)
        try:
            worker.query("'must_not_run'", 1)
        except RuntimeError:
            require(worker.broken and worker.process.returncode is not None, "Worker exit was not detected")
        else:
            raise RuntimeError("Exited worker was replaced or accepted")
        fault_cases.append({"case": "worker_exit", "status": "pass", "exit_detected_without_restart": True})
    finally:
        closures.append({"case": "after_owned_termination_probe", **worker.close()})
    identities.append((worker.worker_pid, worker.worker_started))
    pairs = ",".join("@{Id=" + str(pid) + ";Started='" + started + "'}" for pid, started in identities)
    verify_program = "$ProgressPreference='SilentlyContinue'; $ErrorActionPreference='Stop'; $rows=@(" + pairs + "); @($rows | ForEach-Object { $p=Get-Process -Id $_.Id -ErrorAction SilentlyContinue; [pscustomobject]@{Id=$_.Id;SameInstanceStillRunning=($null -ne $p -and $p.StartTime.ToUniversalTime().ToString('o') -eq $_.Started)} }) | ConvertTo-Json -Compress"
    check = subprocess.run([trace.POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
        base64.b64encode(verify_program.encode("utf-16le")).decode()], capture_output=True, text=True, timeout=10)
    require(check.returncode == 0, "Could not verify owned Windows worker closure")
    windows_closure = json.loads(check.stdout)
    require(len(windows_closure) == len(identities) and all(not row["SameInstanceStillRunning"] for row in windows_closure),
            "An owned Windows query worker survived its CPU test")
    require(all(sha(path) == digest for path, digest in bindings.items()), "Qualification source changed")
    write(out / "result.json", {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
          "gpu_workload_started": False, "original_and_persistent_event_payloads_match": True,
          "sentinel_coverage_fault_detection_and_gpu_control_loop_unchanged": True,
          "runtime_deadline_seconds": 10, "original_query_seconds": reference_elapsed,
          "persistent_queries": rows, "fault_cases": fault_cases, "worker_closures": closures,
          "windows_worker_closure_confirmed": windows_closure})
    print(json.dumps({"status": "pass", "persistent_query_seconds": [row["elapsed_seconds"] for row in rows],
                      "original_query_seconds": reference_elapsed, "fault_cases": fault_cases,
                      "all_owned_workers_closed": True, "gpu_workload_started": False}), flush=True)


if __name__ == "__main__":
    main()
