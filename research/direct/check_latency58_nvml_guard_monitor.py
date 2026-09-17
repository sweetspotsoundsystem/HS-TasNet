"""Exercise persistent NVML supervision with real CPU children and simulated telemetry.

This qualification does not query or initialize a GPU, or establish host health.
The unchanged Windows event and GPU threshold functions are compared by AST.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
from types import SimpleNamespace

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, require, sha, write

HERE = PHASE.parent / "latency11/smoke/gpu-crash-followup"
OLD = HERE / "watch_gpu_process_nvml.py"
NEW = HERE / "watch_gpu_process_nvml_guard.py"
OUT = PHASE / "nvml-guard-monitor-check-001"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preserved_functions():
    names = tuple(n.name for n in ast.parse(OLD.read_text()).body if isinstance(n, ast.FunctionDef))
    trees = [ast.parse(path.read_text()) for path in (OLD, NEW)]
    for name in names:
        nodes = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
        require(ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False),
                "Existing telemetry or event coverage changed: " + name)
    return list(names)


def run_case(name, *, rows=(2,), delay=.8, child_code=0, final_timeout=3, fault=None,
             expected_status, expected_reason=None, phase_expected=True, max_runtime=20):
    monitor = load("nvml_monitor_control_" + name, NEW)
    case = OUT / name
    case.mkdir()
    journal = case / "metrics.jsonl"
    # This child imports only the Python standard library and never initializes CUDA.
    child_source = ("import json,time,sys\n"
                    "with open(sys.argv[1], 'x', buffering=1) as f:\n"
                    f" for step in {list(rows)!r}:\n"
                    "  f.write(json.dumps({'step':step})+'\\n')\n"
                    "  f.flush()\n"
                    "  time.sleep(.05)\n"
                    f"time.sleep({delay!r})\n"
                    f"raise SystemExit({child_code})\n")
    fixture = case / "child.py"
    fixture.write_text(child_source)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "expected_final_step": 2,
            "argv": [PYTHON, str(fixture), str(journal)], "cwd": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            "progress_path": str(journal)}
    write(case / "launch-spec.json", spec)
    monitor_out = HERE / ("nvml-guard-cpu-control-001-" + name)
    require(not monitor_out.exists(), "Preserve earlier supervisor fixtures")
    state = {"final_queries": 0, "fault_injected": False, "newest": 100}

    class SimulatedTransport:
        worker_pid = -1
        request_id = 0

        def query(self, program, timeout):
            self.request_id += 1
            final = monitor.progress_step(journal) == 2
            if final:
                state["final_queries"] += 1
            # Let one complete finalization iteration happen before injecting faults.
            inject = state["final_queries"] >= 2 and not state["fault_injected"]
            if fault == "query_timeout" and inject:
                state["fault_injected"] = True
                raise subprocess.TimeoutExpired("simulated_event_query", timeout, stderr=b"fixture timeout")
            if fault == "event_worker_exit" and inject:
                state["fault_injected"] = True
                raise RuntimeError("Simulated persistent event worker exited")
            if fault == "host_fault" and inject:
                state["fault_injected"] = True
                state["newest"] += 1
            previous = int(re.search(r"\$previous=\[long\](-?\d+)", program).group(1))
            ids = list(range(state["newest"], 0, -1))
            records = [{"RecordId": i, "Provider": "nvlddmkm" if i == 101 else "Fixture",
                        "Id": 153 if i == 101 else 1, "Level": 2 if i == 101 else 4,
                        "Xml": "<Event fixture='true' />"} for i in ids if i > previous]
            payload = {"Status": "ok", "Count": len(ids), "RecordIds": ids,
                       "NewRecords": records, "SentinelVerified": True}
            return {"stdout": json.dumps(payload), "stderr": "simulated telemetry; no host query",
                    "worker_pid": self.worker_pid, "request_id": self.request_id, "worker_still_running": True}

        def close(self):
            return {"closed": True, "forced": False, "actual_exit_code": 0, "simulated": True}

    class SimulatedNvmlTransport:
        worker_pid = -2
        request_id = 0

        def query(self, timeout):
            require(0 < timeout <= 10, "NVML deadline changed")
            self.request_id += 1
            inject = state["final_queries"] >= 2
            temperature, used, reserved, identity = 40, 1000, 128, "GPU-fixture"
            if inject and fault == "gpu_query_timeout":
                state["fault_injected"] = True
                raise subprocess.TimeoutExpired("simulated_nvml_query", timeout, stderr=b"fixture timeout")
            if inject and fault == "gpu_worker_exit":
                state["fault_injected"] = True
                raise RuntimeError("Simulated persistent NVML worker exited")
            if inject and fault == "temperature":
                temperature = 80
            if inject and fault == "memory":
                used = 15000
            if inject and fault == "physical_free_memory":
                reserved = 12000
            if inject and fault == "identity":
                identity = "GPU-changed-fixture"
            if inject and fault in ("temperature", "memory", "identity", "physical_free_memory", "invalid_raw_memory"):
                state["fault_injected"] = True
            memory = {"total": 16384 * 2**20, "used": used * 2**20, "reserved": reserved * 2**20,
                      "free": (16384 - used - reserved) * 2**20}
            if inject and fault == "invalid_raw_memory":
                memory["free"] += 1
            return {"stdout": f"{identity}, Fixture, fixture-driver, 16384, {used}, {temperature}, not_queried, not_queried, not_queried\n",
                    "stderr": "simulated NVML; no GPU query", "worker_pid": self.worker_pid,
                    "request_id": self.request_id, "worker_still_running": True,
                    "nvml": {"memory_bytes": memory, "api_timings": [], "fixture": True}}

        def close(self):
            result = {"closed": True, "forced": False, "actual_exit_code": 0,
                      "identities_unchanged": True, "simulated": True}
            if fault in ("gpu_close_nonzero", "gpu_close_forced", "gpu_close_identity", "gpu_close_exception"):
                state["fault_injected"] = True
                if fault == "gpu_close_exception":
                    raise RuntimeError("Simulated NVML cleanup error")
                result.update({"actual_exit_code": 7} if fault == "gpu_close_nonzero" else
                              {"forced": True} if fault == "gpu_close_forced" else {"identities_unchanged": False})
            return result

    def load_simulated_gpu():
        if fault == "gpu_preflight_failure":
            state["fault_injected"] = True
            raise RuntimeError("Simulated NVML construction failure")
        return SimulatedNvmlTransport()

    def forbidden_external_query(*args, **kwargs):
        raise RuntimeError("CPU qualification attempted an external query")

    monitor.load_event_transport = SimulatedTransport
    monitor.load_gpu_transport = load_simulated_gpu
    monitor.subprocess = SimpleNamespace(run=forbidden_external_query, Popen=subprocess.Popen,
                                        TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT)
    old_argv = sys.argv
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    sys.argv = [str(NEW), "--launch-spec", str(case / "launch-spec.json"), "--launch-spec-sha256",
                sha(case / "launch-spec.json"), "--output-dir", str(monitor_out),
                "--max-runtime-seconds", str(max_runtime), "--poll-seconds", ".5",
                "--progress-timeout-seconds", ".2", "--startup-grace-seconds", "2",
                "--finalization-timeout-seconds", str(final_timeout), "--stop-grace-seconds", "1",
                "--post-exit-quiet-seconds", "10"]
    write(case / "command.json", {"argv": sys.argv, "host_telemetry_simulated": True,
                                  "real_child_is_cpu_stdlib_fixture": True})
    try:
        with (case / "console.log").open("x") as log, redirect_stdout(log):
            exit_code = monitor.main()
    finally:
        sys.argv = old_argv
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    result = json.loads((monitor_out / "result.json").read_text())
    events = [json.loads(line) for line in (monitor_out / "watchdog.jsonl").read_text().splitlines()]
    require(result["status"] == expected_status and result["finalization_started"] == phase_expected,
            "Unexpected finalization fixture outcome: " + name)
    require(exit_code == (0 if expected_status == "pass" else 1) and result["identities_unchanged"]
            and not Path("/proc", str(result["child_pid"])).exists(), "Fixture child or identity did not close")
    if expected_reason:
        require(expected_reason in result["reason"], "Fixture failed for an unrelated reason: " + name)
    if fault:
        require(state["fault_injected"] and (not phase_expected or
                any(r["event"] == "finalization_started" for r in events)), "Fault was not exercised")
    if expected_status in ("pass", "child_failed"):
        require(result["post_exit_quiet_completed"] and result["child_exit_code"] == child_code,
                "Actual child exit or quiet period was lost")
    if name == "delayed_clean_exit":
        require(result["finalization_elapsed_seconds"] > .2 and result["child_exit_code"] == 0,
                "Finalization did not outlast the update-stall deadline")
    record = {"case": name, "status": "pass", "actual_monitor_return_code": exit_code,
              "actual_child_exit_code": result["child_exit_code"], "monitor_status": result["status"],
              "monitor_reason": result["reason"], "finalization_started": result["finalization_started"],
              "finalization_elapsed_seconds": result["finalization_elapsed_seconds"],
              "actual_owned_child_reaped": result["child_pid"] is not None,
              "child_never_started": result["child_pid"] is None, "telemetry_simulated": True,
              "gpu_worker_close": result["gpu_worker_close"], "event_worker_close": result["event_worker_close"],
              "gpu_workload_started": False, "host_stability_proven": False,
              "result": {"path": str(monitor_out / "result.json"), "sha256": sha(monitor_out / "result.json")}}
    write(case / "result.json", record)
    print(json.dumps(record), flush=True)
    return record


def main():
    require(Path.cwd() == ROOT and not OUT.exists(), "Preserve existing monitor qualification")
    require(sha(OLD) == "76f5da360bbfafb91c4705144265d92dd913a285ae0c0500833b38e54f74fc37",
            "Original supervisor changed")
    preserved = preserved_functions()
    bindings = {str(p): sha(p) for p in (Path(__file__).resolve(), OLD, NEW,
                                       ROOT / "research/direct/latency58_windows_event_transport.py")}
    monitor = load("nvml_monitor_bindings", NEW)
    bindings.update(monitor.GPU_BINDINGS)
    require(all(sha(p) == digest for p, digest in bindings.items()), "Qualified NVML source changed")
    OUT.mkdir()
    write(OUT / "plan.json", {"schema": "latency58-nvml-monitor-cpu-control-v1",
          "source_bindings": bindings, "host_telemetry_simulated": True, "gpu_workload_started": False})
    cases = [run_case("delayed_clean_exit", expected_status="pass"),
             run_case("nonzero_exit", child_code=7, expected_status="child_failed", expected_reason="Child exit code 7"),
             run_case("incomplete_clean_exit", rows=(1,), delay=.05, expected_status="health_failed_after_child_exit",
                      expected_reason="before its planned final update", phase_expected=False),
             run_case("training_stall", rows=(1,), delay=5, expected_status="stopped_by_watchdog",
                      expected_reason="stopped reporting completed updates", phase_expected=False),
             run_case("finalization_timeout", delay=5, final_timeout=.8, expected_status="stopped_by_watchdog",
                      expected_reason="bounded finalization period"),
             run_case("runtime_limit", delay=5, max_runtime=.8, expected_status="stopped_by_watchdog",
                      expected_reason="runtime limit"),
             run_case("overshot_update", rows=(3,), delay=5, expected_status="stopped_by_watchdog",
                      expected_reason="exceeded the planned final update", phase_expected=False)]
    for fault, reason in (("temperature", "temperature reached"), ("memory", "free memory fell"),
                          ("identity", "identity, driver, or total memory changed"),
                          ("host_fault", "Fresh NVIDIA/Display reset"),
                          ("query_timeout", "windows_system telemetry timed out"),
                          ("event_worker_exit", "Simulated persistent event worker exited"),
                          ("gpu_query_timeout", "nvml telemetry timed out"),
                          ("gpu_worker_exit", "Simulated persistent NVML worker exited"),
                          ("physical_free_memory", "physical free memory fell"),
                          ("invalid_raw_memory", "NVML raw memory accounting changed")):
        cases.append(run_case(fault, delay=5, fault=fault, expected_status="stopped_by_watchdog", expected_reason=reason))
    for fault in ("gpu_close_nonzero", "gpu_close_forced", "gpu_close_identity", "gpu_close_exception"):
        cases.append(run_case(fault, fault=fault, expected_status="gpu_worker_cleanup_failed",
                              expected_reason="Persistent NVML worker did not close normally"))
    cases.append(run_case("gpu_preflight_failure", fault="gpu_preflight_failure", expected_status="blocked_preflight",
                          expected_reason="Simulated NVML construction failure", phase_expected=False))
    require(all(sha(p) == digest for p, digest in bindings.items()), "Monitor qualification inputs changed")
    write(OUT / "result.json", {"status": "pass", "plan_sha256": sha(OUT / "plan.json"),
          "source_bindings": bindings, "source_bindings_unchanged": True, "cases": cases,
          "unchanged_function_asts": preserved, "all_actual_cpu_children_reaped": True,
          "host_telemetry_simulated": True, "gpu_workload_started": False, "host_stability_proven": False})
    print(json.dumps({"status": "pass", "cases": len(cases), "gpu_workload_started": False}), flush=True)


if __name__ == "__main__":
    main()
