"""Run the unchanged NVML worker/transport against an owned CPU-only C fixture."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "research/direct"
OUT = DIRECT / "runs/latency58/nvml-guard-worker-cpu-001"
READER = DIRECT / "latency58_nvml_guard_reader.py"
WORKER = DIRECT / "latency58_nvml_worker.py"
TRANSPORT = DIRECT / "latency58_nvml_transport.py"
WATCH = DIRECT / "runs/latency11/smoke/gpu-crash-followup/watch_gpu_process_nvml.py"
POLICY = "latency58-nvml-required-health-getters-v1"
REQUIRED = ["nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetUUID",
            "nvmlDeviceGetName", "nvmlSystemGetDriverVersion", "nvmlDeviceGetMemoryInfo_v2",
            "nvmlDeviceGetTemperature"]
C_SOURCE = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { unsigned version; uint64_t total, reserved, free, used; } Memory;
_Static_assert(sizeof(Memory) == 40 && offsetof(Memory, total) == 8 && offsetof(Memory, used) == 32, "ABI");
static int initialized = 0;
static int fault(const char *name) {
    const char *value = getenv("LATENCY58_CPU_NVML_FIXTURE_FAULT");
    return value != NULL && strcmp(value, name) == 0;
}
static int text(char *buffer, unsigned length, const char *value) {
    if (!initialized) return 1;
    if (length <= strlen(value)) return 7;
    strcpy(buffer, value);
    return 0;
}
int nvmlInit_v2(void) { if (fault("init_error")) return 3; initialized = 1; return 0; }
int nvmlShutdown(void) { if (!initialized) return 1; initialized = 0; return 0; }
int nvmlDeviceGetCount_v2(unsigned *count) { if (!initialized) return 1; *count = 1; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned index, void **device) {
    if (!initialized || index != 0) return 1;
    *device = (void *)(uintptr_t)1234;
    return 0;
}
int nvmlDeviceGetUUID(void *device, char *buffer, unsigned length) {
    (void)device; return text(buffer, length, "GPU-cpu-native-fixture");
}
int nvmlDeviceGetName(void *device, char *buffer, unsigned length) {
    (void)device; return text(buffer, length, "CPU fixture, no GPU access");
}
int nvmlSystemGetDriverVersion(char *buffer, unsigned length) {
    return text(buffer, length, "cpu-fixture-driver");
}
int nvmlDeviceGetMemoryInfo_v2(void *device, Memory *memory) {
    (void)device;
    if (!initialized || memory->version != 0x02000028) return 1;
    memory->total = UINT64_C(34190917632);
    memory->reserved = UINT64_C(437256192);
    memory->used = UINT64_C(1217794048);
    memory->free = memory->total - memory->reserved - memory->used;
    if (fault("bad_memory")) memory->free++;
    return 0;
}
#ifndef OMIT_TEMPERATURE
int nvmlDeviceGetTemperature(void *device, unsigned sensor, unsigned *temperature) {
    (void)device; (void)sensor;
    if (!initialized) return 1;
    if (fault("temperature_error")) return 15;
    if (fault("temperature_timeout")) {
        struct timespec delay = {3, 0}; nanosleep(&delay, NULL);
    }
    *temperature = 45;
    return 0;
}
#endif
/* There are deliberately no power/utilization exports and no GPU library links. */
'''


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(module, monitor, name, library, *, fault=None, failure=False, timeout=False):
    original = os.environ.get("LATENCY58_CPU_NVML_FIXTURE_FAULT")
    if fault:
        os.environ["LATENCY58_CPU_NVML_FIXTURE_FAULT"] = fault
    else:
        os.environ.pop("LATENCY58_CPU_NVML_FIXTURE_FAULT", None)
    transport = module.PersistentNvmlQuery(worker_source=WORKER, worker_sha256=sha(WORKER),
        reader_source=READER, reader_sha256=sha(READER), library=library, library_sha256=sha(library))
    responses, error, retry_rejected = [], None, False
    try:
        try:
            for _ in range(3):
                responses.append(transport.query(1 if timeout else 3))
        except (RuntimeError, subprocess.TimeoutExpired) as caught:
            error = caught
        require((error is not None) == failure, "Unexpected worker result: " + name)
        pid = transport.process.pid
        if failure:
            try:
                transport.query(.5)
            except RuntimeError:
                retry_rejected = True
            require(retry_rejected and transport.process.pid == pid, "Failed worker was restarted")
            require(isinstance(error, subprocess.TimeoutExpired) == timeout, "Wrong failure classification")
        else:
            require(len(responses) == 3 and [r["request_id"] for r in responses] == [1, 2, 3]
                    and {r["worker_pid"] for r in responses} == {pid}, "Owned PID/request sequence changed")
            for response in responses:
                sample = response["nvml"]
                require(sample["sampling_policy"] == POLICY
                        and [row["api"] for row in sample["api_timings"]] == REQUIRED
                        and all(sample["values"][key] == "not_queried" for key in
                                ("power.draw", "power.limit", "utilization.gpu")), "Worker changed getter coverage")
                parsed = monitor.parse_gpu(response["stdout"])
                require(monitor.gpu_alert(parsed, parsed, max_temperature=80, memory_headroom=4096) is None
                        and sample["memory_bytes"]["free"] >= 4096 * 2**20
                        and parsed["name"] == "CPU fixture, no GPU access", "CSV/memory contract differs")
    finally:
        closed = transport.close(timeout=.3 if timeout else 3)
        if original is None:
            os.environ.pop("LATENCY58_CPU_NVML_FIXTURE_FAULT", None)
        else:
            os.environ["LATENCY58_CPU_NVML_FIXTURE_FAULT"] = original
    require(closed["closed"] and closed["identities_unchanged"]
            and not Path("/proc", str(closed["linux_pid"])).exists(), "Owned CPU worker remains")
    require((closed["forced"] and closed["actual_exit_code"] == -15) if timeout else
            (not closed["forced"] and closed["actual_exit_code"] == (1 if failure else 0)),
            "Worker cleanup classification differs")
    result = {"case": name, "status": "pass", "error": repr(error) if error else None,
              "retry_rejected": retry_rejected, "close": closed, "responses": responses,
              "library_path": str(library), "native_library_is_cpu_fixture": True,
              "nvidia_library_loaded": False, "gpu_queried": False, "active_monitor_switched": False}
    write(OUT / (name + ".json"), result)
    return result


def main():
    require(Path.cwd() == ROOT and not OUT.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Preserve earlier qualification and hide CUDA")
    expected = {READER: "d1d236cafc36a92f6faa6869176bd4d85ec273fb0f6cc92b794d53d0dfaffaed",
        WORKER: "f3a93b09e6cc46329479ff262e6d1b34b7a3f5d95f3db9aa7deb1a23617ea15a",
        TRANSPORT: "a8373cc3aa4e26e94a84b50a6ec8492e68bf33023f836863a4d2c5f8f085b35c",
        WATCH: "76f5da360bbfafb91c4705144265d92dd913a285ae0c0500833b38e54f74fc37"}
    require(all(sha(path) == digest for path, digest in expected.items()), "Qualified inputs changed")
    OUT.mkdir(); source = OUT / "cpu_nvml_fixture.c"; source.write_text(C_SOURCE)
    libraries = []
    for variant, flags in (("complete", []), ("missing_temperature", ["-DOMIT_TEMPERATURE"])):
        library = OUT / (variant + ".so")
        argv = ["/usr/bin/cc", "-std=c11", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror",
                *flags, str(source), "-o", str(library)]
        began = time.monotonic()
        compiled = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=20)
        write(OUT / (variant + "-compile.json"), {"argv": argv, "actual_exit_code": compiled.returncode,
            "stdout": compiled.stdout, "stderr": compiled.stderr, "elapsed_seconds": time.monotonic() - began})
        require(compiled.returncode == 0, "CPU fixture compilation failed")
        libraries.append(library)
    paths = [Path(__file__).resolve(), source, *libraries, *expected,
             DIRECT / "latency58_windows_event_transport.py"]
    bindings = {str(path): sha(path) for path in paths}
    write(OUT / "plan.json", {"schema": "latency58-nvml-guard-worker-cpu-v1", "source_bindings": bindings,
                              "nvidia_library_loaded": False, "active_monitor_switched": False})
    module, monitor = load("guard_transport_cpu", TRANSPORT), load("guard_existing_monitor_cpu", WATCH)
    cases = [run_case(module, monitor, "healthy", libraries[0])]
    for fault in ("init_error", "temperature_error", "bad_memory"):
        cases.append(run_case(module, monitor, fault, libraries[0], fault=fault, failure=True))
    cases.append(run_case(module, monitor, "missing_required_symbol", libraries[1], failure=True))
    cases.append(run_case(module, monitor, "temperature_timeout", libraries[0],
                          fault="temperature_timeout", failure=True, timeout=True))
    require(all(sha(path) == digest for path, digest in bindings.items()), "Qualification source changed")
    write(OUT / "result.json", {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
        "plan_sha256": sha(OUT / "plan.json"), "cases": cases, "all_owned_cpu_workers_reaped": True,
        "guard_reader_real_worker_and_pipe_protocol_qualified": True,
        "nvidia_library_loaded": False, "gpu_queried": False, "active_monitor_switched": False,
        "real_host_runtime_qualified": False})
    print(json.dumps({"status": "pass", "cases": len(cases), "all_owned_cpu_workers_reaped": True,
                      "gpu_queried": False, "active_monitor_switched": False}), flush=True)


if __name__ == "__main__":
    main()
