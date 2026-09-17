"""Qualify required NVML getter coverage using CPU fakes; never load a native library."""
from __future__ import annotations

import ast
import ctypes as c
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "research/direct"
OLD = DIRECT / "latency58_nvml_reader.py"
NEW = DIRECT / "latency58_nvml_guard_reader.py"
WATCH = DIRECT / "runs/latency11/smoke/gpu-crash-followup/watch_gpu_process_nvml.py"
OUT = DIRECT / "runs/latency58/nvml-guard-reader-cpu-001"
REQUIRED = ("nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetUUID",
            "nvmlDeviceGetName", "nvmlSystemGetDriverVersion", "nvmlDeviceGetMemoryInfo_v2",
            "nvmlDeviceGetTemperature")
SUPPLEMENTAL = ("nvmlDeviceGetPowerUsage", "nvmlDeviceGetPowerManagementLimit", "nvmlDeviceGetUtilizationRates")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLibrary:
    def __init__(self, module, *, allow_supplemental=False, failing_api=None, code=0, fault=None):
        self.module, self.allow_supplemental = module, allow_supplemental
        self.failing_api, self.code, self.fault = failing_api, code, fault
        self.bound, self.called = [], []

    def __getattr__(self, name):
        require(name in ("nvmlInit_v2", "nvmlShutdown", *REQUIRED, *SUPPLEMENTAL), "Unreviewed API bound")
        require(self.allow_supplemental or name not in SUPPLEMENTAL, "Supplemental API must not even be bound")
        self.bound.append(name)
        def getter(*args):
            self.called.append(name)
            if name == self.failing_api:
                return self.code
            if name == "nvmlDeviceGetCount_v2":
                c.cast(args[0], c.POINTER(c.c_uint)).contents.value = 2 if self.fault == "count" else 1
            elif name == "nvmlDeviceGetHandleByIndex_v2":
                c.cast(args[1], c.POINTER(c.c_void_p)).contents.value = 0 if self.fault == "handle" else 1234
            elif name in ("nvmlDeviceGetUUID", "nvmlDeviceGetName", "nvmlSystemGetDriverVersion"):
                value = {"nvmlDeviceGetUUID": b"GPU-fixture", "nvmlDeviceGetName": b"CPU fixture",
                         "nvmlSystemGetDriverVersion": b"fixture-driver"}[name]
                if self.fault == "uuid" and name == "nvmlDeviceGetUUID":
                    value = b"invalid"
                if self.fault == "empty_driver" and name == "nvmlSystemGetDriverVersion":
                    value = b""
                args[-2].value = value
            elif name == "nvmlDeviceGetMemoryInfo_v2":
                memory = c.cast(args[1], c.POINTER(self.module.MemoryV2)).contents
                require(memory.version == 0x02000028, "Memory ABI version changed")
                memory.total, memory.reserved, memory.used = 34190917632, 437256192, 1217794048
                memory.free = memory.total - memory.reserved - memory.used
                if self.fault == "memory_sum":
                    memory.free += 1
                elif self.fault == "memory_sentinel":
                    memory.used = 2**64 - 1
                elif self.fault == "memory_zero":
                    memory.total = memory.reserved = memory.used = memory.free = 0
            elif name == "nvmlDeviceGetTemperature":
                c.cast(args[2], c.POINTER(c.c_uint)).contents.value = 45
            elif name in ("nvmlDeviceGetPowerUsage", "nvmlDeviceGetPowerManagementLimit"):
                c.cast(args[1], c.POINTER(c.c_uint)).contents.value = 37000 if name.endswith("PowerUsage") else 575000
            elif name == "nvmlDeviceGetUtilizationRates":
                utilization = c.cast(args[1], c.POINTER(self.module.Utilization)).contents
                utilization.gpu, utilization.memory = 2, 3
            return 0
        return getter


def exercise(module, name, *, allow_supplemental=False, failing_api=None, code=0, fault=None, expected_error=False):
    fake = FakeLibrary(module, allow_supplemental=allow_supplemental, failing_api=failing_api, code=code, fault=fault)
    reader, error, samples = None, None, []
    with patch.object(c, "CDLL", return_value=fake) as loader:
        try:
            reader = module.ReadOnlyNvml()
            for _ in range(3):
                samples.append(reader.sample())
        except RuntimeError as caught:
            error = str(caught)
        finally:
            if reader is not None:
                try:
                    reader.close()
                    reader.close()
                except RuntimeError as caught:
                    error = str(caught)
        require(loader.call_count == 1, "Expected exactly one intercepted CDLL construction")
    require(bool(error) == expected_error, "Unexpected fixture outcome: " + name)
    require(fake.called.count("nvmlInit_v2") == 1
            and fake.called.count("nvmlShutdown") == (0 if failing_api == "nvmlInit_v2" else 1),
            "Library initialization/cleanup inventory differs: " + name)
    if not allow_supplemental:
        require(not set(SUPPLEMENTAL).intersection(fake.bound + fake.called), "A supplemental API was touched")
    if not expected_error:
        expected_calls = ["nvmlInit_v2", *(list(REQUIRED) + (list(SUPPLEMENTAL) if allow_supplemental else [])) * 3, "nvmlShutdown"]
        require(fake.called == expected_calls and len(samples) == 3, "Required sampling coverage changed")
    return {"case": name, "status": "pass", "error": error, "bound_apis": fake.bound,
            "called_apis": fake.called, "samples": samples, "native_library_loaded": False, "gpu_queried": False}


def main():
    require(Path.cwd() == ROOT and not OUT.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Run on CPU with CUDA hidden and preserve previous qualification")
    require(sha(OLD) == "6470883f05e07db05b9ce3910104715e39963ca124d949fc3762bfef05556f10"
            and sha(WATCH) == "76f5da360bbfafb91c4705144265d92dd913a285ae0c0500833b38e54f74fc37",
            "Current production reader or monitor changed")
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(), OLD, NEW, WATCH)}
    trees = [ast.parse(p.read_text()) for p in (OLD, NEW)]
    for name in ("MemoryV2", "Utilization", "require", "sha", "validate_memory"):
        nodes = [next(n for n in tree.body if getattr(n, "name", None) == name) for tree in trees]
        require(ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False),
                "ABI or validation function changed: " + name)
    classes = [next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReadOnlyNvml") for tree in trees]
    for name in ("__init__", "_call", "_text", "close"):
        nodes = [next(n for n in cls.body if getattr(n, "name", None) == name) for cls in classes]
        require(ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False),
                "Reader lifecycle or required return-code check changed: " + name)
    with patch.object(c, "CDLL", side_effect=RuntimeError("Import attempted native loading")):
        old, new, monitor = load("legacy_reader_cpu", OLD), load("guard_reader_cpu", NEW), load("existing_monitor_cpu", WATCH)
    require(set(new.SIGNATURES) == {"nvmlInit_v2", "nvmlShutdown", *REQUIRED}
            and not new.OPTIONAL, "Required operations must all fail on any nonzero return code")
    OUT.mkdir()
    write(OUT / "plan.json", {"schema": "latency58-nvml-guard-reader-cpu-v1", "source_bindings": sources,
                              "native_gpu_queries_planned": False, "active_monitor_switched": False})
    cases = [exercise(old, "legacy_success", allow_supplemental=True), exercise(new, "required_success")]
    for left, right in zip(cases[0]["samples"], cases[1]["samples"], strict=True):
        require(left["memory_bytes"] == right["memory_bytes"]
                and all(left["values"][key] == right["values"][key] for key in new.FIELDS[:6]),
                "Required values differ from the qualified production reader")
        require(all(right["values"][key] == "not_queried" for key in new.FIELDS[6:])
                and right["sampling_policy"] == new.SAMPLING_POLICY
                and right["omitted_supplemental_getters"] == list(SUPPLEMENTAL), "Omitted counter annotation is incomplete")
        stream = io.StringIO()
        csv.writer(stream).writerow([right["values"][key] for key in new.FIELDS])
        parsed = monitor.parse_gpu(stream.getvalue())
        require(monitor.gpu_alert(parsed, parsed, max_temperature=80, memory_headroom=4096) is None,
                "Existing parser or threshold contract rejected the sample")
    for api in ("nvmlInit_v2", *REQUIRED, "nvmlShutdown"):
        cases.append(exercise(new, "unsupported_" + api, failing_api=api, code=3, expected_error=True))
    for code in (15, 999):
        cases.append(exercise(new, "temperature_error_" + str(code), failing_api="nvmlDeviceGetTemperature",
                              code=code, expected_error=True))
    for fault in ("count", "handle", "uuid", "empty_driver", "memory_sum", "memory_sentinel", "memory_zero"):
        cases.append(exercise(new, fault, fault=fault, expected_error=True))
    require(all(sha(path) == digest for path, digest in sources.items()), "Qualification source changed")
    write(OUT / "result.json", {"status": "pass", "source_bindings": sources, "source_bindings_unchanged": True,
        "plan_sha256": sha(OUT / "plan.json"), "cases": cases, "required_values_equal_legacy": True,
        "all_required_getters_fail_closed": True, "supplemental_getters_never_bound_or_called": True,
        "existing_parser_and_threshold_contract_preserved": True,
        "native_library_loaded": False, "gpu_queried": False, "active_monitor_switched": False,
        "guard_reader_worker_and_real_host_runtime_qualified": False})
    print(json.dumps({"status": "pass", "cases": len(cases), "native_library_loaded": False,
                      "active_monitor_switched": False}), flush=True)


if __name__ == "__main__":
    main()
