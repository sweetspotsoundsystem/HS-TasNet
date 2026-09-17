"""Exercise soak-runner evidence rejection using explicit simulated Mac commands.

This is a shell-control test on Linux. No audio, native Mac binary, model
inference, timing, or physical-machine qualification is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


SHIM = r'''#!/usr/bin/env python3
import os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
if name == "uname":
    print("Darwin" if sys.argv[1] == "-s" else "arm64")
elif name == "lipo":
    print("arm64")
elif name == "git":
    if "rev-parse" in sys.argv:
        print("0" * 40)
elif name == "cmake":
    print("model_sha256=" + os.environ["TRACE_FIXTURE_MODEL_SHA"])
    print("model_bytes=" + os.environ["TRACE_FIXTURE_MODEL_BYTES"])
    print("sample_rate=44100\ncallback_samples=128\npdc_samples=256")
elif name == "stat":
    assert sys.argv[1:3] == ["-f", "%z"]
    print(pathlib.Path(sys.argv[3]).stat().st_size)
else:
    print("simulated_mac_shell_control_fixture_no_hardware_measurement")
'''

BINARY = r'''#!/usr/bin/env python3
import os, sys
case = os.environ["TRACE_FIXTURE_CASE"]
traced = os.environ["STEMGENRT_TRACE_WORKER"] == "1"
assert os.environ["STEMGENRT_QUALIFICATION_CALLBACKS"] == "620157"
assert os.environ["STEMGENRT_PACED_ORT_THREADS"] == "0"
mode = "enabled" if traced else "disabled"
summary = ("STEMGENRT_QUALIFICATION_SUMMARY status=pass warmup_callbacks=100 "
           "measured_callbacks=620157 worker_trace=" + mode +
           " ort_intra_op_threads_override=0 ort_intra_op_threads=1 "
           "callback_samples=128 sample_rate=44100 pdc_samples=256 deadline_us=2902.494")
print(summary)
if case == "duplicate_summary":
    print(summary)
print("STEMGENRT_PACED_FAILURE_PHASES measured_underrun_samples=0 "
      "measured_due_boundary_misses=0 retained_events=0 omitted_events=0")
if traced and case != "missing_worker":
    fields = dict(scope="matched_measured_due_requests", traced_samples="620258",
                  omitted_samples="0", storage_complete="1",
                  measured_due_requests="620157", matched_measured_requests="620157",
                  missing_measured_requests="0", duplicate_sequences="0", run_max_us="1000")
    if case == "old_truncated_prefix":
        fields.update(traced_samples="100000", omitted_samples="520258", storage_complete="0",
                      matched_measured_requests="99899", missing_measured_requests="520258")
    if case == "missing_request":
        fields.update(matched_measured_requests="620156", missing_measured_requests="1")
    if case == "old_scope":
        fields["scope"] = "measured_due_requests"
    if case == "duplicate_sequence":
        fields["duplicate_sequences"] = "1"
    worker = "STEMGENRT_WORKER_TIMING " + " ".join(k + "=" + v for k, v in fields.items())
    print(worker)
    if case == "duplicate_worker":
        print(worker)
if case == "child_exit_second" and any("repetition-2.xml" in x for x in sys.argv):
    sys.exit(3)
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    assert not output.exists(), "Preserve previous test evidence"
    output.mkdir()
    fixture = output / "fixture"
    shims = output / "simulated-mac-commands"
    fixture.mkdir()
    shims.mkdir()
    runner = fixture / "scripts/extended-soak-macos.sh"
    runner.parent.mkdir()
    shutil.copyfile(source / "scripts/extended-soak-macos.sh", runner)
    for relative in (
        "cmake/QualifiedModelContract.cmake", "test/source/RealtimeStemSanityTest.cpp",
        "test/source/PacedQualificationTrace.h", "plugin/source/InferenceQueue.cpp",
        "plugin/source/OnnxRuntime.cpp", "plugin/include/StemgenRT/WorkerTimingTrace.h",
        "libs/onnxruntime/lib/libonnxruntime.dylib", "model/model.onnx",
    ):
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Simulated shell control fixture. Not a model or native Mac runtime.\n")
    binary = fixture / "build-release/test/AudioPluginTest"
    binary.parent.mkdir(parents=True)
    binary.write_text(BINARY)
    binary.chmod(0o700)
    for name in ("uname", "lipo", "git", "cmake", "stat", "sw_vers", "sysctl", "pmset"):
        path = shims / name
        path.write_text(SHIM)
        path.chmod(0o700)
    model = fixture / "model/model.onnx"
    env = {**os.environ, "PATH": str(shims) + os.pathsep + os.environ["PATH"],
           "TRACE_FIXTURE_MODEL_SHA": sha(model), "TRACE_FIXTURE_MODEL_BYTES": str(model.stat().st_size)}
    cases = [("untraced", False, 0), ("complete", True, 0)]
    cases.extend((name, True, 1) for name in (
        "old_truncated_prefix", "missing_worker", "missing_request", "old_scope",
        "duplicate_sequence", "duplicate_worker", "duplicate_summary", "child_exit_second"))
    results = []
    for name, traced, expected in cases:
        destination = output / name
        argv = ["bash", str(runner), str(fixture), str(destination)]
        if traced:
            argv.append("--trace")
        result = subprocess.run(argv, env={**env, "TRACE_FIXTURE_CASE": name},
                                capture_output=True, text=True, timeout=30)
        (output / (name + "-console.log")).write_text(result.stdout + result.stderr)
        status = (destination / "status.txt").read_text().strip() if (destination / "status.txt").exists() else None
        expected_status = ("synthetic_host_traced_diagnostic_" if traced else "synthetic_host_soak_")
        expected_status += "passed" if expected == 0 else "failed_or_incomplete"
        exits = [int((destination / f"repetition-{n}-exit-code.txt").read_text()) for n in (1, 2)]
        assert exits == ([0, 3] if name == "child_exit_second" else [0, 0]), (name, exits)
        assert result.returncode == expected and status == expected_status, (name, result, status)
        results.append({"case": name, "actual_exit_code": result.returncode, "expected_exit_code": expected,
                        "status": status, "retained_child_exits": exits})
        print(json.dumps(results[-1]), flush=True)
    report = {"status": "pass", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": sha(runner), "checker_sha256": sha(Path(__file__)), "cases": results,
              "macos_commands_simulated": True, "native_mac_execution": False,
              "inference_or_timing_performed": False,
              "scope": "Only runner argument propagation, log acceptance/rejection, and retention of actual child exit codes."}
    assert sha(source / "scripts/extended-soak-macos.sh") == report["runner_sha256"]
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
