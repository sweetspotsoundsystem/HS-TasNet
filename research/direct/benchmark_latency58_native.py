"""Compare saved FP32/U8U8 graphs with the plugin's native ORT 1.26 CPU SDK.

Uses Python's standard library and a C++20 compiler; no Python ORT or Torch.
Run on each Mac with the same arguments. This times inference, not DAW playback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time

BASELINE_SHA = "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3"
CANDIDATE_SHA = "979314cfd480ea1b8d861fd210aecc1ba7c208a344d1deab423daf83f148c53a"


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True, help="Plugin libs/onnxruntime directory")
    parser.add_argument("--baseline", type=Path, required=True, help="Preserved C204 model.onnx")
    parser.add_argument("--candidate", type=Path, required=True, help="Saved m4-int8-screen-002/quantized.onnx")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--hops", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=4, choices=(2, 4))
    args = parser.parse_args()
    require(platform.system() in ("Darwin", "Linux"), "Probe supports macOS and Linux")
    require(platform.system() != "Darwin" or platform.machine() == "arm64",
            "Run natively on Apple silicon; a Rosetta measurement is not comparable")
    require(16 <= args.warmup <= 16384 and 64 <= args.hops <= 65536, "Hop count out of range")
    require(not args.output.exists(), "Preserve existing timing results")
    models = {"fp32": args.baseline.resolve(), "u8u8": args.candidate.resolve()}
    require(sha(models["fp32"]) == BASELINE_SHA and sha(models["u8u8"]) == CANDIDATE_SHA,
            "Timing must use the reviewed saved artifacts")
    sdk, source = args.sdk.resolve(), Path(__file__).with_suffix(".cpp").resolve()
    require((sdk / "include/onnxruntime_cxx_api.h").is_file(), "Missing C++ SDK headers")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Install the platform C++ compiler")
    libraries = sorted({p.resolve() for pattern in ("libonnxruntime*.so*", "libonnxruntime*.dylib")
                        for p in (sdk / "lib").glob(pattern) if p.is_file()})
    require(libraries, "Missing native ORT libraries")
    bindings = {str(path): sha(path) for path in [Path(__file__).resolve(), source, *models.values(), *libraries]}
    cpu = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
           if platform.system() == "Darwin" else platform.processor())
    plan = {"schema": "latency58-native-cpu1-comparison-v1", "source_bindings": bindings,
            "system": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "sdk": str(sdk), "expected_runtime_version": "1.26.0", "warmup_hops": args.warmup,
            "measured_hops_per_cycle": args.hops, "cycles": args.cycles,
            "input": "Deterministic stereo broadband/tone signal; identical for each fresh process",
            "scope": "Native preallocated ORT CPU1 inference; no DAW callback, queue, quality score or pacing",
            "graph_delay_samples": 128, "host_queue_samples": 128, "native_host_qualified": False}
    args.output.mkdir(parents=True)
    # Retain the exact portable probe with its evidence across later code edits.
    shutil.copyfile(source, args.output / source.name)
    shutil.copyfile(Path(__file__).resolve(), args.output / Path(__file__).name)
    write(args.output / "plan.json", plan)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    rows, began = [], time.monotonic()
    with tempfile.TemporaryDirectory(prefix="latency58-native-") as temporary:
        binary = Path(temporary) / "benchmark"
        command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
                   str(source), "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"),
                   "-Wl,-rpath," + str(sdk / "lib"), "-lonnxruntime", "-o", str(binary)]
        subprocess.run(command, check=True, env=environment)
        if platform.system() == "Darwin":
            binary_arch = subprocess.check_output(["lipo", "-archs", str(binary)], text=True).strip()
            require(binary_arch == "arm64", "Compiler produced a non-native benchmark")
        orders = (("fp32", "u8u8"), ("u8u8", "fp32"), ("u8u8", "fp32"), ("fp32", "u8u8"))
        for cycle, order in enumerate(orders[:args.cycles], start=1):
            for label in order:
                completed = subprocess.run([str(binary), str(models[label]), str(args.warmup), str(args.hops)],
                    check=True, text=True, capture_output=True, env=environment)
                row = {"cycle": cycle, "variant": label, **json.loads(completed.stdout)}
                require(row["status"] == "pass" and row["onnxruntime_version"] == "1.26.0",
                        "Native run failed or SDK differs from the plugin")
                row["runtime_stderr"] = completed.stderr
                write(args.output / f"cycle-{cycle}-{label}.json", row)
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("cycle", "variant", "run")}), flush=True)
    require(all(sha(path) == digest for path, digest in bindings.items()), "Bound input changed during measurement")
    medians = {label: statistics.median(row["run"]["p50_ms"] for row in rows if row["variant"] == label)
               for label in models}
    result = {"status": "pass", "plan_sha256": sha(args.output / "plan.json"), "cycles": rows,
              "median_of_cycle_p50_ms": medians, "candidate_over_baseline_ratio": medians["u8u8"] / medians["fp32"],
              "elapsed_seconds": time.monotonic() - began, "source_bindings_unchanged": True,
              "native_host_qualified": False,
              "limitations": "Unpaced synthetic inference. Full plugin callback/worker timing in the actual DAW remains necessary."}
    write(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "median_of_cycle_p50_ms", "candidate_over_baseline_ratio")}), flush=True)


if __name__ == "__main__":
    main()
