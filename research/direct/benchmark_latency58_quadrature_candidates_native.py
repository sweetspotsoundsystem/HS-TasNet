"""Compare C204 FP32 and the checked C204 and quadrature precise U8U8 candidates with the plugin's native ORT 1.26 CPU SDK.

Uses Python's standard library and a C++20 compiler; no Python ORT or Torch.
Run on each Mac with the same arguments. This times inference, not DAW playback.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
C204_INT8_SHA = "a550c904ef501fe98f3afa010eca5a53abd5bf66d63906e018511a01c5076d61"
QUADRATURE_INT8_SHA = "6cfcc9d9ad70473dcaf9ce822af413e3d01fc303820798b8b726976e0b08cfcf"


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
    parser.add_argument("--c204-int8", type=Path, required=True, help="Saved precise C204 integer graph")
    parser.add_argument("--quadrature-int8", type=Path, required=True, help="Saved precise quadrature-magnitude integer graph")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--hops", type=int, default=2048)
    args = parser.parse_args()
    require(platform.system() in ("Darwin", "Linux"), "Probe supports macOS and Linux")
    require(platform.system() != "Darwin" or platform.machine() == "arm64",
            "Run natively on Apple silicon; a Rosetta measurement is not comparable")
    require(16 <= args.warmup <= 16384 and 64 <= args.hops <= 65536, "Hop count out of range")
    require(not args.output.exists(), "Preserve existing timing results")
    models = {"fp32_c204": args.baseline.resolve(), "int8_c204": args.c204_int8.resolve(),
              "int8_quadrature": args.quadrature_int8.resolve()}
    require(sha(models["fp32_c204"]) == BASELINE_SHA and sha(models["int8_c204"]) == C204_INT8_SHA
            and sha(models["int8_quadrature"]) == QUADRATURE_INT8_SHA,
            "Timing must use the reviewed saved artifacts")
    sdk, source = args.sdk.resolve(), Path(__file__).with_name("benchmark_latency58_native.cpp").resolve()
    require((sdk / "include/onnxruntime_cxx_api.h").is_file(), "Missing C++ SDK headers")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Install the platform C++ compiler")
    libraries = sorted({p.resolve() for pattern in ("libonnxruntime*.so*", "libonnxruntime*.dylib")
                        for p in (sdk / "lib").glob(pattern) if p.is_file()})
    require(libraries, "Missing native ORT libraries")
    headers = sorted((sdk / "include").rglob("*.h"))
    require(headers, "Missing SDK headers")
    bindings = {str(path): sha(path) for path in [Path(__file__).resolve(), source, *models.values(), *headers, *libraries]}
    cpu = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
           if platform.system() == "Darwin" else platform.processor())
    plan = {"schema": "latency58-native-cpu1-three-model-comparison-v1", "source_bindings": bindings,
            "system": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "sdk": str(sdk), "expected_runtime_version": "1.26.0", "warmup_hops": args.warmup,
            "measured_hops_per_cycle": args.hops, "cycles": 6,
            "ordering": "All six permutations; each model occupies each position twice",
            "input": "Deterministic stereo broadband/tone signal; identical for each fresh process",
            "scope": "Native preallocated ORT CPU1 inference; no DAW callback, queue, quality score or pacing",
            "graph_delay_samples": 128, "host_queue_samples": 128, "native_host_qualified": False,
            "candidate_precision": "C204: nine U8U8 projections and FP64 quantizer ancestors; quadrature: ten U8U8 projections including magnitude and FP64 quantizer ancestors; both: FP32 public state and decoding; quadrature phase factors stay FP32"}
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
        orders = tuple(itertools.permutations(models))
        for cycle, order in enumerate(orders, start=1):
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
              "median_of_cycle_p50_ms": medians,
              "ratios_to_fp32_c204": {label: value / medians["fp32_c204"] for label, value in medians.items()},
              "quadrature_over_c204_integer_ratio": medians["int8_quadrature"] / medians["int8_c204"],
              "elapsed_seconds": time.monotonic() - began, "source_bindings_unchanged": True,
              "native_host_qualified": False,
              "limitations": "Unpaced synthetic inference. Full plugin callback/worker timing in the actual DAW remains necessary."}
    write(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "median_of_cycle_p50_ms", "ratios_to_fp32_c204", "quadrature_over_c204_integer_ratio")}), flush=True)


if __name__ == "__main__":
    main()
