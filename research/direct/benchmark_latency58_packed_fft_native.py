"""Compare precise C204 integer graphs before and after synthesis FFT packing with the plugin's native ORT 1.26 CPU SDK.

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

BASELINE_SHA = "a550c904ef501fe98f3afa010eca5a53abd5bf66d63906e018511a01c5076d61"
CANDIDATE_SHA = "5b0437e94c9d334b7ec941c95641526ef721e38b6f8072dac43bfbbd96a25d38"


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
    parser.add_argument("--baseline", type=Path, required=True, help="Preserved precise C204 integer model.onnx")
    parser.add_argument("--candidate", type=Path, required=True, help="Saved m4-packed-fft-001/model.onnx")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--hops", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=6, choices=(6,))
    args = parser.parse_args()
    require(platform.system() in ("Darwin", "Linux"), "Probe supports macOS and Linux")
    require(platform.system() != "Darwin" or platform.machine() == "arm64",
            "Run natively on Apple silicon; a Rosetta measurement is not comparable")
    require(16 <= args.warmup <= 16384 and 64 <= args.hops <= 65536, "Hop count out of range")
    require(not args.output.exists(), "Preserve existing timing results")
    models = {"baseline": args.baseline.resolve(), "packed_fft": args.candidate.resolve()}
    require(sha(models["baseline"]) == BASELINE_SHA and sha(models["packed_fft"]) == CANDIDATE_SHA,
            "Timing must use the reviewed saved artifacts")
    sdk, source = args.sdk.resolve(), Path(__file__).with_name("benchmark_latency58_native.cpp").resolve()
    require((sdk / "include/onnxruntime_cxx_api.h").is_file(), "Missing C++ SDK headers")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Install the platform C++ compiler")
    libraries = sorted({p.resolve() for pattern in ("libonnxruntime*.so*", "libonnxruntime*.dylib")
                        for p in (sdk / "lib").glob(pattern) if p.is_file()})
    require(libraries, "Missing native ORT libraries")
    headers = sorted((sdk / "include").rglob("*.h"))
    bindings = {str(path): sha(path) for path in [Path(__file__).resolve(), source, *models.values(), *libraries, *headers]}
    cpu = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
           if platform.system() == "Darwin" else platform.processor())
    plan = {"schema": "latency58-native-cpu1-packed-fft-comparison-v1", "source_bindings": bindings,
            "system": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "sdk": str(sdk), "expected_runtime_version": "1.26.0", "warmup_hops": args.warmup,
            "measured_hops_per_cycle": args.hops, "cycles": args.cycles,
            "input": "Deterministic stereo broadband/tone signal; identical for each fresh process",
            "scope": "Native preallocated ORT CPU1 inference; no DAW callback, queue, quality score or pacing",
            "graph_delay_samples": 128, "host_queue_samples": 128, "native_host_qualified": False,
            "candidate_precision": "Same integer weights and FP64 quantizer ancestors; paired FP32 synthesis transforms",
            "advance_rule": "Candidate median of cycle medians <=0.97 baseline; faster in at least four of six cycles"}
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
        orders = (("baseline", "packed_fft"), ("packed_fft", "baseline"), ("packed_fft", "baseline"),
                  ("baseline", "packed_fft"), ("baseline", "packed_fft"), ("packed_fft", "baseline"))
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
    cycle_medians = {(row["cycle"], row["variant"]): row["run"]["p50_ms"] for row in rows}
    cycle_ratios = [cycle_medians[(i, "packed_fft")] / cycle_medians[(i, "baseline")]
                    for i in range(1, args.cycles + 1)]
    advance = medians["packed_fft"] / medians["baseline"] <= .97 and sum(r < 1 for r in cycle_ratios) >= 4
    result = {"status": "pass", "plan_sha256": sha(args.output / "plan.json"), "cycles": rows,
              "cycle_p50_ratios": cycle_ratios, "advance_to_long_parity_and_quality": advance,
              "median_of_cycle_p50_ms": medians, "candidate_over_baseline_ratio": medians["packed_fft"] / medians["baseline"],
              "elapsed_seconds": time.monotonic() - began, "source_bindings_unchanged": True,
              "native_host_qualified": False,
              "limitations": "Unpaced synthetic inference. Full plugin callback/worker timing in the actual DAW remains necessary."}
    write(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "median_of_cycle_p50_ms", "candidate_over_baseline_ratio")}), flush=True)


if __name__ == "__main__":
    main()
