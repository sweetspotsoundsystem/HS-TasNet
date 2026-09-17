"""Portable native CPU1 comparison of v0.4.0 and the four-GRU experiment.

Requires only Python's standard library, a C++20 compiler and the ORT 1.26 CPU
SDK. This compares preallocated inference; it does not qualify DAW playback.
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

IDENTITIES = {
    "released": "d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa",
    "candidate": "878c74694fa4c558de1c5a75837893a0afeadcf57f6e3b860d5904cab04e9fc9",
}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--released", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--hops", type=int, default=2048)
    parser.add_argument("--background-load", default="uncontrolled")
    args = parser.parse_args()
    require(platform.system() in ("Darwin", "Linux") and
            (platform.system() != "Darwin" or platform.machine() == "arm64"),
            "Use native Apple Silicon or a clearly labelled Linux comparison")
    require(16 <= args.warmup <= 16384 and 64 <= args.hops <= 65536, "Invalid hop counts")
    out, sdk = args.output.resolve(), args.sdk.resolve(strict=True)
    require(not out.exists(), "Preserve existing comparison evidence")
    models = {"released": args.released.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
    require(all(sha(p) == IDENTITIES[k] for k, p in models.items()), "Use the exact declared saved graphs")
    source = Path(__file__).with_name("benchmark_latency58_branch_native.cpp").resolve(strict=True)
    headers = sorted((sdk / "include").rglob("*.h"))
    libraries = sorted({p.resolve() for pattern in ("libonnxruntime*.so*", "libonnxruntime*.dylib")
                        for p in (sdk / "lib").glob(pattern) if p.is_file()})
    require(headers and libraries and (sdk / "include/onnxruntime_cxx_api.h").is_file(), "Missing native CPU SDK")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "C++ compiler unavailable")
    bindings = {str(p): sha(p) for p in [Path(__file__).resolve(), source, *models.values(), *headers, *libraries]}
    cpu = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
           if platform.system() == "Darwin" else next((line.split(":", 1)[1].strip()
               for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()))
    plan = {"schema": "latency58-eight-state-native-pair-v1", "source_bindings": bindings,
        "models": {k: {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for k, p in models.items()},
        "sdk": str(sdk), "system": platform.platform(), "machine": platform.machine(), "cpu": cpu,
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "expected_runtime_version": "1.26.0", "warmup_hops": args.warmup,
        "measured_hops_per_block": args.hops, "ordering": ["released", "candidate", "candidate", "released"] * 2,
        "background_load_declared": args.background_load, "inference_threads": 1,
        "input": "Identical deterministic broadband/tone stream in each fresh child process",
        "scope": "Native preallocated nine-input/nine-output inference and state copies/checks; no plugin queue or host pacing",
        "native_host_qualified": False, "quality_measured": False, "audio_saved": False,
        "source_code_copies_retained": True}
    out.mkdir(parents=True)
    shutil.copyfile(source, out / source.name)
    shutil.copyfile(Path(__file__).resolve(), out / Path(__file__).name)
    write(out / "plan.json", plan)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    rows, began = [], time.monotonic()
    with tempfile.TemporaryDirectory(prefix="build-", dir=out) as directory:
        binary = Path(directory) / "benchmark"
        argv = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
            str(source), "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"),
            "-Wl,-rpath," + str(sdk / "lib"), "-lonnxruntime", "-o", str(binary)]
        build = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=120)
        write(out / "compile.json", {"argv": argv, "actual_exit_code": build.returncode,
            "stdout": build.stdout, "stderr": build.stderr})
        require(build.returncode == 0, "Native compilation failed; inspect compile.json")
        if platform.system() == "Darwin":
            require(subprocess.check_output(["lipo", "-archs", str(binary)], text=True).strip() == "arm64", "Compiler architecture differs")
        binary_sha = sha(binary)
        for index, label in enumerate(plan["ordering"]):
            argv = [str(binary), str(models[label]), str(args.warmup), str(args.hops)]
            completed = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=1800)
            receipt = {"block": index, "variant": label, "argv": argv,
                "actual_exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
            write(out / ("block-%02d-execution.json" % index), receipt)
            require(completed.returncode == 0, "Native block failed; inspect retained execution")
            row = {"block": index, "variant": label, **json.loads(completed.stdout)}
            require(row["status"] == "pass" and row["onnxruntime_version"] == "1.26.0"
                    and row["intra_op_threads"] == row["inter_op_threads"] == 1
                    and row["preallocated_tensors"] and not row["spinning"]
                    and row["warmup_hops"] == args.warmup and row["measured_hops"] == args.hops,
                    "Native runtime or configuration changed")
            write(out / ("block-%02d.json" % index), row)
            rows.append(row)
            print(json.dumps({"block": index, "variant": label, "run": row["run"]}), flush=True)
    require(all(sha(p) == h for p, h in bindings.items()), "Comparison inputs changed")
    medians = {label: statistics.median(r["run"]["p50_ms"] for r in rows if r["variant"] == label) for label in models}
    result = {"status": "pass", "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "binary_sha256": binary_sha, "blocks": rows, "median_of_block_p50_ms": medians,
        "candidate_over_released_ratio": medians["candidate"] / medians["released"],
        "background_load_declared": args.background_load, "system": platform.platform(), "cpu": cpu,
        "elapsed_seconds": time.monotonic() - began, "native_host_qualified": False,
        "quality_measured": False, "gpu_used": False,
        "limitations": "Unpaced native inference only. Device identity and declared background load apply to this run. Full AU callback/worker timing and installed DAW playback remain separate requirements."}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "median_of_block_p50_ms", "candidate_over_released_ratio")}), flush=True)


if __name__ == "__main__":
    main()
