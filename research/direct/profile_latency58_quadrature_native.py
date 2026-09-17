"""Locate native CPU operator costs in the saved C204 and quadrature integer graphs."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs


def summarize(events):
    runs = sorted((e for e in events if e.get("name") == "model_run" and e.get("ph") == "X"),
                  key=lambda e: e["ts"])
    require(len(runs) == 32, "Require all 16 warmup and 16 diagnostic calls")
    selected = runs[16:]
    nodes = [e for e in events if e.get("cat") == "Node" and e.get("ph") == "X"
             and e.get("name", "").endswith("_kernel_time")
             and any(r["ts"] <= e["ts"] and e["ts"] + e["dur"] <= r["ts"] + r["dur"] for r in selected)]
    require(nodes and all(e["args"]["provider"] == "CPUExecutionProvider" for e in nodes),
            "Require native CPU kernel events within measured calls")
    grouped, operators = collections.defaultdict(list), collections.defaultdict(list)
    for event in nodes:
        grouped[(event["name"], event["args"]["op_name"])].append(event)
        operators[event["args"]["op_name"]].append(event["dur"])
    require(all(len(group) == 16 for group in grouped.values()), "Operator coverage differs across calls")
    total = sum(e["dur"] for e in nodes)
    require(total > 0, "Profile has no measured CPU time")
    rows = []
    for (name, operator), group in grouped.items():
        durations = [e["dur"] for e in group]
        rows.append({"name": name, "operator": operator, "calls": len(group),
                     "median_us": statistics.median(durations), "mean_us": statistics.mean(durations),
                     "summed_us": sum(durations), "fraction_of_summed_kernel_time": sum(durations) / total,
                     "input_type_shape": group[0]["args"].get("input_type_shape"),
                     "output_type_shape": group[0]["args"].get("output_type_shape")})
    return {"profiled_calls": 16, "excluded_warmup_calls": 16, "node_count_per_call": len(grouped),
            "sum_kernel_mean_us_per_call": total / 16,
            "model_run_median_us": statistics.median(r["dur"] for r in selected),
            "nodes_by_summed_time": sorted(rows, key=lambda r: r["summed_us"], reverse=True),
            "operators_by_summed_time": sorted(({
                "operator": name, "summed_us": sum(values), "mean_us_per_call": sum(values) / 16,
                "fraction_of_summed_kernel_time": sum(values) / total,
            } for name, values in operators.items()), key=lambda r: r["summed_us"], reverse=True)}


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    training_path = PHASE / "fusion-refinement-001/plan.json"
    training = read(training_path)
    # Include the pending endpoint and temporary profiling allowance in the forecast.
    counted = require_space(training, 380_000_000 + 40_000_000)
    native_path = PHASE / "m4-quadrature-candidates-quiet-native-001/plan.json"
    native = read(native_path)
    verify_inputs(native)
    sdk = Path(native["sdk"])
    source = Path(__file__).with_name("profile_latency58_native.cpp").resolve()
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Native compiler unavailable")
    models = {"int8_c204": PHASE / "m4-int8-precise-core-full14-001/model.onnx",
              "int8_quadrature": PHASE / "m4-quadrature-magint8-saved-001/model.onnx"}
    bindings = {**native["source_bindings"], **{str(p): sha(p) for p in
                (source, Path(__file__).resolve(), native_path, training_path)}}
    require(all(bindings[str(p)] == sha(p) for p in models.values()), "Saved graph differs from quiet timing")
    out = PHASE / "quadrature-native-operator-profile-001"
    require(not out.exists(), "Preserve prior operator profiles")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "warmup_calls": 16, "diagnostic_calls": 16,
          "counted_bytes_before": counted, "pending_training_reservation_bytes": 380_000_000,
          "profiling_reservation_bytes": 40_000_000,
          "expected_runtime_version": "1.26.0", "inference_threads": 1,
          "gpu_training_may_run_concurrently": True,
          "scope": "Instrumented native CPU operator attribution; not comparable benchmark latency or quality",
          "raw_profile_retention": "Exact JSON bytes compressed losslessly with SHA256 readback; temporary raw files removed"})
    reports = {}
    with tempfile.TemporaryDirectory(prefix="latency58-operator-profile-") as temporary:
        temporary = Path(temporary)
        binary = temporary / "profile"
        argv = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", str(source),
                "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"),
                "-Wl,-rpath," + str(sdk / "lib"), "-lonnxruntime", "-o", str(binary)]
        write(out / "compiler-command.json", {"argv": argv})
        subprocess.run(argv, check=True, timeout=90)
        for label, model in models.items():
            completed = subprocess.run([str(binary), str(model), str(temporary / label)],
                                       check=True, capture_output=True, text=True, timeout=120)
            result = json.loads(completed.stdout)
            require(result["status"] == "pass" and result["onnxruntime_version"] == "1.26.0"
                    and result["instrumented_diagnostic_only"], "Native profiling failed")
            profile_path = Path(result["profile_path"])
            require(profile_path.parent == temporary and profile_path.stat().st_size < 35_000_000,
                    "Unexpected raw profile location or size")
            data = profile_path.read_bytes()
            compressed = gzip.compress(data, mtime=0)
            require(gzip.decompress(compressed) == data, "Compressed profile readback differs")
            artifact = out / (label + "-profile.json.gz")
            with artifact.open("xb") as stream:
                stream.write(compressed)
            require(hashlib.sha256(gzip.decompress(artifact.read_bytes())).hexdigest()
                    == hashlib.sha256(data).hexdigest(), "Saved profile readback differs")
            report = {"native_run": result, "runtime_stderr": completed.stderr,
                      "raw_bytes": len(data), "raw_sha256": hashlib.sha256(data).hexdigest(),
                      "compressed_profile": {"path": str(artifact), "sha256": sha(artifact), "bytes": len(compressed)},
                      **summarize(json.loads(data))}
            write(out / (label + "-summary.json"), report)
            reports[label] = report
            profile_path.unlink()
            require_space(training, 380_000_000 + 40_000_000)
            print(json.dumps({"variant": label, "node_count": report["node_count_per_call"],
                              "top_operators": report["operators_by_summed_time"][:5]}), flush=True)
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "pass", "plan_sha256": sha(out / "plan.json"),
          "source_bindings_unchanged": True, "variants": reports,
          "counted_bytes_after": require_space(training, 380_000_000),
          "native_host_qualified": False, "quality_measured": False,
          "limitation": "Short, instrumented profiles with possible concurrent GPU training identify candidate hotspots only. Use the completed quiet comparison for benchmark results; confirm optimizations in a new quiet benchmark."})


if __name__ == "__main__":
    main()
