"""Quiet native CPU1 comparison of the actual C204 release and saved attention graph."""
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.benchmark_latency58_quadrature_all_s8_native import require_quiet
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and platform.system() == "Linux" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use Linux CPU1 diagnostic scope")
    budget = read(PHASE / "temporal-attention-001/plan.json")
    require_space(budget, 5_000_000)
    require_quiet()
    saved_root = PHASE / "best-model-onnx-saved-001"
    saved = read(saved_root / "result.json")
    require(saved["status"] == "pass" and saved["source_bindings_unchanged"]
            and saved["saved_bytes_identical_to_full14_scored_graph"], "Require the saved, quality-tested graph")
    prerequisite_paths = []
    for name in ("attention-int8-screen-stage-001", "attention-int8-long-stage-001",
                 "attention-int8-full14-memory-stage-001", "best-model-onnx-saved-stage-001"):
        path = PHASE / name / "execution.json"
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "A graph preparation stage is incomplete")
        prerequisite_paths.append(path)
    baseline = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    candidate = Path(saved["onnx"]["path"])
    require(sha(baseline) == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3"
            and baseline.stat().st_size == 111344465 and sha(candidate) == saved["onnx"]["sha256"],
            "Require the actual release FP32 graph and the exact checked attention graph")
    sdk = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/libs/onnxruntime").resolve()
    cpp = ROOT / "research/direct/benchmark_latency58_attention_native.cpp"
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Native compiler unavailable")
    libraries = sorted({p.resolve() for p in (sdk / "lib").glob("libonnxruntime*.so*") if p.is_file()})
    headers = sorted((sdk / "include").rglob("*.h"))
    require(headers and libraries, "Native SDK incomplete")
    paths = [Path(__file__).resolve(), cpp, Path(compiler), baseline, candidate,
             saved_root / "result.json", saved_root / "plan.json", *prerequisite_paths,
             ROOT / "research/direct/benchmark_latency58_quadrature_all_s8_native.py",
             ROOT / "research/direct/run_latency58_quadrature_quiet_native.py", *libraries, *headers]
    bindings = {**read(saved_root / "plan.json")["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-plugin-quiet-native-001"
    require(not out.exists(), "Preserve native timings")
    out.mkdir()
    models = {"c204_release_fp32": baseline, "attention_signed_integer": candidate}
    orders = [("c204_release_fp32", "attention_signed_integer"),
              ("attention_signed_integer", "c204_release_fp32")] * 3
    write(out / "plan.json", {"source_bindings": bindings, "system": platform.platform(),
          "machine": platform.machine(), "sdk": str(sdk),
          "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
          "models": {key: {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for key, p in models.items()},
          "warmup_hops": 256, "measured_hops_per_cycle": 2048, "cycles": 6, "orders": orders,
          "scope": "Preallocated native ORT CPU1 inference; balanced fresh-process ordering; no concurrent training or scoring",
          "expected_persistent_states": {"c204_release_fp32": 4, "attention_signed_integer": 6},
          "same_deterministic_audio_every_process": True, "plugin_queue_measured": False,
          "native_host_qualified": False, "mac_execution_performed": False,
          "our_training_and_scoring_active_at_start": False, "other_user_programs_idle_claimed": False})
    shutil.copyfile(cpp, out / cpp.name)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    began, rows = time.monotonic(), []
    with tempfile.TemporaryDirectory(prefix="latency58-attention-native-") as temporary:
        binary = Path(temporary) / "benchmark"
        command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", str(cpp),
                   "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"),
                   "-Wl,-rpath," + str(sdk / "lib"), "-lonnxruntime", "-o", str(binary)]
        subprocess.run(command, check=True, env=environment)
        for cycle, order in enumerate(orders, 1):
            for label in order:
                require_quiet()
                completed = subprocess.run([str(binary), str(models[label]), "256", "2048"],
                                           text=True, capture_output=True, env=environment)
                if completed.returncode:
                    write(out / f"failed-cycle-{cycle}-{label}.json", {"actual_exit_code": completed.returncode,
                          "stdout": completed.stdout, "stderr": completed.stderr})
                require(completed.returncode == 0, "Native inference failed; retained output identifies the failure")
                row = {"cycle": cycle, "variant": label, **json.loads(completed.stdout),
                       "runtime_stderr": completed.stderr}
                require(row["status"] == "pass" and row["onnxruntime_version"] == "1.26.0"
                        and row["intra_op_threads"] == row["inter_op_threads"] == 1
                        and row["persistent_states"] == (4 if label == "c204_release_fp32" else 6),
                        "Runtime, worker configuration or state count differs")
                write(out / f"cycle-{cycle}-{label}.json", row)
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("cycle", "variant", "run")}), flush=True)
    require_quiet()
    verify_inputs({"source_bindings": bindings})
    medians = {label: statistics.median(r["run"]["p50_ms"] for r in rows if r["variant"] == label) for label in models}
    ratio = medians["attention_signed_integer"] / medians["c204_release_fp32"]
    faster = sum(next(r["run"]["p50_ms"] for r in rows if r["cycle"] == cycle and r["variant"] == "attention_signed_integer")
                 < next(r["run"]["p50_ms"] for r in rows if r["cycle"] == cycle and r["variant"] == "c204_release_fp32")
                 for cycle in range(1, 7))
    result = {"status": "pass", "cycles": rows, "plan_sha256": sha(out / "plan.json"),
              "median_of_cycle_p50_ms": medians, "attention_over_c204_median_ratio": ratio, "faster_cycles": faster,
              "source_bindings_unchanged": True, "our_training_and_scoring_active_at_end": False,
              "elapsed_seconds": time.monotonic() - began, "native_host_qualified": False,
              "mac_execution_performed": False, "plugin_queue_measured": False,
              "counted_bytes_after": require_space(budget, 0)}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "median_of_cycle_p50_ms",
                                                 "attention_over_c204_median_ratio", "faster_cycles")}), flush=True)


if __name__ == "__main__":
    main()
