"""Bounded CPU profile of the exact PR #17 graph during monitored training."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

from research.direct.latency58_m4_followup_budget import ROOT, POLICY, snapshot

OUT = ROOT / "research/m4_followup_20260916/pr17-operator-profile-001"
PHASE = ROOT / "research/direct/runs/latency58"
GRAPH = PHASE / "m4-inference-diagnostics-plugin-001/model/model.onnx"
GRAPH_SHA = "c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only execution required")
    require(all(os.environ.get(k) == "1" for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "One-thread environment required")
    import numpy as np
    import onnxruntime as ort

    began = time.monotonic()
    before = snapshot()
    require(not OUT.exists(), "Preserve earlier profile evidence")
    reference_plan = PHASE / "branch-output-int8-screen-003/plan.json"
    contract = json.loads(reference_plan.read_text())["interface"]
    data = GRAPH.read_bytes()
    require(hashlib.sha256(data).hexdigest() == GRAPH_SHA and len(data) == 38_298_020,
            "Exact PR #17 graph identity changed")
    bindings = {str(p): sha(p) for p in
                (Path(__file__), POLICY, reference_plan, GRAPH,
                 ROOT / "research/direct/latency58_m4_followup_budget.py")}
    runtime_module = Path(ort.__file__).resolve()
    for path in (runtime_module, *sorted((runtime_module.parent / "capi").glob("*.so*"))):
        bindings[str(path)] = sha(path)
    OUT.mkdir(parents=True)
    write(OUT / "plan.json", {"graph": str(GRAPH), "graph_sha256": GRAPH_SHA,
        "source_bindings": bindings, "interface": contract, "budget_before": before,
        "plain_calls": 64, "plain_warmup": 16, "profile_calls": 32, "profile_warmup": 8,
        "raw_trace_ceiling_bytes": 20_000_000, "output_ceiling_bytes": 25_000_000,
        "cpu_only": True, "ort_threads": 1, "concurrent_gpu_training": True,
        "input": "Seed 20260916, stereo normal noise std=0.03; fresh state for each session",
        "purpose": "Identify current kernel costs; no M4, callback or quality qualification"})
    audio = np.random.default_rng(20260916).normal(0, .03, (64, 1, 2, 128)).astype(np.float32)

    def session(profile):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            options.add_session_config_entry(key, "0")
        if profile:
            options.enable_profiling = True
            options.profile_file_prefix = str(OUT / "raw-operators")
        value = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        require(value.get_providers() == ["CPUExecutionProvider"], "Unexpected provider")
        for direction, values in (("input", value.get_inputs()), ("output", value.get_outputs())):
            require([v.name for v in values] == contract[direction + "_names"]
                    and [list(v.shape) for v in values] == contract[direction + "_shapes"]
                    and all(v.type == "tensor(float)" for v in values), "Public interface changed")
        return value

    def execute(value, calls):
        states = [np.zeros(s, np.float32) for s in contract["state_shapes"]]
        previous = np.zeros((2, 128), np.float32)
        durations, closure = [], 0.0
        for index in range(calls):
            expected_history = np.concatenate((states[0][..., 128:], audio[index]), axis=-1)
            inputs = dict(zip(contract["input_names"], [audio[index], *states], strict=True))
            start = time.perf_counter_ns()
            values = value.run(contract["output_names"], inputs)
            durations.append((time.perf_counter_ns() - start) / 1e6)
            require(all(v.shape == tuple(s) and v.dtype == np.float32 and np.isfinite(v).all()
                        for v, s in zip(values, contract["output_shapes"], strict=True)),
                    "Invalid public output")
            require(np.array_equal(values[1], expected_history), "History shift changed")
            closure = max(closure, float(np.abs(values[0][0].sum(axis=0) - previous).max()))
            states, previous = values[1:], audio[index][0]
        require(closure <= 2e-6, "Reconstruction closure changed")
        return durations, closure

    plain = session(False)
    durations, plain_closure = execute(plain, 64)
    del plain
    traced = session(True)
    _, traced_closure = execute(traced, 32)
    trace_path = Path(traced.end_profiling())
    del traced
    require(trace_path.stat().st_size <= 20_000_000, "Trace exceeded its allocation")
    events = json.loads(trace_path.read_text())
    runs = sorted((v for v in events if v.get("name") == "model_run"), key=lambda v: v["ts"])
    require(len(runs) == 32, "Incomplete profile calls")
    start, stop = runs[8]["ts"], runs[-1]["ts"] + runs[-1]["dur"]
    groups, attributes = defaultdict(list), {}
    for event in events:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time") and start <= event["ts"] < stop:
            name = event["name"]
            groups[name].append(event["dur"])
            attributes[name] = {k: event["args"].get(k) for k in
                ("op_name", "provider", "input_type_shape", "output_type_shape", "parameter_size")}
    total = sum(sum(v) for v in groups.values())
    require(total > 0 and all(len(v) == 24 for v in groups.values()), "Incomplete kernel coverage")
    operators = sorted([{"name": k, **attributes[k], "calls": len(v),
        "mean_us": statistics.mean(v), "median_us": statistics.median(v),
        "percent_kernel_time": 100 * sum(v) / total} for k, v in groups.items()],
        key=lambda v: v["mean_us"], reverse=True)
    by_type = defaultdict(float)
    for row in operators:
        by_type[row["op_name"]] += row["percent_kernel_time"]
    require(all(sha(p) == h for p, h in bindings.items()), "Profile inputs changed")
    timings = durations[16:]
    result = {"status": "pass", "graph_sha256": GRAPH_SHA,
        "source_bindings": bindings, "source_bindings_unchanged": True,
        "ort_version": ort.__version__, "system": platform.platform(), "machine": platform.machine(),
        "gpu_used": False, "one_ort_thread": True, "concurrent_gpu_training": True,
        "operators": operators, "percent_kernel_time_by_operator_type": dict(by_type),
        "raw_trace": {"path": str(trace_path), "sha256": sha(trace_path), "bytes": trace_path.stat().st_size},
        "plain_python_run_timing_ms": {"calls": len(timings), "median": statistics.median(timings),
            "maximum": max(timings), "mean": statistics.mean(timings)},
        "maximum_closure": {"plain": plain_closure, "traced": traced_closure},
        "budget_after": snapshot(), "elapsed_seconds": time.monotonic() - began,
        "target_mac_qualified": False, "quality_measured": False,
        "limitations": "Synthetic-noise operator diagnosis on Linux during GPU training. Tracing and Python output allocation add cost. No callback, M4 timing, numerical-reference or quality acceptance is established."}
    write(OUT / "result.json", result)
    require(sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) <= 25_000_000,
            "Profile output exceeded its allowance")
    print(json.dumps({"status": "pass", "top_operators": operators[:18],
        "by_type": dict(by_type), "timing": result["plain_python_run_timing_ms"],
        "combined_budget_peak_bytes": result["budget_after"]["combined_peak_bytes"],
        "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
