"""Bounded local operator diagnosis of the exact v0.4.0 eight-state graph.

Concurrent training and validation make these observations unsuitable for
deadline qualification or target-Mac speed claims.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from research.direct.run_latency58_quality import PHASE, require, read, sha, write


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    from research.direct.run_latency58_deployed_vocal_views import require_cpu, budget_snapshot
    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    require_cpu(baseline)
    budget = {**baseline, "diagnostic_artifact_allowance_bytes": 250_000_000}
    before = budget_snapshot(budget)
    out = PHASE / "deployed-operator-profile-001"
    require(not out.exists(), "Preserve previous operator profiles")
    checkpoint, contract = baseline["checkpoint"], baseline["interface"]
    data = Path(checkpoint["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == checkpoint["sha256"] and len(data) == checkpoint["bytes"],
            "Exact deployed graph changed")
    graph = onnx.load_model_from_string(data)
    require(not any(v.external_data for v in graph.graph.initializer), "Graph is not self-contained")
    bindings = {str(Path(__file__).resolve()): sha(__file__), str(baseline_path): sha(baseline_path),
                checkpoint["path"]: checkpoint["sha256"]}
    module = Path(ort.__file__).resolve()
    for path in (module, *sorted((module.parent / "capi").glob("*.so*"))):
        bindings[str(path)] = sha(path)
    out.mkdir()
    write(out / "plan.json", {"checkpoint": checkpoint, "interface": contract,
        "source_bindings": bindings, "budget_before": before, "profile_calls": 32,
        "profile_warmup_calls": 8, "plain_calls": 64, "plain_warmup_calls": 16,
        "concurrent_training_and_validation": True, "target_mac_or_host_qualification": False})
    audio = np.random.default_rng(20260913).normal(0, .03, (64, 1, 2, 128)).astype(np.float32)

    def make_session(profile=False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            options.add_session_config_entry(key, "0")
        if profile:
            options.enable_profiling = True
            options.profile_file_prefix = str(out / "raw-operators")
        session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        require(session.get_providers() == ["CPUExecutionProvider"], "Unexpected execution provider")
        for direction, values in (("input", session.get_inputs()), ("output", session.get_outputs())):
            require([v.name for v in values] == contract[direction + "_names"]
                    and [v.shape for v in values] == contract[direction + "_shapes"]
                    and all(v.type == "tensor(float)" for v in values), "Public graph interface changed")
        return session

    def execute(session, calls):
        states = [np.zeros(s, np.float32) for s in contract["state_shapes"]]
        previous = np.zeros((2, 128), np.float32)
        durations, maximum_closure = [], 0.
        for index in range(calls):
            expected_history = np.concatenate((states[0][..., 128:], audio[index]), axis=-1)
            inputs = dict(zip(contract["input_names"], [audio[index], *states], strict=True))
            began = time.perf_counter_ns()
            values = session.run(contract["output_names"], inputs)
            durations.append((time.perf_counter_ns() - began) / 1e6)
            require(all(v.shape == tuple(s) and v.dtype == np.float32 and np.isfinite(v).all()
                        for v, s in zip(values, contract["output_shapes"], strict=True)), "Invalid public output")
            require(np.array_equal(values[1], expected_history), "Physical history changed")
            maximum_closure = max(maximum_closure, float(np.abs(values[0][0].sum(axis=0) - previous).max()))
            states, previous = values[1:], audio[index][0]
        require(maximum_closure <= 2e-6, "Raw graph closure changed")
        return durations, maximum_closure

    began = time.monotonic()
    session = make_session()
    durations, plain_closure = execute(session, 64)
    del session
    session = make_session(profile=True)
    _, profile_closure = execute(session, 32)
    trace_path = Path(session.end_profiling())
    del session
    trace_bytes = trace_path.stat().st_size
    require(trace_bytes < 240_000_000, "Raw trace exceeded its reserved allowance")
    events = json.loads(trace_path.read_text())
    runs = sorted((e for e in events if e.get("name") == "model_run"), key=lambda e: e["ts"])
    require(len(runs) == 32, "Unexpected number of profiled graph calls")
    start, stop = runs[8]["ts"], runs[-1]["ts"] + runs[-1]["dur"]
    groups, attributes = defaultdict(list), {}
    for event in events:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time") and start <= event["ts"] < stop:
            name = event["name"]
            groups[name].append(event["dur"])
            attributes[name] = {k: event["args"].get(k) for k in
                ("op_name", "provider", "input_type_shape", "output_type_shape", "parameter_size")}
    total = sum(sum(v) for v in groups.values())
    require(total > 0 and all(len(v) == 24 for v in groups.values()), "Incomplete steady profile coverage")
    operators = sorted([{"name": k, **attributes[k], "calls": len(v),
        "mean_us": statistics.mean(v), "median_us": statistics.median(v),
        "percent_kernel_time": 100 * sum(v) / total} for k, v in groups.items()],
        key=lambda v: v["mean_us"], reverse=True)
    by_type = defaultdict(float)
    for row in operators:
        by_type[row["op_name"]] += row["percent_kernel_time"]
    require(all(sha(p) == h for p, h in bindings.items()), "Profile inputs changed")
    timings = durations[16:]
    result = {"status": "pass", "checkpoint": checkpoint, "runtime": baseline["runtime"],
        "system": platform.platform(), "machine": platform.machine(), "gpu_used": False,
        "one_ort_thread": True, "concurrent_training_and_validation": True,
        "source_bindings": bindings, "source_bindings_unchanged": True,
        "operators": operators, "percent_kernel_time_by_operator_type": dict(by_type),
        "raw_trace": {"path": str(trace_path), "sha256": sha(trace_path), "bytes": trace_bytes},
        "plain_python_run_timing_ms": {"calls": len(timings), "mean": statistics.mean(timings),
            "median": statistics.median(timings), "maximum": max(timings)},
        "maximum_raw_closure": {"plain": plain_closure, "profiled": profile_closure},
        "profile_calls_scored": 24, "host_qualified": False, "target_mac_qualified": False,
        "quality_measured": False, "budget_before": before, "budget_after": budget_snapshot(budget),
        "elapsed_seconds": time.monotonic() - began,
        "limitations": "Local x86 CPU diagnosis under concurrent load. Tracing and Python output allocation add overhead. Operator shares guide hypotheses; absolute times cannot qualify the plugin, establish M4 speed, or substitute for the native preallocated callback/worker path."}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "top_operators": operators[:15],
        "percent_kernel_time_by_operator_type": dict(by_type), "raw_trace_bytes": trace_bytes,
        "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
