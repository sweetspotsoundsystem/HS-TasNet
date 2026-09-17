"""Profile a hop128 ONNX graph on one CPU thread without changing its bytes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert not args.output.exists()
    import numpy as np
    import onnx
    import onnxruntime as ort
    model_path = args.model.resolve()
    original_sha = digest(model_path)
    graph = onnx.load(model_path, load_external_data=False)
    assert not any(t.external_data for t in graph.graph.initializer)
    initializers = sorted(({"name": t.name, "shape": list(t.dims), "elements": int(np.prod(t.dims)),
                            "bytes": len(t.raw_data)} for t in graph.graph.initializer), key=lambda x: x["bytes"], reverse=True)
    input_names = ["audio_chunk", "audio_history", "fusion_hidden", "spectral_numerator_tail", "waveform_tail"]
    input_shapes = [(1, 2, 128), (1, 2, 896), (2, 1, 1000), (1, 4, 2, 128), (1, 4, 2, 128)]
    output_names = ["separated_chunk", "next_audio_history", "next_fusion_hidden", "next_spectral_numerator_tail", "next_waveform_tail"]
    generator = np.random.default_rng(20261009)
    audio = generator.normal(0, .03, (320, 1, 2, 128)).astype(np.float32)

    def session(profile_prefix=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            options.add_session_config_entry(key, "0")
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        value = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        assert value.get_providers() == ["CPUExecutionProvider"]
        assert [x.name for x in value.get_inputs()] == input_names
        assert [tuple(x.shape) for x in value.get_inputs()] == input_shapes
        assert [x.name for x in value.get_outputs()] == output_names
        return value

    def execute(value, count):
        states = [np.zeros(shape, dtype=np.float32) for shape in input_shapes[1:]]
        durations = []
        for index in range(count):
            inputs = dict(zip(input_names, [audio[index], *states], strict=True))
            start = time.perf_counter_ns()
            output = value.run(output_names, inputs)
            durations.append((time.perf_counter_ns() - start) / 1e6)
            assert all(np.isfinite(x).all() for x in output)
            states = output[1:]
        return durations

    began = time.monotonic()
    plain = session()
    durations = execute(plain, 320)[64:]
    del plain
    with tempfile.TemporaryDirectory(prefix="latency58-operator-profile-") as temporary:
        profiled = session(Path(temporary) / "operators")
        execute(profiled, 48)
        trace_path = Path(profiled.end_profiling())
        del profiled
        trace_bytes = trace_path.stat().st_size
        events = json.loads(trace_path.read_text())
    runs = sorted((e for e in events if e.get("name") == "model_run"), key=lambda e: e["ts"])
    assert len(runs) == 48
    start, stop = runs[16]["ts"], runs[-1]["ts"] + runs[-1]["dur"]
    groups = defaultdict(list)
    attributes = {}
    for event in events:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time") and start <= event["ts"] < stop:
            key = event["name"]
            groups[key].append(event["dur"])
            attributes[key] = {k: event["args"].get(k) for k in ("op_name", "provider", "input_type_shape", "output_type_shape", "parameter_size")}
    total = sum(sum(v) for v in groups.values())
    assert total > 0 and all(len(v) == 32 for v in groups.values())
    operators = sorted(({"name": key, **attributes[key], "calls": len(values),
                         "mean_microseconds": statistics.mean(values), "percent_kernel_time": 100 * sum(values) / total}
                        for key, values in groups.items()), key=lambda x: x["mean_microseconds"], reverse=True)
    budget = 128 / 44.1
    assert digest(model_path) == original_sha
    result = {"status": "pass", "model": {"path": str(model_path), "sha256": original_sha, "bytes": model_path.stat().st_size},
              "system": platform.platform(), "machine": platform.machine(), "onnxruntime": ort.__version__,
              "single_cpu_thread": True, "gpu_used": False, "model_unchanged": True,
              "host_callback_timing_measured": False, "target_m4_qualified": False, "quality_measured": False,
              "uninstrumented_timing": {"warmup_hops": 64, "measured_hops": len(durations),
                  "mean_ms": statistics.mean(durations), "p50_ms": float(np.percentile(durations, 50)),
                  "p95_ms": float(np.percentile(durations, 95)), "p99_ms": float(np.percentile(durations, 99)),
                  "maximum_ms": max(durations), "hop_budget_ms": budget, "misses": sum(t > budget for t in durations)},
              "profile_warmup_hops_discarded": 16, "profile_hops": 32, "operators": operators,
              "initializers": initializers, "temporary_trace_bytes": trace_bytes,
              "raw_trace_retained": False, "elapsed_seconds": time.monotonic() - began,
              "source_sha256": digest(__file__),
              "limitations": "Local CPU operator diagnosis with training active on the GPU. Profiled durations include tracing overhead; uninstrumented times exclude the plugin queue and cannot qualify a Mac."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "system", "uninstrumented_timing", "temporary_trace_bytes", "elapsed_seconds")}))
    print(json.dumps({"top_operators": operators[:12]}))


if __name__ == "__main__":
    main()
