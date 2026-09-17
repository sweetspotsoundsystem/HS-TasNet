"""Decompose the released graph's instrumental vocal output on Rockshow.

Expose two existing tensors in a private in-memory graph. Every public output
and carried state must remain bit exact against the untouched shipping graph.
This measures an existing failure; it does not select a residual coefficient.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_attention_int8_verify import session_for
    from research.direct.latency58_evaluate import plan_latency58_stream, latency58_stream_metadata
    from research.direct.latency58_vocal_views import combine_sources
    from research.direct.run_latency58_deployed_vocal_views import require_cpu, budget_snapshot
    from research.metrics import SOURCE_ORDER, MetricConfig, frame_ranges, rms_dbfs, db_ratio

    source = PHASE / "deployed-vocal-views-001"
    baseline = read(source / "plan.json")
    require_cpu(baseline)
    out = PHASE / "deployed-vocal-residual-001"
    require(not out.exists(), "Preserve earlier decomposition evidence")
    baseline_result, baseline_execution = (read(source / name) for name in ("result.json", "execution.json"))
    require(baseline_result["status"] == "pass" and baseline_result["source_bindings_unchanged"]
            and baseline_result["plan_sha256"] == sha(source / "plan.json")
            and baseline_execution["actual_exit_code"] == 0 and not baseline_execution["timed_out"]
            and baseline_execution["source_bindings_unchanged"], "Released baseline did not complete")
    previous = read(source / "track-00.json")
    require(previous == baseline_result["tracks"][0], "Separate track report differs from the completed baseline")
    checkpoint, contract = baseline["checkpoint"], baseline["interface"]
    data = Path(checkpoint["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == checkpoint["sha256"]
            == "d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa"
            and len(data) == checkpoint["bytes"], "Exact released graph changed")
    original = onnx.load_model_from_string(data)
    instrumented = copy.deepcopy(original)
    tensor_names = ["/Mul_21_output_0", "/Mul_22_output_0"]
    tensor_shapes = [[1, 4, 2, 128], [1, 1, 2, 128]]
    nodes = {node.name: node for node in original.graph.node}
    expected_nodes = {
        "/Mul_21": ("Mul", ["/Add_17_output_0", "onnx::Mul_810"], [tensor_names[0]]),
        "/Sub_1": ("Sub", ["/Slice_17_output_0", "/Add_20_output_0"], ["/Sub_1_output_0"]),
        "/Mul_22": ("Mul", ["model.fixed_residual_share", "/Unsqueeze_11_output_0"], [tensor_names[1]]),
        "/Add_21": ("Add", ["/Slice_18_output_0", tensor_names[1]], ["/Add_21_output_0"]),
    }
    require(all((nodes[name].op_type, list(nodes[name].input), list(nodes[name].output)) == value
                for name, value in expected_nodes.items()), "Residual decomposition topology changed")
    share = next(t for t in original.graph.initializer if t.name == "model.fixed_residual_share")
    require(float(onnx.numpy_helper.to_array(share)) == 1 / 16, "Fixed correction share changed")
    for name, shape in zip(tensor_names, tensor_shapes, strict=True):
        instrumented.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    probe_data = instrumented.SerializeToString()
    # Removing only the added output declarations must recover the complete graph.
    stripped = copy.deepcopy(instrumented)
    del stripped.graph.output[-2:]
    require(stripped.SerializeToString() == original.SerializeToString(), "Instrumentation modified the model")
    onnx.checker.check_model(instrumented)
    probe_contract = {**contract, "output_names": [*contract["output_names"], *tensor_names],
                      "output_shapes": [*contract["output_shapes"], *tensor_shapes]}
    manifest, config = read(baseline["manifest"]["path"]), read(baseline["config"]["path"])
    tracks, config = shared.select_panel(manifest, config, panel="full", track_indices=[0],
        excerpt_starts=None, duration=15., alignment_samples=128)
    track = tracks[0]
    intervals = legacy._reference_intervals(track, config)
    require(track["name"] == previous["name"] == "ANiMAL - Rockshow"
            and intervals == previous["intervals"] == baseline["track_intervals"]["0"]["intervals"],
            "Predeclared physical excerpts changed")
    paths = [legacy._safe_dataset_path(Path(manifest["root"]), track["stems"][stem]) for stem in SOURCE_ORDER]
    bindings = {str(p): baseline["source_bindings"][str(p)] for p in paths}
    code = ["research/evaluate.py", "research/metrics.py", "research/direct/evaluate.py",
            "research/direct/latency58_evaluate.py", "research/direct/latency58_vocal_views.py",
            "research/direct/latency58_attention_int8_verify.py", "research/direct/latency58_best_onnx.py",
            "research/direct/run_latency58_deployed_vocal_views.py", "research/direct/run_latency58_quality.py",
            "research/direct/train_latency58.py"]
    for name in code:
        path = ROOT / name
        require(sha(path) == baseline["source_bindings"][str(path)], "Qualified diagnostic dependency changed")
        bindings[str(path)] = sha(path)
    inputs = [Path(__file__).resolve(), Path(checkpoint["path"]),
              Path(baseline["manifest"]["path"]), Path(baseline["config"]["path"])]
    inputs.extend(source / name for name in ("plan.json", "result.json", "execution.json", "track-00.json"))
    module = Path(ort.__file__).resolve()
    inputs.extend([module, *sorted((module.parent / "capi").glob("*.so*"))])
    bindings.update({str(p): sha(p) for p in inputs})
    verify_inputs({"source_bindings": bindings})
    budget = read(PHASE / "branch-gru-int8-post-ci-storage-001.json")
    before = budget_snapshot(budget)
    stream = plan_latency58_stream(intervals, int(track["frames"]), unroll_hops=1, io_block_hops=64)
    out.mkdir()
    plan = {"schema": "latency58-deployed-vocal-residual-diagnostic-v1", "checkpoint": checkpoint,
        "source_bindings": bindings, "runtime": baseline["runtime"], "budget_before": before,
        "track_index": 0, "track_name": track["name"], "intervals": intervals, "view": "instrumental",
        "selection": "Previously measured worst instrumental track, including worst absolute 78-79 s and relative 80-81 s windows; both existing excerpts retained without further selection.",
        "exposed_existing_tensors": dict(zip(tensor_names, tensor_shapes, strict=True)),
        "in_memory_graph_sha256": hashlib.sha256(probe_data).hexdigest(),
        "only_two_output_declarations_added": True, "fixed_residual_share": 1 / 16,
        "required_public_parity": "All nine outputs including every carried state bit exact on every real hop",
        "stream": latency58_stream_metadata(stream), "gpu_used": False, "quality_selection": False,
        "new_audio_exports": False, "graph_saved_or_replaced": False}
    write(out / "plan.json", plan)
    plan_sha = sha(out / "plan.json")
    began = time.monotonic()
    sessions = [session_for(data, contract), session_for(probe_data, probe_contract)]
    states = [[np.zeros(shape, np.float32) for shape in contract["state_shapes"]] for _ in sessions]
    previous_mix = np.zeros((2, 128), np.float32)
    captures = {name: legacy._Capture(stream.capture_intervals, shape) for name, shape in
                (("deployed", (4, 2)), ("native", (4, 2)), ("correction", (2,)), ("mixture", (2,)))}
    references = legacy._Capture(stream.reference_intervals, (4, 2))
    digest = hashlib.sha256()
    readers = legacy._open_blocked_readers(paths, [stream.expected_frames] * 4, hop=128, block_hops=64)
    maximum_closure = 0.
    calls = cursor = 0
    with (out / "progress.jsonl").open("x", buffering=1) as log:
        try:
            for start, stop in stream.call_slices:
                require(start == cursor and stop - start == 128, "Skipped physical input")
                blocks = [reader.read_hop() for reader in readers]
                require(all(b.shape == (128, 2) and b.dtype == np.float32 and np.isfinite(b).all() for b in blocks),
                        "Invalid source audio or early EOF")
                sources = np.stack([block.T for block in blocks])
                references.add(start, sources)
                mixture = combine_sources(sources, "instrumental")
                digest.update(np.ascontiguousarray(mixture.T).tobytes())
                audio = np.ascontiguousarray(mixture[None])
                expected_history = np.concatenate((states[0][0][..., 128:], audio), axis=-1)
                values = [session.run(c["output_names"], dict(zip(contract["input_names"], [audio, *state], strict=True)))
                          for session, state, c in zip(sessions, states, (contract, probe_contract), strict=True)]
                require(all(np.array_equal(a, b) for a, b in zip(values[0], values[1][:9], strict=True)),
                        "Adding diagnostic outputs changed a public output or carried state")
                require(all(v.shape == tuple(s) and v.dtype == np.float32 and np.isfinite(v).all()
                            for v, s in zip(values[1], probe_contract["output_shapes"], strict=True)), "Invalid graph output")
                require(np.array_equal(values[0][1], expected_history), "Physical history changed")
                native, correction = values[1][-2][0], values[1][-1][0, 0]
                independent_correction = np.float32(1 / 16) * (previous_mix - (((native[0] + native[1]) + native[2]) + native[3]))
                require(np.array_equal(correction, independent_correction)
                        and np.array_equal(native[:3] + correction[None], values[0][0][0, :3]),
                        "Exposed components do not reconstruct the original DBV output")
                deployed = shared.shipping_residual(values[0][0][0], previous_mix)
                maximum_closure = max(maximum_closure, float(np.abs(deployed.sum(0, dtype=np.float32) - previous_mix).max()))
                require(maximum_closure <= 1e-6, "Shipping reconstruction changed")
                for name, value in (("deployed", deployed), ("native", native),
                                    ("correction", correction), ("mixture", previous_mix)):
                    captures[name].add(start, value)
                states, previous_mix = [v[1:9] for v in values], mixture
                cursor, calls = stop, calls + 1
                if calls % 1024 == 0 or calls == stream.literal_hop_count:
                    row = {"calls_per_graph": calls, "total_calls_per_graph": stream.literal_hop_count,
                           "elapsed_seconds": time.monotonic() - began, "worker_pid": os.getpid()}
                    log.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
        finally:
            for reader in readers:
                reader.close()
    require(calls == stream.literal_hop_count == previous["stream"]["actual_forward_call_count_per_view"]
            and cursor == stream.receive_end and digest.hexdigest() == previous["stream"]["input_stream_sha256"]["instrumental"],
            "Continuous physical inputs differ from the completed released baseline")
    refs, captured = references.finish(), {name: capture.finish() for name, capture in captures.items()}
    metric = MetricConfig.from_mapping(config["metrics"])
    old_windows = previous["views"]["instrumental"]["windows"]
    rows = []
    for excerpt, interval in enumerate(intervals):
        independent = np.stack([legacy._read_excerpt(p, interval["reference_start"], interval["reference_end"],
            expected_frames=int(track["frames"])) for p in paths]).astype(np.float32)
        require(np.array_equal(refs[excerpt], independent)
                and np.array_equal(captured["mixture"][excerpt], combine_sources(independent, "instrumental")),
                "Independent physical reference or delayed-mixture alignment differs")
        for start, stop in frame_ranges(independent.shape[-1], metric.window_samples, metric.hop_samples):
            native = captured["native"][excerpt][2, :, start:stop].astype(np.float64)
            correction = captured["correction"][excerpt][:, start:stop].astype(np.float64)
            vocal = captured["deployed"][excerpt][2, :, start:stop].astype(np.float64)
            mixture = captured["mixture"][excerpt][:, start:stop].astype(np.float64)
            old = old_windows[len(rows)]
            def ratio(a, b):
                return db_ratio(float(np.square(a).sum()), float(np.square(b).sum()), epsilon=metric.epsilon,
                                floor=metric.db_floor, ceiling=metric.db_ceiling)
            vocal_level = rms_dbfs(vocal, metric.epsilon)
            require(old["excerpt_index"] == excerpt and old["physical_start"] == interval["reference_start"] + start
                    and old["physical_end"] == interval["reference_start"] + stop
                    and abs(vocal_level - old["per_stem"]["vocals"]["output_rms_dbfs"]) < 1e-12
                    and abs(ratio(vocal, mixture) - old["per_stem"]["vocals"]["output_to_input_db"]) < 1e-12,
                    "Deployed window does not reproduce its previously measured baseline")
            pn, pc, pv = (float(np.square(v).mean()) for v in (native, correction, vocal))
            cross = float(2 * (native * correction).mean())
            rows.append({"excerpt_index": excerpt, "physical_start": old["physical_start"],
                "physical_end": old["physical_end"], "input_active": old["input_active"],
                "mixture_dbfs": rms_dbfs(mixture, metric.epsilon), "deployed_vocal_dbfs": vocal_level,
                "native_vocal_dbfs": rms_dbfs(native, metric.epsilon),
                "fixed_correction_dbfs": rms_dbfs(correction, metric.epsilon),
                "deployed_vocal_to_input_db": ratio(vocal, mixture), "native_vocal_to_input_db": ratio(native, mixture),
                "correction_to_deployed_vocal_db": ratio(correction, vocal),
                "vocal_level_change_from_adding_fixed_correction_db": vocal_level - rms_dbfs(native, metric.epsilon),
                "power_decomposition": {"native": pn, "correction": pc, "twice_cross": cross, "deployed": pv,
                    "fp32_addition_power_error": pv - (pn + pc + cross)},
                "native_vocal_projection_on_other_reference": float((native * independent[3, :, start:stop]).sum()
                    / (np.square(independent[3, :, start:stop].astype(np.float64)).sum() + metric.epsilon))})
    require(len(rows) == len(old_windows) == 30, "Incomplete predeclared window coverage")
    verify_inputs({"source_bindings": bindings})
    require(sha(out / "plan.json") == plan_sha and not torch.cuda.is_initialized(), "Plan changed or GPU initialized")
    result = {"status": "pass", "plan_sha256": plan_sha, "checkpoint": checkpoint, "track": track["name"],
        "source_bindings_unchanged": True, "only_two_output_declarations_added": True,
        "all_nine_public_outputs_bit_exact_every_hop": True, "component_equations_bit_exact_every_hop": True,
        "baseline_instrumental_input_sha256": digest.hexdigest(), "all_thirty_baseline_vocal_windows_reproduced": True,
        "calls_per_graph": calls, "source_file_hop_reads": calls * 4,
        "independent_physical_references_and_alignment_verified": True, "maximum_shipping_closure": maximum_closure,
        "windows": rows, "elapsed_seconds": time.monotonic() - began,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_used": False, "audio_exported": False, "training_updates": 0, "quality_selected": False,
        "graph_saved_or_replaced": False, "budget_after": budget_snapshot(budget),
        "limitations": ["Previously selected development failure on one track, not a 14-track improvement estimate.",
            "Native and correction energies include a cross term and are not additive percentages of output energy.",
            "Other-reference projection includes correlated sources; it is not causal source attribution.",
            "Removing a component algebraically does not establish real-vocal preservation, Other quality, or a useful coefficient.",
            "No deployment, M4 timing, listening acceptance, or new checkpoint quality measured."]}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "calls_per_graph": calls, "elapsed_seconds": result["elapsed_seconds"],
                      "predeclared_worst_windows": [row for row in rows if row["physical_start"] in (78 * 44100, 80 * 44100)]}), flush=True)


if __name__ == "__main__":
    main()
