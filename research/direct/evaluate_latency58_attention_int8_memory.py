"""Score the attention integer graph on the unchanged full14 panel, without saving weights."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

_SESSION = _PLAN = None


def initialize(plan, data):
    global _SESSION, _PLAN
    import onnxruntime as ort
    import torch
    from research.direct.latency58_attention_int8_verify import session_for
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(ort.__version__ == "1.26.0" and str(Path(ort.__file__).resolve()) == plan["runtime"]["python_module"]
            and hashlib.sha256(data).hexdigest() == plan["checkpoint"]["sha256"]
            and len(data) == plan["checkpoint"]["bytes"] and not torch.cuda.is_initialized(),
            "Worker graph, runtime or CPU scope differs")
    _PLAN = plan
    _SESSION = session_for(data, plan["interface"])


def score(index):
    import numpy as np
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import plan_latency58_stream, latency58_stream_metadata, SOURCE_ORDER
    INPUT_NAMES, OUTPUT_NAMES = _PLAN["interface"]["input_names"], _PLAN["interface"]["output_names"]
    from research.metrics import MetricConfig
    manifest, source_config = read(_PLAN["manifest"]["path"]), read(_PLAN["config"]["path"])
    tracks, config = shared.select_panel(manifest, source_config, panel="full", track_indices=[index],
        excerpt_starts=None, duration=15.0, alignment_samples=128)
    require(len(tracks) == 1, "Wrong validation track selection")
    track = tracks[0]
    rows = legacy._reference_intervals(track, config)
    require(len(rows) == 2, "Validation excerpt count changed")
    plan = plan_latency58_stream(rows, int(track["frames"]), unroll_hops=1, io_block_hops=64)
    captured = legacy._Capture(plan.capture_intervals, (4, 2))
    delayed_capture = legacy._Capture(plan.capture_intervals, (2,))
    root = Path(manifest["root"])
    reader = legacy._open_blocked_readers([legacy._safe_dataset_path(root, track["mixture"])],
        [plan.expected_frames], hop=128, block_hops=64)[0]
    states = [np.zeros(shape, np.float32) for shape in _PLAN["interface"]["state_shapes"]]
    previous_audio = np.zeros((2, 128), dtype=np.float32)
    calls, cursor, maximum_closure = 0, 0, 0.
    began = time.monotonic()
    try:
        for start, stop in plan.call_slices:
            require(start == cursor and stop - start == 128, "Literal graph stream changed")
            block = reader.read_hop()
            require(block.shape == (128, 2) and np.isfinite(block).all(), "Invalid real validation input")
            audio = np.ascontiguousarray(block.T[None])
            values = _SESSION.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [audio, *states], strict=True)))
            require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in values),
                    "Nonfinite or non-FP32 public output/state")
            require(values[0].shape == (1, 4, 2, 128)
                    and np.array_equal(values[1][0, :, -128:], audio[0]), "Graph output or audio history differs")
            states = values[1:]
            maximum_closure = max(maximum_closure, float(np.abs(values[0][0].sum(axis=0) - previous_audio).max()))
            captured.add(start, values[0][0])
            delayed_capture.add(start, previous_audio)
            previous_audio = audio[0]
            cursor = stop
            calls += 1
    finally:
        reader.close()
    require(cursor == plan.receive_end and calls == plan.literal_hop_count and maximum_closure < 2e-6,
            "Graph execution, mixture closure, or continuous coverage differs")
    estimates, delayed = captured.finish(), delayed_capture.finish()
    def read_excerpt(relative, row):
        return legacy._read_excerpt(legacy._safe_dataset_path(root, relative), row["reference_start"],
            row["reference_end"], expected_frames=int(track["frames"]))
    mixtures = [read_excerpt(track["mixture"], row) for row in rows]
    references = [np.stack([read_excerpt(track["stems"][stem], row) for stem in SOURCE_ORDER]) for row in rows]
    require(all(np.array_equal(left, right.astype(np.float32)) for left, right in zip(delayed, mixtures, strict=True)),
            "Captured mixture differs from physical scoring references")
    # Match the shipping writer's float32 residual correction after ORT.
    estimates = [shared.shipping_residual(value, mixture) for value, mixture in zip(estimates, mixtures, strict=True)]
    scored = legacy._score_track(track["name"], rows, mixtures, references, estimates, MetricConfig.from_mapping(config["metrics"]))
    stream = {**latency58_stream_metadata(plan), "track": track["name"], "actual_forward_call_count": calls,
              "actual_literal_read_count": calls, "actual_received_samples": cursor, "coverage_complete": True,
              "physical_alignment_verified_by_delayed_mixture": True,
              "runtime": "ONNX Runtime CPU1 literal hop128", "precision": "Ten U8S8 projections, FP64 quantizer ancestors, FP32 output decoding and six public states",
              "graph_deployed_outputs_captured": True, "maximum_graph_closure": maximum_closure,
              "elapsed_seconds": time.monotonic() - began}
    reconstruction = max(float(np.abs(estimate.sum(axis=0, dtype=np.float32) - mixture.astype(np.float32)).max())
                         for estimate, mixture in zip(estimates, mixtures, strict=True))
    return index, scored, stream, reconstruction


def main():
    import onnxruntime as ort
    import torch
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.latency58_attention_int8 import build
    from research.direct.latency58_best_onnx import interface
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.train_latency58 import state_sha256
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 workers and shipping ORT")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget_path = PHASE / "temporal-attention-001/plan.json"
    budget = read(budget_path)
    counted_before = require_space(budget, 5_000_000)
    screen_root = PHASE / "attention-int8-screen-001"
    screen, screen_plan = read(screen_root / "result.json"), read(screen_root / "plan.json")
    screen_execution_path = PHASE / "attention-int8-screen-stage-001/execution.json"
    execution = read(screen_execution_path)
    require(screen["status"] == "pass" and screen["strict_parity_passed"] and screen["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"], "Complete the independent short screen first")
    verify_inputs(screen_plan)
    native, payload, training, parent, reference_path, review_path = selected_endpoint()
    fingerprint, contract = state_sha256(native.state_dict()), interface(native)
    integer, graph, conversion = build(native, payload, training, parent, reference_path)
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"] and len(data) == screen["graph_bytes"]
            and state_sha256(native.state_dict()) == fingerprint, "Exact checked graph or native model changed")
    del integer, graph, native
    reference_result = read(reference_path)
    reference = reference_result["results"][0]
    prior_path = PHASE / "fusion-refinement-001/full14/result.json"
    c204_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    prior, c204 = read(prior_path)["results"][0], read(c204_path)["results"][0]
    minimum = prior["aggregate"]["full_sdr_db"]
    require(minimum == 4.266897232064164 and reference["aggregate"]["full_sdr_db"] == 4.288099064999147,
            "Saved comparison endpoints differ")
    manifest, config = ROOT / "research/manifests/valid.json", ROOT / "research/eval_config.json"
    module = Path(ort.__file__).resolve()
    paths = [Path(__file__).resolve(), budget_path, reference_path, prior_path, c204_path, review_path,
             screen_root / "plan.json", screen_root / "result.json", screen_execution_path, manifest, config,
             ROOT / "research/direct/evaluate_latency58_int8.py", ROOT / "research/direct/latency58_attention_int8.py",
             ROOT / "research/direct/latency58_attention_int8_verify.py", module,
             *sorted((module.parent / "capi").glob("*.so*"))]
    bindings = {**screen_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    checkpoint = {"kind": "in_memory_onnx", "sha256": screen["graph_sha256"], "bytes": len(data),
                  "saved": False, "source_checkpoint": parent}
    out = PHASE / "attention-int8-full14-memory-001"
    require(not out.exists(), "Preserve quality evaluations")
    out.mkdir()
    plan = {"source_bindings": bindings, "checkpoint": checkpoint, "graph_saved": False, "interface": contract,
            "reference_result": str(reference_path), "previous_research_best_result": str(prior_path),
            "c204_reference_result": str(c204_path), "counted_bytes_before": counted_before,
            "manifest": {"path": str(manifest), "sha256": sha(manifest)},
            "config": {"path": str(config), "sha256": sha(config)}, "workers": 2,
            "track_indices": list(range(14)), "source_order": ["drums", "bass", "vocals", "other"],
            "graph_delay_samples": 128, "host_queue_samples": 128, "graph_calls_are_literal_128_samples": True,
            "continuous_prefix_and_gap_input": True, "audio_saved": False,
            "minimum_full_sdr_for_handoff_db": minimum, "selected_fp32_checkpoint_full_sdr_db": 4.288099064999147,
            "decision_rule": "Keep the saved best checkpoint; require the integer graph to exceed the previous FP32 research best before M4 handoff. Retain all per-track/stem/band/absence regressions. Long reference parity and quiet CPU timing also remain required.",
            "runtime": {"version": ort.__version__, "python_module": str(module)}}
    require(plan["manifest"]["sha256"] == reference_result["source_bindings"][str(manifest)]
            and plan["config"]["sha256"] == reference_result["source_bindings"][str(config)], "Validation protocol changed")
    write(out / "plan.json", plan)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan, data)) as pool:
        futures = {pool.submit(score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure = future.result()
            require(index == futures[future] and index not in scored, "Wrong or repeated quality track")
            scored[index] = track, stream, closure
            row = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                   "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    tracks = [scored[i][0] for i in range(14)]
    require([t["name"] for t in tracks] == [t["name"] for t in reference["tracks"]], "Full14 order changed")
    candidate = {"model": {"label": "Saved attention checkpoint, ten signed integer projections",
                          "runtime_artifact": checkpoint, "source_model_state_sha256": fingerprint,
                          "training_updates": reference["model"]["training_updates"], "additional_training_updates": 0,
                          "runtime_precision_changed": True}, "checkpoint": checkpoint, "tracks": tracks,
                 "aggregate": legacy._aggregate_tracks(tracks), "stream_batches": [scored[i][1] for i in range(14)],
                 "reconstruction_max_abs": max(row[2] for row in scored.values())}
    verify_inputs(plan)
    quality_passed = candidate["aggregate"]["full_sdr_db"] >= minimum
    result = {"status": "pass", "quality_handoff_gate_passed": quality_passed, "results": [candidate],
              "comparison": compare(reference, candidate), "previous_research_best_comparison": compare(prior, candidate),
              "c204_comparison": compare(c204, candidate), "all_track_stem_cells": music_cells(reference, candidate),
              "previous_research_best_all_track_stem_cells": music_cells(prior, candidate),
              "c204_all_track_stem_cells": music_cells(c204, candidate),
              "track_count": 14, "excerpt_count": 28, "source_bindings": bindings, "source_bindings_unchanged": True,
              "plan_sha256": sha(out / "plan.json"), "runtime": plan["runtime"], "graph_sha256": screen["graph_sha256"],
              "graph_bytes": len(data), "graph_saved": False, "saved_graph_qualified": False,
              "native_host_qualified": False, "plugin_modified": False, "gpu_used": False,
              "graph_delay_samples": 128, "host_queue_samples": 128, "elapsed_seconds": time.monotonic() - began,
              "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5., "counted_bytes_after": require_space(budget, 0)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "quality_handoff_gate_passed": quality_passed,
                      "delta_from_selected_fp32_checkpoint": candidate["aggregate"]["full_sdr_db"] - reference["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
