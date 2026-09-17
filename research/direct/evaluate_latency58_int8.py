"""Score saved integer ONNX inference on the unchanged continuous full14 panel."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

_SESSION = _PLAN = None


def initialize(plan):
    global _SESSION, _PLAN
    import torch
    from research.direct.check_latency58_fused_gru import session_for
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized() and sha(plan["checkpoint"]["path"]) == plan["checkpoint"]["sha256"],
            "Require unchanged CPU inference artifact")
    _PLAN = plan
    _SESSION = session_for(Path(plan["checkpoint"]["path"]).read_bytes())


def score(index):
    import numpy as np
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import plan_latency58_stream, latency58_stream_metadata, SOURCE_ORDER
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, _initial_states
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
    states = _initial_states()
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
              "runtime": "ONNX Runtime CPU1 literal hop128", "precision": "U8U8 matrix operations; FP32 public state, FFT and synthesis",
              "graph_deployed_outputs_captured": True, "maximum_graph_closure": maximum_closure,
              "elapsed_seconds": time.monotonic() - began}
    reconstruction = max(float(np.abs(estimate.sum(axis=0, dtype=np.float32) - mixture.astype(np.float32)).max())
                         for estimate, mixture in zip(estimates, mixtures, strict=True))
    return index, scored, stream, reconstruction


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 workers")
    from research.direct.latency58_sdr_checkpoint import require_space
    source_path = PHASE / "full-magnitude-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    require_space(source, 375_000_000)
    screen_path = PHASE / "m4-int8-screen-002/result.json"
    screen = read(screen_path)
    require(screen["status"] == "screen_complete" and screen["source_bindings_unchanged"], "Quantization screen did not finish")
    checkpoint = {key: screen["quantized"][key] for key in ("path", "sha256")}
    reference_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    reference = read(reference_path)
    require(reference["results"][0]["model"]["model_state_sha256_after"] ==
            "c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb", "Wrong FP32 reference")
    out = PHASE / "m4-int8-full14-001"
    require(not out.exists(), "Preserve full-panel evaluations")
    paths = [source_path, screen_path, PHASE / "m4-int8-screen-002/plan.json", reference_path,
             ROOT / "research/manifests/valid.json", ROOT / "research/eval_config.json", Path(__file__).resolve(),
             ROOT / "research/direct/latency58_evaluate.py", ROOT / "research/direct/latency58_asymmetric_onnx.py",
             ROOT / "research/direct/evaluate.py", ROOT / "research/evaluate.py", ROOT / "research/metrics.py",
             ROOT / "research/direct/check_latency58_fused_gru.py", Path(checkpoint["path"])]
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    plan = {"source_bindings": bindings, "checkpoint": checkpoint, "reference_result": str(reference_path),
            "manifest": {"path": str(paths[4]), "sha256": sha(paths[4])},
            "config": {"path": str(paths[5]), "sha256": sha(paths[5])}, "track_indices": list(range(14)), "workers": 2,
            "source_order": ["drums", "bass", "vocals", "other"], "graph_delay_samples": 128, "host_queue_samples": 128,
            "graph_calls_are_literal_128_samples": True, "continuous_prefix_and_gap_input": True, "audio_saved": False}
    require(plan["manifest"]["sha256"] == reference["manifest_sha256"]
            and plan["config"]["sha256"] == reference["config_sha256"], "Validation protocol differs")
    out.mkdir()
    write(out / "plan.json", plan)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,)) as pool:
        futures = {pool.submit(score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure = future.result()
            require(index == futures[future] and index not in scored, "Wrong or duplicate validation result")
            scored[index] = (track, stream, closure)
            row = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                   "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    tracks = [scored[index][0] for index in range(14)]
    require([track["name"] for track in tracks] == [track["name"] for track in reference["results"][0]["tracks"]],
            "Full14 order changed")
    candidate = {"model": {"label": "C204 dynamic U8U8 inference", "runtime_artifact": checkpoint,
                          "source_model_state_sha256": reference["results"][0]["model"]["model_state_sha256_after"],
                          "training_updates": 8250, "additional_training_updates": 0, "runtime_precision_changed": True},
                 "checkpoint": checkpoint, "tracks": tracks, "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [scored[index][1] for index in range(14)],
                 "reconstruction_max_abs": max(row[2] for row in scored.values())}
    verify_inputs(plan)
    result = {"status": "pass", "results": [candidate], "comparison": compare(reference["results"][0], candidate),
              "all_track_stem_cells": music_cells(reference["results"][0], candidate), "track_count": 14, "excerpt_count": 28,
              "source_bindings": bindings, "source_bindings_unchanged": True, "plan_sha256": sha(out / "plan.json"),
              "elapsed_seconds": time.monotonic() - began, "graph_delay_samples": 128, "host_queue_samples": 128,
              "native_host_qualified": False, "gpu_used": False, "plugin_modified": False,
              "target_full_sdr_db": 5.0, "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0,
              "counted_bytes_after": require_space(source, 370_000_000)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "delta_from_fp32": candidate["aggregate"]["full_sdr_db"] - reference["results"][0]["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
