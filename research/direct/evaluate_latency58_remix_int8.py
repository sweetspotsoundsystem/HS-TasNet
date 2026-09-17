"""Save precise integer full-magnitude inference and score the unchanged full14 panel."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct import evaluate_latency58_int8 as common
from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

PRECISION = "U8U8 projections; FP64 floating quantizer ancestors and magnitude projection; FP32 decoding and public state"


def initialize(plan):
    import onnxruntime as ort
    require(ort.__version__ == "1.26.0" and str(Path(ort.__file__).resolve()) == plan["runtime"]["python_module"],
            "Wrong worker runtime")
    common.initialize(plan)


def score(index):
    index, track, stream, closure = common.score(index)
    stream["precision"] = PRECISION
    return index, track, stream, closure


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 workers")
    import onnx
    import onnxruntime as ort
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.latency58_pending_training_space import require_cpu_space
    from research.direct.latency58_magnitude_int8 import build
    from research.direct.latency58_full_magnitude_checkpoint import load_model
    from research.direct.train_latency58 import state_sha256
    require(ort.__version__ == "1.26.0", "Use shipping ORT")
    source_path = PHASE / "quadrature-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted_before = require_cpu_space(source, 35_000_000)
    screen_path = PHASE / "m4-remix-int8-screen-001/result.json"
    long_path = PHASE / "m4-remix-int8-long-001/result.json"
    screen, long = read(screen_path), read(long_path)
    prerequisite_executions = [PHASE / name / "execution.json" for name in
                              ("m4-remix-int8-screen-stage-001", "m4-remix-int8-long-stage-001")]
    for path in prerequisite_executions:
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "An independent parity stage has not closed successfully")
    require(screen["strict_parity_passed"] and screen["float_export_parity_passed"]
            and screen["source_bindings_unchanged"] and long["source_bindings_unchanged"]
            and all(row["existing_strict_tolerances_passed"] for row in long["cases"]),
            "Both independent parity checks must pass")
    screen_plan, long_plan = read(screen_path.parent / "plan.json"), read(long_path.parent / "plan.json")
    verify_inputs(screen_plan)
    verify_inputs(long_plan)
    import statistics
    medians = {label: statistics.median(row["timing"]["p50_ms"] for row in screen["timings"] if row["variant"] == label)
               for label in ("float", "precise")}
    require(medians["precise"] < medians["float"], "No measured CPU speed benefit")
    parent = screen["checkpoint"]
    native, _ = load_model(parent)
    fingerprint = state_sha256(native.state_dict())
    float_graph, integer_graph, graph, proof = build(native)
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"] == long_plan["graph_sha256"]
            and len(data) < 35_000_000, "Graph differs from both completed parity checks")
    del native, float_graph, integer_graph
    reference_path = PHASE / "remix-magnitude-001/full14/result.json"
    reference_result = read(reference_path)
    reference = reference_result["results"][0]
    require(reference["model"]["model_state_sha256_after"] == fingerprint
            and reference["checkpoint"] == parent and reference["aggregate"]["full_sdr_db"] == 4.2127785034162875,
            "Wrong best saved magnitude reference")
    c204_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    c204_result = read(c204_path)
    module = Path(ort.__file__).resolve()
    manifest, config = ROOT / "research/manifests/valid.json", ROOT / "research/eval_config.json"
    paths = [ROOT / "research/direct/latency58_pending_training_space.py", *prerequisite_executions,
             source_path, screen_path, screen_path.parent / "plan.json", long_path, long_path.parent / "plan.json",
             reference_path, c204_path, Path(parent["path"]), manifest, config, Path(__file__).resolve(),
             ROOT / "research/direct/evaluate_latency58_int8.py", ROOT / "research/direct/latency58_magnitude_int8.py",
             module, *sorted((module.parent / "capi").glob("*.so*"))]
    bindings = {**long_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    out = PHASE / "m4-remix-int8-full14-001"
    require(not out.exists(), "Preserve full14 evaluations")
    out.mkdir()
    write(out / "save-intent.json", {"source_bindings": bindings, "graph_sha256": screen["graph_sha256"],
          "graph_bytes": len(data), "counted_bytes_before": counted_before, "reserved_pending_training_bytes": 380000000})
    graph_path = out / "model.onnx"
    with graph_path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(sha(graph_path) == screen["graph_sha256"], "Saved graph differs")
    bindings[str(graph_path)] = sha(graph_path)
    checkpoint = {"path": str(graph_path), "sha256": sha(graph_path)}
    del graph, data
    plan = {"source_bindings": bindings, "checkpoint": checkpoint, "reference_result": str(reference_path),
            "parent_float_checkpoint": parent, "c204_reference_result": str(c204_path),
            "manifest": {"path": str(manifest), "sha256": sha(manifest)},
            "config": {"path": str(config), "sha256": sha(config)}, "workers": 2,
            "track_indices": list(range(14)), "source_order": ["drums", "bass", "vocals", "other"],
            "graph_delay_samples": 128, "host_queue_samples": 128, "graph_calls_are_literal_128_samples": True,
            "continuous_prefix_and_gap_input": True, "audio_saved": False, "precision": PRECISION,
            "runtime": {"version": ort.__version__, "python_module": str(module)}}
    require(plan["manifest"]["sha256"] == reference_result["source_bindings"][str(manifest)]
            and plan["config"]["sha256"] == reference_result["source_bindings"][str(config)], "Validation protocol changed")
    write(out / "plan.json", plan)
    require_cpu_space(source, 0)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,)) as pool:
        futures = {pool.submit(score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure = future.result()
            require(index == futures[future] and index not in scored, "Wrong track index")
            scored[index] = track, stream, closure
            row = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                   "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    tracks = [scored[index][0] for index in range(14)]
    require([t["name"] for t in tracks] == [t["name"] for t in reference["tracks"]], "Track order changed")
    candidate = {"model": {"label": "Remix magnitude U8U8 with precise quantizer inputs", "runtime_artifact": checkpoint,
                          "source_model_state_sha256": reference["model"]["model_state_sha256_after"],
                          "training_updates": reference["model"]["training_updates"], "additional_training_updates": 0, "runtime_precision_changed": True},
                 "checkpoint": checkpoint, "tracks": tracks, "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [scored[index][1] for index in range(14)],
                 "reconstruction_max_abs": max(row[2] for row in scored.values())}
    verify_inputs(plan)
    result = {"status": "pass", "results": [candidate], "comparison": compare(reference, candidate),
              "c204_comparison": compare(c204_result["results"][0], candidate),
              "all_track_stem_cells": music_cells(reference, candidate), "track_count": 14, "excerpt_count": 28,
              "source_bindings": bindings, "source_bindings_unchanged": True, "plan_sha256": sha(out / "plan.json"),
              "runtime": plan["runtime"], "elapsed_seconds": time.monotonic() - began,
              "graph_delay_samples": 128, "host_queue_samples": 128,
              "native_host_qualified": False, "plugin_modified": False, "gpu_used": False,
              "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.,
              "counted_bytes_after": require_cpu_space(source, 0)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "delta_from_saved_magnitude": candidate["aggregate"]["full_sdr_db"] - reference["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
