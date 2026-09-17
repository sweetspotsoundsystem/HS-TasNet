"""Score an in-memory quadrature graph with all ten integer projections signed on unchanged full14 music.

This diagnostic writes no ONNX file and cannot qualify a saved deployment graph.
"""
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

PRECISION = "Ten U8S8 reduced-range projections; FP64 quantizer ancestors; FP32 phase factors, decoding and public state"


def initialize(plan, data):
    import onnxruntime as ort
    import torch
    from research.direct.check_latency58_fused_gru import session_for
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(ort.__version__ == "1.26.0" and str(Path(ort.__file__).resolve()) == plan["runtime"]["python_module"]
            and not torch.cuda.is_initialized() and not plan["graph_saved"]
            and len(data) == plan["checkpoint"]["bytes"]
            and hashlib.sha256(data).hexdigest() == plan["checkpoint"]["sha256"],
            "Worker runtime or in-memory graph identity differs")
    common._PLAN = plan
    common._SESSION = session_for(data)


def score(index):
    index, track, stream, closure = common.score(index)
    stream["precision"] = PRECISION
    stream["in_memory_graph_sha256_verified"] = common._PLAN["checkpoint"]["sha256"]
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
    from research.direct.latency58_temporal_pending_space import require_cpu_space as require_space
    from research.direct.latency58_quadrature_all_s8 import build
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.train_latency58 import state_sha256
    require(ort.__version__ == "1.26.0", "Use shipping ORT")
    source_path = PHASE / "temporal-attention-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted_before = require_space(source, 5_000_000)
    screen_path = PHASE / "quadrature-all-s8-screen-001/result.json"
    long_path = PHASE / "quadrature-all-s8-long-001/result.json"
    screen, long = read(screen_path), read(long_path)
    prerequisite_executions = [PHASE / name / "execution.json" for name in
                              ("quadrature-all-s8-screen-stage-001", "quadrature-all-s8-long-stage-001")]
    for path in prerequisite_executions:
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "An independent parity stage has not closed successfully")
    require(screen["strict_parity_passed"]
            and screen["source_bindings_unchanged"] and long["source_bindings_unchanged"]
            and all(row["existing_strict_tolerances_passed"] for row in long["cases"]),
            "Both independent parity checks must pass")
    screen_plan, long_plan = read(screen_path.parent / "plan.json"), read(long_path.parent / "plan.json")
    verify_inputs(screen_plan)
    verify_inputs(long_plan)
    parent = screen["checkpoint"]
    native, _ = load_model(parent)
    fingerprint = state_sha256(native.state_dict())
    original_path = PHASE / "m4-quadrature-magint8-saved-001/model.onnx"
    original = onnx.load(original_path)
    original_proof_path = PHASE / "m4-quadrature-magint8-screen-001/graph.json"
    graph, proof = build(native, original, read(original_proof_path)["graph_conversion"])
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"] == long_plan["graph_sha256"]
            and len(data) < 35_000_000, "Graph differs from both completed parity checks")
    del native, original
    reference_path = PHASE / "quadrature-001/full14/result.json"
    reference_result = read(reference_path)
    reference = reference_result["results"][0]
    require(reference["model"]["model_state_sha256_after"] == fingerprint
            and reference["checkpoint"] == parent and reference["aggregate"]["full_sdr_db"] == 4.227699923177355,
            "Wrong saved quadrature reference")
    unsigned_path = PHASE / "m4-quadrature-magint8-full14-memory-001/result.json"
    unsigned_result = read(unsigned_path)
    unsigned = unsigned_result["results"][0]
    require(unsigned_result["status"] == "pass" and unsigned_result["source_bindings_unchanged"]
            and unsigned_result["track_count"] == 14 and unsigned_result["excerpt_count"] == 28
            and unsigned["checkpoint"]["sha256"] == sha(original_path), "Wrong unsigned comparison")
    unsigned_execution = PHASE / "m4-quadrature-magint8-full14-memory-stage-001/execution.json"
    require(read(unsigned_execution)["actual_exit_code"] == 0
            and read(unsigned_execution)["source_bindings_unchanged"], "Unsigned evaluation not completed")
    encoder_path = PHASE / "quadrature-encoder-s8-full14-memory-001/result.json"
    encoder_result = read(encoder_path)
    encoder = encoder_result["results"][0]
    require(encoder_result["status"] == "pass" and encoder_result["source_bindings_unchanged"]
            and encoder["checkpoint"]["sha256"] == "2ae64f082db7f4f54196aab3132a29cd6b381a8f47764aff96b1e947b659db6e",
            "Wrong completed signed-encoder quality reference")
    c204_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    c204_result = read(c204_path)
    module = Path(ort.__file__).resolve()
    manifest, config = ROOT / "research/manifests/valid.json", ROOT / "research/eval_config.json"
    paths = [ROOT / "research/direct/latency58_temporal_pending_space.py", original_path, original_proof_path, unsigned_path, unsigned_execution, *prerequisite_executions,
             source_path, screen_path, screen_path.parent / "plan.json", long_path, long_path.parent / "plan.json",
             reference_path, c204_path, encoder_path, Path(parent["path"]), manifest, config, Path(__file__).resolve(),
             ROOT / "research/direct/evaluate_latency58_int8.py", ROOT / "research/direct/latency58_quadrature_all_s8.py",
             module, *sorted((module.parent / "capi").glob("*.so*"))]
    bindings = {**long_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    out = PHASE / "quadrature-all-s8-full14-memory-001"
    require(not out.exists(), "Preserve full14 evaluations")
    out.mkdir()
    write(out / "in-memory-graph.json", {"source_bindings": bindings, "graph_sha256": screen["graph_sha256"],
          "graph_bytes": len(data), "graph_saved": False, "source_checkpoint": parent,
          "counted_bytes_before": counted_before, "reserved_pending_training_bytes": 380_000_000,
          "limitation": "Quality diagnostic from authenticated graph bytes in memory; no saved deployment artifact is qualified."})
    checkpoint = {"kind": "in_memory_onnx", "sha256": screen["graph_sha256"], "bytes": len(data),
                  "saved": False, "source_checkpoint": parent}
    del graph
    plan = {"source_bindings": bindings, "checkpoint": checkpoint, "graph_saved": False, "reference_result": str(reference_path),
            "parent_float_checkpoint": parent, "unsigned_reference_result": str(unsigned_path), "signed_encoder_reference_result": str(encoder_path), "c204_reference_result": str(c204_path),
            "manifest": {"path": str(manifest), "sha256": sha(manifest)},
            "config": {"path": str(config), "sha256": sha(config)}, "workers": 2,
            "track_indices": list(range(14)), "source_order": ["drums", "bass", "vocals", "other"],
            "graph_delay_samples": 128, "host_queue_samples": 128, "graph_calls_are_literal_128_samples": True,
            "continuous_prefix_and_gap_input": True, "audio_saved": False, "precision": PRECISION,
            "runtime": {"version": ort.__version__, "python_module": str(module)}}
    require(plan["manifest"]["sha256"] == reference_result["source_bindings"][str(manifest)]
            and plan["config"]["sha256"] == reference_result["source_bindings"][str(config)], "Validation protocol changed")
    write(out / "plan.json", plan)
    require_space(source, 0)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan, data)) as pool:
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
    candidate = {"model": {"label": "In-memory quadrature with all ten integer matrices reduced-range signed", "runtime_artifact": checkpoint,
                          "source_model_state_sha256": reference["model"]["model_state_sha256_after"],
                          "training_updates": reference["model"]["training_updates"], "additional_training_updates": 0, "runtime_precision_changed": True},
                 "checkpoint": checkpoint, "tracks": tracks, "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [scored[index][1] for index in range(14)],
                 "reconstruction_max_abs": max(row[2] for row in scored.values())}
    verify_inputs(plan)
    result = {"status": "pass", "results": [candidate], "comparison": compare(reference, candidate),
              "c204_comparison": compare(c204_result["results"][0], candidate),
              "unsigned_comparison": compare(unsigned, candidate),
              "signed_encoder_comparison": compare(encoder, candidate),
              "signed_encoder_all_track_stem_cells": music_cells(encoder, candidate),
              "unsigned_all_track_stem_cells": music_cells(unsigned, candidate),
              "c204_all_track_stem_cells": music_cells(c204_result["results"][0], candidate),
              "all_track_stem_cells": music_cells(reference, candidate), "track_count": 14, "excerpt_count": 28,
              "source_bindings": bindings, "source_bindings_unchanged": True, "plan_sha256": sha(out / "plan.json"),
              "runtime": plan["runtime"], "elapsed_seconds": time.monotonic() - began,
              "graph_delay_samples": 128, "host_queue_samples": 128,
              "native_host_qualified": False, "plugin_modified": False, "gpu_used": False,
              "graph_saved": False, "saved_graph_qualified": False,
              "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.,
              "counted_bytes_after": require_space(source, 0)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "delta_from_unsigned": candidate["aggregate"]["full_sdr_db"] - unsigned["aggregate"]["full_sdr_db"],
                      "delta_from_saved_quadrature_fp32": candidate["aggregate"]["full_sdr_db"] - reference["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
