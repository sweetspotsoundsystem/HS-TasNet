"""Save the screened rank-128 graph and score the unchanged full14 protocol."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct import evaluate_latency58_int8 as stream_evaluation
from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def initialize(plan):
    import onnxruntime as ort
    require(ort.__version__ == "1.26.0" and str(Path(ort.__file__).resolve()) == plan["runtime"]["python_module"],
            "Wrong worker runtime")
    stream_evaluation.initialize(plan)


def score(index):
    index, track, stream, closure = stream_evaluation.score(index)
    # The shared evaluator executes whichever authenticated graph the plan
    # supplies. This candidate uses FP32 factors rather than integer matrices.
    stream["precision"] = "FP32 throughout; rank-128 spectral mask projection"
    return index, track, stream, closure


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 workers")
    import onnx
    import onnxruntime as ort
    import torch
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_low_rank_mask import convert
    require(ort.__version__ == "1.26.0", "Use the shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "full-magnitude-sdr-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted_before = require_space(source, 472_000_000)
    screen_path = PHASE / "m4-rank128-mask-001/result.json"
    screen = read(screen_path)
    screen_plan = read(screen_path.parent / "plan.json")
    require(screen["status"] == "screen_complete" and screen["advance_to_full14"]
            and screen["parity_passed"] and screen["source_bindings_unchanged"], "Screen did not pass")
    verify_inputs(screen_plan)
    parent_plan_path = PHASE / "full-magnitude-001/plan.json"
    parent_binding = read(parent_plan_path)["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Wrong C204 source")
    original_path = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    require(sha(original_path) == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3",
            "C204 graph changed")
    graph = onnx.load(original_path, load_external_data=False)
    proof = convert(native, graph)
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"]
            and proof == screen["factorization"] and len(data) == screen["graph_bytes"]
            and state_sha256(native.state_dict()) == screen["candidate_model_state_sha256"],
            "Recreated factors or graph differ from the completed screen")
    onnx.checker.check_model(graph, full_check=True)
    out = PHASE / "m4-rank128-mask-full14-001"
    require(not out.exists(), "Preserve full14 results")
    reference_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    reference_result = read(reference_path)
    reference = reference_result["results"][0]
    require(reference["model"]["model_state_sha256_after"] == BASE_STATE, "Wrong reference")
    module = Path(ort.__file__).resolve()
    paths = [source_path, screen_path, screen_path.parent / "plan.json", reference_path,
             ROOT / "research/manifests/valid.json", ROOT / "research/eval_config.json", Path(__file__).resolve(),
             ROOT / "research/direct/evaluate_latency58_int8.py", module,
             *sorted((module.parent / "capi").glob("*.so*"))]
    bindings = {**screen_plan["source_bindings"], **{str(path): sha(path) for path in paths}}
    out.mkdir()
    write(out / "save-intent.json", {"graph_sha256": screen["graph_sha256"], "graph_bytes": len(data),
          "source_bindings": bindings, "counted_bytes_before": counted_before,
          "pending_training_save_reserved_bytes": 370_000_000})
    checkpoint_path = out / "model.onnx"
    with checkpoint_path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(sha(checkpoint_path) == screen["graph_sha256"], "Saved graph differs")
    bindings[str(checkpoint_path)] = sha(checkpoint_path)
    checkpoint = {"path": str(checkpoint_path), "sha256": sha(checkpoint_path)}
    del data, graph, native, parent
    plan = {"source_bindings": bindings, "checkpoint": checkpoint, "reference_result": str(reference_path),
            "manifest": {"path": str(paths[4]), "sha256": sha(paths[4])},
            "config": {"path": str(paths[5]), "sha256": sha(paths[5])}, "track_indices": list(range(14)),
            "source_order": ["drums", "bass", "vocals", "other"], "workers": 2,
            "graph_delay_samples": 128, "host_queue_samples": 128,
            "graph_calls_are_literal_128_samples": True, "continuous_prefix_and_gap_input": True,
            "precision": "FP32 throughout; rank-128 spectral mask projection", "audio_saved": False,
            "runtime": {"version": ort.__version__, "python_module": str(module)},
            "factorization": proof, "quality_screen_result": str(screen_path)}
    require(plan["manifest"]["sha256"] == reference_result["manifest_sha256"]
            and plan["config"]["sha256"] == reference_result["config_sha256"], "Validation protocol changed")
    write(out / "plan.json", plan)
    require_space(source, 370_000_000)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,)) as pool:
        futures = {pool.submit(score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure = future.result()
            require(index == futures[future] and index not in scored, "Wrong validation index")
            scored[index] = track, stream, closure
            row = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                   "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    tracks = [scored[index][0] for index in range(14)]
    require([t["name"] for t in tracks] == [t["name"] for t in reference["tracks"]], "Track order changed")
    candidate = {"model": {"label": "C204 FP32 rank-128 spectral mask", "runtime_artifact": checkpoint,
                          "source_model_state_sha256": BASE_STATE, "training_updates": 8250,
                          "additional_training_updates": 0, "runtime_precision_changed": False,
                          "spectral_mask_factorization_rank": 128},
                 "checkpoint": checkpoint, "tracks": tracks, "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [scored[index][1] for index in range(14)],
                 "reconstruction_max_abs": max(row[2] for row in scored.values())}
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "CPU evaluation initialized CUDA")
    result = {"status": "pass", "results": [candidate], "comparison": compare(reference, candidate),
              "all_track_stem_cells": music_cells(reference, candidate), "track_count": 14, "excerpt_count": 28,
              "source_bindings": bindings, "source_bindings_unchanged": True, "plan_sha256": sha(out / "plan.json"),
              "runtime": plan["runtime"], "elapsed_seconds": time.monotonic() - began,
              "graph_delay_samples": 128, "host_queue_samples": 128,
              "native_host_qualified": False, "plugin_modified": False, "gpu_used": False,
              "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0,
              "counted_bytes_after": require_space(source, 370_000_000)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "delta_from_c204": candidate["aggregate"]["full_sdr_db"] - reference["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
