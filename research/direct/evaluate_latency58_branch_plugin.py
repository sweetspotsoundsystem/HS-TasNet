"""Measure the saved eight-state deployment graph on the unchanged full14 panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.evaluate_latency58_attention_int8_memory import initialize


def score(index):
    from research.direct.evaluate_latency58_attention_int8_memory import score as literal_score
    index, track, stream, closure = literal_score(index)
    stream["precision"] = "Ten U8S8 projections, FP64 quantizer ancestors, FP32 decoding and eight public states"
    return index, track, stream, closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CPU1 workers without CUDA")
    import onnxruntime as ort
    import torch
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(ort.__version__ == "1.26.0", "Use shipping runtime")
    screen_root = args.screen.resolve()
    screen, screen_plan = read(screen_root / "result.json"), read(screen_root / "plan.json")
    execution = read(screen_root / "execution.json")
    require(screen["status"] == "pass" and screen["strict_parity_passed"]
            and screen["source_bindings_unchanged"] and execution["actual_exit_code"] == 0,
            "Complete the actual short parity execution first")
    graph_path = screen_root / "model.onnx"
    data = graph_path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"]
            and len(data) == screen["graph_bytes"], "Saved deployment graph changed")
    reference_path = Path(screen_plan["quality_result"])
    reference_result = read(reference_path)
    reference = reference_result["results"][0]
    previous_path = PHASE / "attention-int8-full14-memory-001/result.json"
    c204_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    previous, c204 = (read(p)["results"][0] for p in (previous_path, c204_path))
    budget = read(reference_path.parents[1] / "plan.json")
    counted = require_space(budget, 600_000_000 + 5_000_000)
    manifest, config = ROOT / "research/manifests/valid.json", ROOT / "research/eval_config.json"
    module = Path(ort.__file__).resolve()
    paths = [Path(__file__).resolve(), reference_path, previous_path, c204_path, manifest, config,
             screen_root / "plan.json", screen_root / "result.json", screen_root / "execution.json", graph_path,
             ROOT / "research/direct/evaluate_latency58_attention_int8_memory.py",
             ROOT / "research/direct/latency58_attention_int8_verify.py", module,
             *sorted((module.parent / "capi").glob("*.so*"))]
    bindings = {**screen_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = args.output.resolve()
    require(out.parent == PHASE and not out.exists(), "Preserve previous full14 evidence")
    out.mkdir()
    checkpoint = {"kind": "saved_onnx", "path": str(graph_path), "sha256": screen["graph_sha256"],
                  "bytes": len(data), "saved": True, "source_checkpoint": screen["checkpoint"]}
    plan = {"source_bindings": bindings, "checkpoint": checkpoint, "graph_saved": True,
            "interface": screen_plan["interface"], "reference_result": str(reference_path),
            "previous_pr_deployment_result": str(previous_path), "c204_reference_result": str(c204_path),
            "counted_bytes_before": counted, "manifest": {"path": str(manifest), "sha256": sha(manifest)},
            "config": {"path": str(config), "sha256": sha(config)}, "workers": 2,
            "track_indices": list(range(14)), "source_order": ["drums", "bass", "vocals", "other"],
            "graph_delay_samples": 128, "host_queue_samples": 128, "graph_calls_are_literal_128_samples": True,
            "continuous_prefix_and_gap_input": True, "audio_saved": False,
            "minimum_full_sdr_for_handoff_db": previous["aggregate"]["full_sdr_db"],
            "selected_fp32_checkpoint_full_sdr_db": reference["aggregate"]["full_sdr_db"],
            "decision_rule": "Require improvement over the previous PR graph; review all source and baseline per-stem/track regressions. M4 timing and listening remain separate.",
            "runtime": {"version": ort.__version__, "python_module": str(module)}}
    require(plan["manifest"]["sha256"] == reference_result["source_bindings"][str(manifest)]
            and plan["config"]["sha256"] == reference_result["source_bindings"][str(config)],
            "The validation protocol changed")
    write(out / "plan.json", plan)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(plan, data)) as pool:
        futures = {pool.submit(score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure = future.result()
            require(index == futures[future] and index not in scored, "Repeated or incorrect quality track")
            scored[index] = track, stream, closure
            row = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                   "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    tracks = [scored[i][0] for i in range(14)]
    require([t["name"] for t in tracks] == [t["name"] for t in reference["tracks"]], "Full14 track order changed")
    candidate = {"model": {"label": "Saved branch-memory EMA, ten signed integer projections",
                          "runtime_artifact": checkpoint, "source_model_state_sha256": screen["model_state_sha256"],
                          "training_updates": 39250, "additional_training_updates": 0,
                          "runtime_precision_changed": True}, "checkpoint": checkpoint, "tracks": tracks,
                 "aggregate": legacy._aggregate_tracks(tracks), "stream_batches": [scored[i][1] for i in range(14)],
                 "reconstruction_max_abs": max(row[2] for row in scored.values())}
    verify_inputs(plan)
    quality_passed = candidate["aggregate"]["full_sdr_db"] > plan["minimum_full_sdr_for_handoff_db"]
    result = {"status": "pass", "quality_handoff_gate_passed": quality_passed, "results": [candidate],
              "source_checkpoint_comparison": compare(reference, candidate),
              "previous_pr_comparison": compare(previous, candidate), "c204_comparison": compare(c204, candidate),
              "source_checkpoint_all_track_stem_cells": music_cells(reference, candidate),
              "previous_pr_all_track_stem_cells": music_cells(previous, candidate),
              "c204_all_track_stem_cells": music_cells(c204, candidate),
              "track_count": 14, "excerpt_count": 28, "source_bindings": bindings, "source_bindings_unchanged": True,
              "plan_sha256": sha(out / "plan.json"), "runtime": plan["runtime"], "graph_sha256": screen["graph_sha256"],
              "graph_bytes": len(data), "graph_saved": True, "native_host_qualified": False,
              "gpu_used": False, "graph_delay_samples": 128, "host_queue_samples": 128,
              "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_after": require_space(budget, 600_000_000)}
    write(out / "result.json", result)
    require(quality_passed, "The deployment graph did not improve on the prior PR")
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
