"""Rescore the saved residual model with the unchanged full14 evaluator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute

_MODEL = _IDENTITY = None


def initialize(plan):
    global _MODEL, _IDENTITY
    import torch
    from research.direct.latency58_residual_model import load_checkpoint
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = plan["checkpoint"]
    _MODEL, _ = load_checkpoint(checkpoint["path"], checkpoint["sha256"])
    _IDENTITY = {"label": "C204 fixed residual 1/16 saved model", "state_kind": "checkpoint",
                 "training_updates": _MODEL.provenance["training_updates"], "checkpoint": checkpoint,
                 "model_state_sha256": model_state_sha256(_MODEL), "provenance": _MODEL.provenance}


def score(index):
    from research.direct.latency58_evaluate import evaluate_latency58_music
    result = evaluate_latency58_music(_MODEL, identity=_IDENTITY, track_indices=[index])
    require(result["excerpt_count"] == 2 and result["graph_alignment_samples"] == 128,
            "Different original evaluation intervals or physical alignment")
    return index, result["results"][0]


def run(plan_path, plan_sha):
    require(sha(plan_path) == plan_sha, "Saved-model evaluation plan changed")
    plan = read(plan_path)
    require(plan["schema"] == "latency58-saved-residual-full14-plan-v1"
            and plan["workers"] == 2 and plan["track_indices"] == list(range(14))
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Require frozen CPU full14 evaluation")
    out = Path(plan["output_directory"])
    began = time.monotonic()
    reports = {}
    diagnostic = read(plan["diagnostic_result"]["path"])
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(score, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, result = future.result()
            require(index == futures[future] and index not in reports, "Wrong or repeated saved-model track")
            require(result["tracks"][0] == diagnostic["policies"]["fixed_share"]["tracks"][index],
                    "Saved model failed exact diagnostic per-track replay")
            reports[index] = result
            row = {"event": "track", "index": index, "exact_diagnostic_replay": True,
                   "full_sdr_db": result["tracks"][0]["full_sdr_db"]}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    tracks = [reports[i]["tracks"][0] for i in plan["track_indices"]]
    aggregate = legacy._aggregate_tracks(tracks)
    require(aggregate == diagnostic["policies"]["fixed_share"]["aggregate"], "Saved model aggregate differs")
    candidate = {"model": reports[0]["model"], "checkpoint": plan["checkpoint"], "tracks": tracks,
                 "aggregate": aggregate, "stream_batches": [reports[i]["stream_batches"][0] for i in plan["track_indices"]],
                 "reconstruction_max_abs": max(r["reconstruction_max_abs"] for r in reports.values())}
    reference = diagnostic["policies"]["working_policy"]
    comparison = compare(reference, candidate)
    require(comparison["metrics"]["full_sdr_db"]["delta"] > 0
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "SDR failed to improve or inputs changed")
    write(out / "result.json", {"schema": "latency58-saved-residual-full14-result-v1", "status": "pass",
          "plan_sha256": plan_sha, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "results": [candidate], "comparison": comparison, "all_track_stem_cells": music_cells(reference, candidate),
          "exact_diagnostic_tracks_and_aggregate": True, "track_count": 14, "excerpt_count": 28,
          "streaming_state": "independent per track; continuous from sample zero",
          "graph_delay_samples": 128, "host_queue_samples": 128, "total_latency_samples": 256,
          "full14_host_queue_executed": False, "elapsed_seconds": time.monotonic() - began,
          "limitations": ["Development panel, not independent confirmation; small gain has a bootstrap interval crossing zero.",
                          "Interference rejection declines; retained separately from the working plugin."]})
    print(json.dumps({"status": "pass", "full_sdr_db": aggregate["full_sdr_db"],
                      "delta_db": comparison["metrics"]["full_sdr_db"]["delta"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.plan is not None:
        run(args.plan, args.plan_sha256)
        return
    source = PHASE / "c204-residual-model-001"
    qualification = read(source / "qualification.json")
    require(qualification["status"] == "pass" and qualification["total_delay_samples"] == 256,
            "Qualify the saved model before scoring")
    checkpoint = read(source / "checkpoint.json")
    paths = [Path(__file__).resolve(), source / "model.pt", source / "qualification.json", source / "checkpoint.json"]
    bindings = {**qualification["source_bindings"], **{str(p): sha(p) for p in paths}}
    out = PHASE / "c204-residual-model-full14-001"
    require(not out.exists() and all(sha(p) == s for p, s in bindings.items()), "Preserve evaluation and frozen inputs")
    diagnostic_path = PHASE / "c204-residual-share-full14-001/result.json"
    plan = {"schema": "latency58-saved-residual-full14-plan-v1", "workers": 2, "track_indices": list(range(14)),
            "checkpoint": checkpoint, "diagnostic_result": {"path": str(diagnostic_path), "sha256": sha(diagnostic_path)},
            "source_bindings": bindings, "output_directory": str(out)}
    out.mkdir()
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_residual_model", "--plan", str(out / "plan.json"),
             "--plan-sha256", sha(out / "plan.json")], out, "evaluation", 1800, bindings,
            {"plan_sha256": sha(out / "plan.json")})


if __name__ == "__main__":
    main()
