"""Unchanged full14 scoring with explicit roles in a packed saved checkpoint."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write

_MODEL = _IDENTITY = None


def initialize(plan):
    global _MODEL, _IDENTITY
    import torch
    from research.direct.latency58_four_second_shared_evaluation import load_model
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _MODEL, _ = load_model(plan, {"checkpoint": plan["checkpoint"], "checkpoint_role": plan["checkpoint_role"],
                                "model_state_sha256": plan["model_state_sha256"]})
    _IDENTITY = {"label": plan["label"], "state_kind": "checkpoint", "checkpoint": plan["checkpoint"],
                 "training_updates": _MODEL.provenance["training_updates"], "provenance": _MODEL.provenance,
                 "model_state_sha256": model_state_sha256(_MODEL), "checkpoint_role": plan["checkpoint_role"],
                 "packed_training_plan": plan["packed_training_plan"]}


def score(index):
    from research.direct.latency58_evaluate import evaluate_latency58_music
    report = evaluate_latency58_music(_MODEL, identity=_IDENTITY, track_indices=[index])
    require(report["excerpt_count"] == 2 and report["graph_alignment_samples"] == 128, "Protocol changed")
    return index, report["results"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Branch-memory scoring plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-packed-branch-memory-full14-plan-v1" and plan["track_indices"] == list(range(14))
            and plan["workers"] == 2 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Require frozen CPU full14 scoring")
    out = Path(plan["output_directory"])
    from research.direct.latency58_four_second_shared_evaluation import evaluation_storage
    before = evaluation_storage(plan)
    began, reports = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(score, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, report = future.result()
            require(index == futures[future] and index not in reports, "Wrong or repeated track")
            reports[index] = report
            row = {"event": "track", "index": index, "track": report["tracks"][0]["name"],
                   "full_sdr_db": report["aggregate"]["full_sdr_db"]}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    tracks = [reports[i]["tracks"][0] for i in plan["track_indices"]]
    candidate = {"model": reports[0]["model"], "checkpoint": plan["checkpoint"], "checkpoint_role": plan["checkpoint_role"], "tracks": tracks,
                 "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [reports[i]["stream_batches"][0] for i in plan["track_indices"]],
                 "reconstruction_max_abs": max(r["reconstruction_max_abs"] for r in reports.values())}
    reference = read(plan["reference_result"])["results"][0]
    require([t["name"] for t in tracks] == [t["name"] for t in reference["tracks"]]
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Track panel or evaluation inputs changed")
    write(out / "result.json", {"schema": "latency58-packed-branch-memory-full14-result-v1", "status": "pass",
          "results": [candidate], "comparison": compare(reference, candidate),
          "all_track_stem_cells": music_cells(reference, candidate), "plan_sha256": args.plan_sha256,
          "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "track_count": 14, "excerpt_count": 28, "elapsed_seconds": time.monotonic() - began,
          "target_full_sdr_db": 5.0, "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0,
          "graph_delay_samples": 128, "host_queue_samples": 128, "new_checkpoint_host_qualified": False,
          "checkpoint_role": plan["checkpoint_role"], "model_state_sha256": plan["model_state_sha256"],
          "packed_training_plan": plan["packed_training_plan"],
          "storage_before": before, "storage_after": evaluation_storage(plan)})
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0}), flush=True)


if __name__ == "__main__":
    main()
