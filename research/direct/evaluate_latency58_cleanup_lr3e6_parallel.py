"""Evaluate independent tracks in two CPU1 processes using the original scorer.

Each worker calls evaluate_latency58_music unchanged with one complete track.
The parent restores manifest order and calls the original track aggregator.
This executor must reproduce stored serial results before selection use.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write

_MODEL = None
_IDENTITY = None


def initialize_worker(plan):
    global _MODEL, _IDENTITY
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    from research.direct.evaluate_latency58_cleanup_lr3e6 import load_evaluation_model
    _MODEL, _ = load_evaluation_model(plan)
    checkpoint = plan["checkpoint"]
    _IDENTITY = {"label": plan["label"], "state_kind": "checkpoint", "checkpoint": checkpoint,
                 "training_updates": _MODEL.provenance["training_updates"],
                 "provenance": _MODEL.provenance, "model_state_sha256": model_state_sha256(_MODEL)}
    require(_IDENTITY["model_state_sha256"] == plan["expected_model_state_sha256"], "Worker loaded different weights")


def score_track(index):
    from research.direct.latency58_evaluate import evaluate_latency58_music
    require(_MODEL is not None and _IDENTITY is not None, "Worker was not initialized")
    report = evaluate_latency58_music(_MODEL, identity=_IDENTITY, track_indices=[index])
    report["parallel_worker_pid"] = os.getpid()
    return index, report


from research.direct.evaluate_latency58_sdr_parallel import combine_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Parallel evaluation plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-cleanup-lr3e6-parallel-music-plan-v1" and plan["workers"] == 2
            and plan["model_kind"] == "cleanup_lr3e6_candidate", "Unsupported parallel evaluation")
    indices = plan["track_indices"]
    require(indices and all(type(i) is int and 0 <= i < 14 for i in indices)
            and indices == sorted(set(indices)), "Track selection must be unique and in manifest order")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1 workers")
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Evaluation inputs changed")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous result")
    began = time.monotonic()
    reports = {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(score_track, i): i for i in indices}
        for future in as_completed(futures):
            index, report = future.result()
            require(index == futures[future] and index not in reports, "Worker returned the wrong track")
            reports[index] = report
            row = {"event": "track", "index": index, "track": report["track_names"][0],
                   "full_sdr_db": report["results"][0]["aggregate"]["full_sdr_db"]}
            progress.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    require(set(reports) == set(indices), "Incomplete track coverage")
    result = combine_reports(reports, indices=indices, elapsed=time.monotonic() - began)
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "An evaluation input changed")
    result.update(plan_sha256=args.plan_sha256, root_source_bindings=plan["source_bindings"],
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - began)
    write(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
