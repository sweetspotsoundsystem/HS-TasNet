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
    from research.direct.latency58_log_relative_checkpoint import load_parent
    from research.direct.latency58_gate_bias import with_update_gate_bias
    parent_binding = plan["parent_training_plan"]
    require(sha(parent_binding["path"]) == parent_binding["sha256"], "Parent loading plan changed")
    parent = load_parent(read(parent_binding["path"])).eval().requires_grad_(False)
    _MODEL = with_update_gate_bias(parent, offset=plan["offset"])
    fingerprint = model_state_sha256(_MODEL)
    require(fingerprint == plan["expected_model_state_sha256"]
            and model_state_sha256(parent) == plan["parent_model_state_sha256"], "Worker transformed different weights")
    _IDENTITY = {"label": plan["label"], "state_kind": "untrained_initialization", "training_updates": 0,
                 "checkpoint": None, "model_state_sha256": fingerprint,
                 "provenance": {"initialization": "unadapted_update_gate_bias_transform",
                                "training_updates_after_transform": 0,
                                "parent_cumulative_training_updates": parent.provenance["training_updates"],
                                "parameter_transform_provenance": _MODEL.provenance}}


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
    require(plan["schema"] == "latency58-gate-bias-parallel-music-plan-v1" and plan["workers"] == 2
            and plan["model_kind"] == "gate_bias_unadapted" and plan["offset"] == 0.6931471805599453,
            "Unsupported parallel evaluation")
    functional, screen = read(plan["functional_proof"]["path"]), read(plan["actions_screen"]["path"])
    for binding in (plan["functional_proof"], plan["actions_screen"], plan["functional_execution"],
                    plan["screen_execution"], plan["advance_decision"]):
        require(sha(binding["path"]) == binding["sha256"]
                and plan["source_bindings"].get(binding["path"]) == binding["sha256"], "Gate evidence binding differs")
    for binding in (plan["functional_execution"], plan["screen_execution"]):
        execution = read(binding["path"])
        require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "Gate evidence lacks a successful actual execution")
    require(read(plan["functional_execution"]["path"])["plan_sha256"] == functional["plan_sha256"]
            and read(plan["screen_execution"]["path"])["plan_sha256"] == screen["plan_sha256"]
            and read(plan["advance_decision"]["path"])["status"] == "evaluate_unadapted_gate_on_full14",
            "Gate evidence belongs to a different execution or decision")
    qualification = plan["parallel_qualification"]
    require(plan["source_bindings"].get(qualification["path"]) == qualification["sha256"] == sha(qualification["path"]),
            "Original parallel qualification binding differs")
    proof = read(qualification["path"])
    require(proof["status"] == "pass" and proof["exact_track_reports"] and proof["exact_stream_metadata"]
            and proof["exact_original_aggregate"] and proof["track_indices"] == list(range(6))
            and all(sha(p) == s for p, s in proof["source_bindings"].items()),
            "Original parallel evaluation qualification changed")
    require(functional["status"] == "pass" and functional["source_bindings_unchanged"]
            and screen["inputs_unchanged"] and functional["transformed_model_state_sha256"]
            == screen["results"][0]["model"]["model_state_sha256"] == plan["expected_model_state_sha256"],
            "Different gate transforms in the functional and excerpt results")
    indices = plan["track_indices"]
    require(indices == list(range(14)), "Require the complete primary development panel")
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
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - began,
                  checkpoint_written=False, audio_written=False, training_updates_executed=0,
                  quality_selected=False, screen_scope="Complete primary panel of the unadapted transform")
    write(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
