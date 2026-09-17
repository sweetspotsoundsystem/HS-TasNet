"""Evaluate reserved development intervals after a primary-panel selection.

Reuse the validated CPU1-per-track executor and unchanged scorer. This tool
requires a frozen selection and the original confirmation-interval contract;
it cannot choose checkpoints or change the primary-panel scoring protocol.
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
_CONFIRMATION = None


def initialize_worker(plan):
    global _MODEL, _IDENTITY, _CONFIRMATION
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    if plan["model_kind"] == "working_baseline":
        from research.direct.latency58_sdr_teacher import load_initial_student, STUDENT_SHA256
        _MODEL = load_initial_student().eval().requires_grad_(False)
        checkpoint = {"kind": "inference", "path": str(PHASE / "teacher-half-canonical-001/model.pt"),
                      "sha256": STUDENT_SHA256}
    else:
        if plan["model_kind"] == "context_candidate":
            from research.direct.evaluate_latency58_context import load_evaluation_model
        else:
            from research.direct.evaluate_latency58_sdr import load_evaluation_model
        _MODEL, _ = load_evaluation_model(plan)
        checkpoint = plan["checkpoint"]
    _CONFIRMATION = read(plan["confirmation"]["path"])
    _IDENTITY = {"label": plan["label"], "state_kind": "checkpoint", "checkpoint": checkpoint,
                 "training_updates": _MODEL.provenance["training_updates"],
                 "provenance": _MODEL.provenance, "model_state_sha256": model_state_sha256(_MODEL)}
    require(_IDENTITY["model_state_sha256"] == plan["expected_model_state_sha256"], "Worker loaded different weights")


def score_track(index):
    from research.direct.latency58_evaluate import evaluate_latency58_music
    require(_MODEL is not None and _IDENTITY is not None, "Worker was not initialized")
    require(_CONFIRMATION is not None, "Worker lacks confirmation intervals")
    report = evaluate_latency58_music(
        _MODEL, identity=_IDENTITY, track_indices=[index],
        manifest_path=Path(_CONFIRMATION["manifest"]),
        excerpt_starts=_CONFIRMATION["excerpt_starts"], duration=_CONFIRMATION["duration_seconds"],
    )
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
    require(plan["schema"] == "latency58-sdr-confirmation-execution-plan-v1" and plan["workers"] == 2
            and plan["model_kind"] in ("working_baseline", "sdr_candidate", "context_candidate"),
            "Unsupported confirmation evaluation")
    bindings = plan["source_bindings"]
    for item in (plan["confirmation"], plan["selection"]):
        require(bindings.get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Confirmation contract or selection changed")
    confirmation = read(plan["confirmation"]["path"])
    selection = read(plan["selection"]["path"])
    require(confirmation["schema"] == "latency58-sdr-confirmation-plan-v1"
            and confirmation["excerpt_starts"] == [105.0, 135.0]
            and confirmation["duration_seconds"] == 15.0
            and confirmation["manifest_sha256"] == sha(confirmation["manifest"])
            and confirmation["track_names"] == [t["name"] for t in read(confirmation["manifest"])["tracks"]],
            "Reserved confirmation protocol differs")
    require(selection["schema"] == "latency58-sdr-primary-selection-v1"
            and selection["status"] == "selected_for_confirmation"
            and selection["confirmation"] == plan["confirmation"]
            and selection["confirmation_not_yet_evaluated"]
            and all(sha(p) == s for p, s in selection["source_bindings"].items()),
            "Require a frozen primary-panel selection before confirmation")
    if plan["model_kind"] != "working_baseline":
        require(selection["model_kind"] == plan["model_kind"]
                and selection["model_state_sha256"] == plan["expected_model_state_sha256"]
                and selection["checkpoint"] == plan["checkpoint"]
                and selection["generation"] == plan["generation"], "Candidate differs from the frozen selection")
    from research.direct.report_latency58_sdr import load_completed
    evidence_bindings = {}
    _, primary = load_completed(PHASE / (selection["candidate_prefix"] + "-full14-001"), evidence_bindings)
    require(primary["results"][0]["model"]["model_state_sha256"] == selection["model_state_sha256"]
            and len(primary["results"][0]["tracks"]) == 14
            and all(selection["source_bindings"].get(p) == s for p, s in evidence_bindings.items()),
            "Selection does not bind its completed primary panel")
    indices = plan["track_indices"]
    require(indices == list(range(14)), "Confirmation requires the complete reserved panel")
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
                  panel="reserved-within-track-development-confirmation", selection=plan["selection"],
                  confirmation=plan["confirmation"], checkpoint_selection_allowed=False,
                  confirmation_limitation=confirmation["limitation"])
    write(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
