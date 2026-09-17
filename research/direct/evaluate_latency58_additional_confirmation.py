"""Score one frozen choice or comparator on the additional reserved windows.

The model initializer, per-track streamer and report reducer are reused without
changes. Selection and reservation consumption are authenticated before any
worker can load a model. This module never chooses a checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs
from research.direct.latency58_additional_confirmation_gate import (
    ACCEPTED_STATE, record_first_use, validate_selection as authenticate_selection,
)
from research.direct.evaluate_latency58_sdr_confirmation_v5 import initialize_worker, score_track
from research.direct.evaluate_latency58_sdr_parallel import combine_reports
from research.direct.latency58_sdr_teacher import STUDENT_STATE_SHA256

SCHEMA = "latency58-additional-confirmation-execution-plan-v1"
PANEL = "additional-within-track-development-confirmation"
ROOT_OUTPUT = PHASE / "additional-confirmation-execution-001"
ROLE_KINDS = {"candidate": "selected_generation", "accepted": "accepted_reference", "working": "working_baseline"}


def expected_directories(chosen):
    candidate = str(PHASE / "additional-confirmation-candidate-001")
    return {"candidate": candidate, "working": str(PHASE / "additional-confirmation-working-001"),
            "accepted": candidate if chosen["model_state_sha256"] == ACCEPTED_STATE
            else str(PHASE / "additional-confirmation-accepted-001")}


def require_capacity(plan, extra_bytes):
    """Count the same four roots, external model objects and reserved reports."""
    require(type(extra_bytes) is int and extra_bytes >= 0, "Invalid space reservation")
    counted = sum(disk_bytes(Path(root)) for root in plan["counted_roots"])
    outside = 146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects")
    reports = sum(disk_bytes(Path(path)) for path in {*plan["evaluation_directories"].values(), str(ROOT_OUTPUT)})
    require(outside < 500_000_000 and counted + extra_bytes < 79_500_000_000
            and counted + outside + extra_bytes < 80_000_000_000,
            "Confirmation exceeds the combined artifact reservation")
    require(reports + extra_bytes <= 12_000_000, "Confirmation exceeds the reserved report budget")
    return {"counted_bytes": counted, "outside_bytes": outside, "report_bytes": reports}


def validate_plan(plan):
    require(plan["schema"] == SCHEMA and plan["workers"] == 2
            and plan["track_indices"] == list(range(14)), "Unsupported additional confirmation extent")
    reservation, selection, chosen, reviewed, evidence = authenticate_selection(plan["selection"], plan["reservation"])
    require(all(plan["source_bindings"].get(path) == digest for path, digest in evidence.items()),
            "Confirmation omitted authenticated selection evidence")
    require(plan["counted_roots"] == reservation["counted_roots"]
            and plan["stop_counted_bytes"] == reservation["counted_stop_bytes"] == 79_500_000_000
            and plan["maximum_confirmation_report_bytes"] == reservation["maximum_confirmation_report_bytes"] == 12_000_000
            and plan["evaluation_directories"] == expected_directories(chosen), "Confirmation artifact scope differs")
    role = plan["role"]
    require(role in ROLE_KINDS and plan["model_kind"] == ROLE_KINDS[role]
            and plan["output_directory"] == plan["evaluation_directories"][role]
            and not (role == "accepted" and chosen["model_state_sha256"] == ACCEPTED_STATE),
            "Unknown, duplicate or misplaced comparator")
    if role == "working":
        require(plan["expected_model_state_sha256"] == STUDENT_STATE_SHA256
                and "selected_primary_plan" not in plan and "evaluation_loader_module" not in plan,
                "Working comparator differs from the original model")
    else:
        expected = chosen if role == "candidate" else reviewed["leader-cleanup-250"]
        require(plan["expected_model_state_sha256"] == expected["model_state_sha256"]
                and plan["selected_primary_plan"] == expected["primary_plan"]
                and plan["evaluation_loader_module"] == expected["evaluation_loader_module"],
                "Selected checkpoint or accepted comparator differs")
    require(plan["label"] == "additional-confirmation-" + role, "Confirmation label differs")
    return reservation, selection, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Confirmation execution plan changed")
    plan = read(args.plan)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CUDA-hidden CPU1 workers")
    verify_inputs(plan)
    reservation, selection, chosen = validate_plan(plan)
    require_capacity(plan, 2_000_000)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and args.plan.resolve() == out / "plan.json"
            and not (out / "result.json").exists() and not (out / "progress.jsonl").exists(),
            "Preserve confirmation evidence")
    first_use = record_first_use(plan["selection"], plan["reservation"])

    began = time.monotonic()
    reports = {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(score_track, index): index for index in plan["track_indices"]}
        for future in as_completed(futures):
            index, report = future.result()
            require(index == futures[future] and index not in reports, "Worker returned a different track")
            reports[index] = report
            row = {"event": "track", "index": index, "track": report["track_names"][0]}
            progress.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    require(set(reports) == set(range(14)), "Incomplete confirmation coverage")
    result = combine_reports(reports, indices=plan["track_indices"], elapsed=time.monotonic() - began)
    require(result["excerpt_count"] == 28 and result["track_names"] == reservation["track_names"],
            "Confirmation coverage differs from reservation")
    verify_inputs(plan)
    validate_plan(plan)
    require(sha(first_use["path"]) == first_use["sha256"], "Confirmation consumption record changed")
    result.update(plan_sha256=args.plan_sha256, root_source_bindings=plan["source_bindings"],
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - began,
                  panel=PANEL, reservation=plan["reservation"], primary_selection=plan["selection"],
                  selected_model_state_sha256=selection["model_state_sha256"], first_use=first_use,
                  confirmation_material_used_for_selection=False,
                  confirmation_limitations=reservation["limitations"])
    require_capacity(plan, len((json.dumps(result, indent=2, allow_nan=False) + "\n").encode()))
    write(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
