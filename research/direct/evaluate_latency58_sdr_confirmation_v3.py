"""Score reserved development intervals only after recorded primary selection.

This evaluator never chooses a checkpoint. Both the canonical working model
and a selected generation stream each complete track from sample zero through
the reserved intervals, using the original physical-time scorer and metrics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

_MODEL = None
_IDENTITY = None
_RESERVATION = None


def bound_read(binding):
    require(sha(binding["path"]) == binding["sha256"], "Confirmation prerequisite changed")
    return read(binding["path"])


def validate_selection(plan):
    """Authenticate the prospective reservation and completed primary selection."""
    reservation = bound_read(plan["reservation"])
    require(Path(plan["reservation"]["path"]) == PHASE / "sdr-teacher-prep-001/confirmation-plan.json"
            and plan["reservation"]["sha256"] == "4ea574181d51a650e7d1bb9d4cb321b558adf325caac28d3f84835dd9596bf3b"
            and reservation["schema"] == "latency58-sdr-confirmation-plan-v1"
            and reservation["excerpt_starts"] == [105.0, 135.0]
            and reservation["duration_seconds"] == 15.0
            and reservation["manifest"] == str(ROOT / "research/manifests/valid.json")
            and sha(reservation["manifest"]) == reservation["manifest_sha256"], "Reserved intervals differ")
    manifest = read(reservation["manifest"])
    require(reservation["track_names"] == [t["name"] for t in manifest["tracks"]], "Reserved tracks differ")
    selection = bound_read(plan["selection"])
    verify_inputs(selection)
    require(selection["schema"] == "latency58-sdr-primary-selection-v1"
            and selection["status"] == "selected_for_confirmation"
            and selection["confirmation_material_used_for_selection"] is False
            and selection["reservation"] == plan["reservation"], "Primary selection is absent or differs")
    primary_plan = bound_read(selection["primary_plan"])
    primary = bound_read(selection["primary_result"])
    execution = bound_read(selection["primary_execution"])
    summary = bound_read(selection["primary_summary"])
    summary_execution = bound_read(selection["summary_execution"])
    require(primary_plan["mode"] == "full14" and primary_plan["track_indices"] == list(range(14))
            and primary["track_names"] == reservation["track_names"] and primary["excerpt_count"] == 28
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and primary["inputs_unchanged"]
            and execution["plan_sha256"] == primary["plan_sha256"] == selection["primary_plan"]["sha256"]
            and primary["results"][0]["aggregate"]["full_sdr_db"] >= 4.057715948706591
            and primary["results"][0]["model"]["model_state_sha256"] == selection["model_state_sha256"]
            and primary_plan["checkpoint"] == selection["checkpoint"], "Selected primary evidence is incomplete")
    require([e["start_seconds"] for e in primary["excerpts"]] == [30.0, 75.0]
            and all(e["duration_seconds"] == 15.0 for e in primary["excerpts"])
            and summary_execution["actual_exit_code"] == 0 and not summary_execution["timed_out"]
            and summary_execution["source_bindings_unchanged"]
            and summary["model_state_sha256"] == selection["model_state_sha256"]
            and summary["versus_working_baseline"]["full14"]["metrics"]["full_sdr_db"]["candidate"]
            == primary["results"][0]["aggregate"]["full_sdr_db"]
            and selection["quality_and_probe_review_passed"] is True,
            "Selection lacks the completed quality review or used different primary intervals")
    require(plan["model_kind"] in ("working_baseline", "selected_generation"), "Unknown confirmation model")
    if plan["model_kind"] == "selected_generation":
        require(plan["expected_model_state_sha256"] == selection["model_state_sha256"]
                and plan["selected_primary_plan"] == selection["primary_plan"], "Confirmation candidate differs from selection")
    return reservation, selection


def initialize_worker(plan):
    global _MODEL, _IDENTITY, _RESERVATION
    import torch
    from research.direct.latency58_evaluate import model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    _RESERVATION = bound_read(plan["reservation"])
    if plan["model_kind"] == "working_baseline":
        from research.direct.latency58_sdr_teacher import load_initial_student, STUDENT_SHA256
        _MODEL = load_initial_student().eval().requires_grad_(False)
        checkpoint = {"kind": "inference", "path": str(PHASE / "teacher-half-canonical-001/model.pt"),
                      "sha256": STUDENT_SHA256}
    else:
        primary_plan = bound_read(plan["selected_primary_plan"])
        module_name = plan["evaluation_loader_module"]
        require(module_name.startswith("research.direct.evaluate_latency58_")
                and module_name.replace("_", "").replace(".", "").isalnum(), "Invalid selected loader module")
        source = ROOT / (module_name.replace(".", "/") + ".py")
        require(plan["source_bindings"].get(str(source)) == sha(source), "Selected loader is not bound")
        module = importlib.import_module(module_name)
        _MODEL, _ = module.load_evaluation_model(primary_plan)
        checkpoint = primary_plan["checkpoint"]
    _IDENTITY = {"label": plan["label"], "state_kind": "checkpoint", "checkpoint": checkpoint,
                 "training_updates": _MODEL.provenance["training_updates"],
                 "provenance": _MODEL.provenance, "model_state_sha256": model_state_sha256(_MODEL)}
    require(_IDENTITY["model_state_sha256"] == plan["expected_model_state_sha256"], "Worker loaded different weights")


def score_track(index):
    from research.direct.latency58_evaluate import evaluate_latency58_music
    require(_MODEL is not None and _IDENTITY is not None and _RESERVATION is not None, "Worker is uninitialized")
    report = evaluate_latency58_music(
        _MODEL, identity=_IDENTITY, track_indices=[index],
        excerpt_starts=_RESERVATION["excerpt_starts"], duration=_RESERVATION["duration_seconds"])
    report["parallel_worker_pid"] = os.getpid()
    return index, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Confirmation execution plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-confirmation-execution-plan-v1" and plan["workers"] == 2
            and plan["track_indices"] == list(range(14)), "Unsupported confirmation extent")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CUDA-hidden CPU1 workers")
    verify_inputs(plan)
    reservation, selection = validate_selection(plan)
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 2_000_000)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve confirmation evidence")
    from research.direct.evaluate_latency58_sdr_parallel import combine_reports

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
    validate_selection(plan)
    result.update(plan_sha256=args.plan_sha256, root_source_bindings=plan["source_bindings"],
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - began,
                  panel="reserved-development-confirmation", reservation=plan["reservation"],
                  primary_selection=plan["selection"], selected_model_state_sha256=selection["model_state_sha256"],
                  confirmation_material_used_for_selection=False,
                  confirmation_limitation=reservation["limitation"])
    write(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
