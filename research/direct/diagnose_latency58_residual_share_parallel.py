"""Score the fixed raw4 residual correction with exact ordinary-score replay.

Each CPU1 worker streams an unchanged warm500 model once per full track.
Both output policies share the same captures and original metric functions.
All original per-track reports and the aggregate must reproduce exactly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write

_MODEL = None
_PLAN = None


def initialize_worker(plan):
    global _MODEL, _PLAN
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_log_relative_checkpoint import load_parent
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _PLAN = plan
    binding = plan["parent_training_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Parent loading plan changed")
    _MODEL = load_parent(read(binding["path"])).eval().requires_grad_(False)
    require(model_state_sha256(_MODEL) == plan["parent_model_state_sha256"], "Worker loaded different weights")


def score_track(index):
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import (
        HOP, SOURCE_ORDER, model_state_sha256, plan_latency58_stream, stream_latency58_track,
    )
    from research.direct.latency58_residual_share import PRIMARY_SHARE, residual_share
    from research.metrics import MetricConfig
    require(_MODEL is not None and _PLAN is not None, "Worker not initialized")
    fingerprint = model_state_sha256(_MODEL)
    rng = torch.get_rng_state().clone()
    manifest, source_config = read(_PLAN["manifest"]), read(_PLAN["evaluation_config"])
    tracks, config = shared.select_panel(manifest, source_config, panel="full", track_indices=[index],
                                        excerpt_starts=None, duration=15.0, alignment_samples=HOP)
    require(len(tracks) == 1, "One complete track per worker task")
    track = tracks[0]
    rows = legacy._reference_intervals(track, config)
    require(len(rows) == 2, "Original panel has two excerpts per track")
    root = Path(manifest["root"])
    streaming = plan_latency58_stream(rows, int(track["frames"]))
    raw_outputs, delayed_mixtures, metadata = stream_latency58_track(
        _MODEL, legacy._safe_dataset_path(root, track["mixture"]), streaming)

    def excerpt(relative, row):
        return legacy._read_excerpt(legacy._safe_dataset_path(root, relative),
                                    int(row["reference_start"]), int(row["reference_end"]),
                                    expected_frames=int(track["frames"]))

    mixtures = [excerpt(track["mixture"], row) for row in rows]
    references = [np.stack([excerpt(track["stems"][stem], row) for stem in SOURCE_ORDER]) for row in rows]
    require(all(np.array_equal(delayed, physical.astype(np.float32)) for delayed, physical in
                zip(delayed_mixtures, mixtures, strict=True)), "Physical mixture alignment differs")
    scores, closure = {}, 0.0
    for name, share in (("working_policy", 0), ("fixed_share", PRIMARY_SHARE)):
        estimates = [residual_share(raw, physical.astype(np.float32), share=share)
                     for raw, physical in zip(raw_outputs, mixtures, strict=True)]
        scores[name] = legacy._score_track(track["name"], rows, mixtures, references, estimates,
                                          MetricConfig.from_mapping(config["metrics"]))
        closure = max(closure, max(float(np.max(np.abs(estimate.sum(axis=0, dtype=np.float32) - physical)))
                                   for estimate, physical in zip(estimates, mixtures, strict=True)))
    stored = read(_PLAN["parent_full14_result"]["path"])["results"][0]
    require(stored["tracks"][index] == scores["working_policy"], "Ordinary per-track score failed exact replay")
    require(model_state_sha256(_MODEL) == fingerprint == _PLAN["parent_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Worker model, RNG or CPU scope changed")
    return index, {"scores": scores, "stream_metadata": metadata, "closure_max_abs": closure,
                   "exact_stored_track_report": True, "worker_pid": os.getpid()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Full-panel correction plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-residual-share-parallel-diagnostic-v1"
            and plan["workers"] == 2 and plan["primary_share"] == 1 / 16
            and plan["track_indices"] == list(range(14))
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Require the frozen fixed-share CPU full-panel diagnostic")
    for key in ("screen_result", "screen_execution", "advance_decision", "parent_full14_result"):
        binding = plan[key]
        require(sha(binding["path"]) == binding["sha256"]
                and plan["source_bindings"].get(binding["path"]) == binding["sha256"], "Evidence binding differs")
    screen, execution, decision = (read(plan[key]["path"]) for key in
                                   ("screen_result", "screen_execution", "advance_decision"))
    require(screen["status"] == "pass" and screen["zero_share_exact_stored_track_report"]
            and screen["source_bindings_unchanged"] and execution["actual_exit_code"] == 0
            and not execution["timed_out"] and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == screen["plan_sha256"]
            and decision["status"] == "evaluate_fixed_share_on_full14"
            and all(decision["screen_rules_passed"].values())
            and screen["parent_model_state_sha256"] == plan["parent_model_state_sha256"],
            "Fixed correction lacks a completed prerequisite or advance decision")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous diagnostic")
    began, reports = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(score_track, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, report = future.result()
            require(index == futures[future] and index not in reports, "Wrong or repeated track")
            reports[index] = report
            row = {"event": "track", "index": index,
                   "full_sdr_delta": report["scores"]["fixed_share"]["full_sdr_db"]
                                     - report["scores"]["working_policy"]["full_sdr_db"]}
            progress.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    require(set(reports) == set(plan["track_indices"]), "Incomplete primary panel")
    from research import evaluate as legacy
    from research.direct.compare import compare
    policies = {name: {"tracks": [reports[i]["scores"][name] for i in plan["track_indices"]],
                       "checkpoint": None, "output_policy": name}
                for name in ("working_policy", "fixed_share")}
    for policy in policies.values():
        policy["aggregate"] = legacy._aggregate_tracks(policy["tracks"])
    stored = read(plan["parent_full14_result"]["path"])["results"][0]
    require(stored["aggregate"] == policies["working_policy"]["aggregate"]
            and stored["tracks"] == policies["working_policy"]["tracks"]
            and stored["model"]["model_state_sha256"] == plan["parent_model_state_sha256"],
            "Original full-panel aggregate or track reports failed exact replay")
    comparison = compare(policies["working_policy"], policies["fixed_share"])
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Diagnostic input changed")
    result = {"schema": "latency58-residual-share-full14-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "parent_model_state_sha256": plan["parent_model_state_sha256"],
              "primary_share": 1 / 16, "policies": policies, "comparison": comparison,
              "reports": [reports[i] for i in plan["track_indices"]],
              "exact_stored_track_reports_and_aggregate": True, "model_unchanged": True,
              "cuda_initialized": False, "checkpoint_written": False, "audio_written": False,
              "training_updates_executed": 0, "quality_selected": False,
              "elapsed_seconds": time.monotonic() - began,
              "limitations": ["Previously used primary development panel; confirmation excerpts remain unused.",
                              "Fixed output mixing changes; all neural weights and native gains are unchanged.",
                              "Track intervals omit checkpoint selection and training-seed uncertainty.",
                              "No listening, ONNX, native runtime or host-latency qualification."]}
    write(out / "result.json", result)
    print(json.dumps({"event": "complete", "metrics": comparison["metrics"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
