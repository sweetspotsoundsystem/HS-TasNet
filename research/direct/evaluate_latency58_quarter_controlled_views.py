"""Score the same fixed vocal quarter_controlleds for a completed quarter_controlled pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_sdr_checkpoint import require_space

_MODEL = _PLAN = None


def initialize_worker(plan):
    global _MODEL, _PLAN
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260920)
    torch.use_deterministic_algorithms(True)
    spec = plan["model"]
    require(spec["kind"] == "quarter_controlled", "Require the new quarter_controlled checkpoint family")
    binding = spec["quality_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Diagnostic quality plan changed")
    from research.direct.evaluate_latency58_quarter_controlled import load_evaluation_model
    model, _ = load_evaluation_model(read(binding["path"]))
    _MODEL = model.eval().requires_grad_(False)
    _PLAN = plan
    require(state_sha256(_MODEL.state_dict()) == spec["model_state_sha256"] and not torch.cuda.is_initialized(),
            "Diagnostic worker loaded the wrong model or initialized CUDA")


def score_track(index):
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import plan_latency58_stream
    from research.direct.latency58_vocal_views import stream_views, score_views
    from research.metrics import MetricConfig, SOURCE_ORDER
    require(_MODEL is not None and _PLAN is not None, "Diagnostic worker not initialized")
    manifest, config = read(_PLAN["manifest"]["path"]), read(_PLAN["config"]["path"])
    tracks, config = shared.select_panel(manifest, config, panel="full", track_indices=[index],
        excerpt_starts=None, duration=15.0, alignment_samples=128)
    require(len(tracks) == 1, "Expected one complete development track")
    track = tracks[0]
    intervals = legacy._reference_intervals(track, config)
    require(intervals == _PLAN["track_intervals"][str(index)]["intervals"]
            and track["name"] == _PLAN["track_intervals"][str(index)]["name"],
            "Counterfactual physical intervals changed")
    root = Path(manifest["root"])
    paths = [legacy._safe_dataset_path(root, track["stems"][stem]) for stem in SOURCE_ORDER]
    require(all(_PLAN["source_bindings"].get(str(p)) == sha(p) for p in paths), "Original source bytes changed")
    stream_plan = plan_latency58_stream(intervals, int(track["frames"]), unroll_hops=64, io_block_hops=64)
    fingerprint, rng = state_sha256(_MODEL.state_dict()), torch.get_rng_state().clone()
    began = time.monotonic()
    refs, outputs, mixtures, metadata = stream_views(_MODEL, paths, stream_plan)
    # Read independently at the physical reference positions to prove source
    # capture, rather than using the remixed stream to establish its own oracle.
    for captured, interval in zip(refs, intervals, strict=True):
        independently_read = np.stack([legacy._read_excerpt(p, int(interval["reference_start"]),
            int(interval["reference_end"]), expected_frames=int(track["frames"])) for p in paths]).astype(np.float32)
        require(np.array_equal(captured, independently_read), "Source capture differs from independent physical reads")
    scores = score_views(track["name"], intervals, refs, outputs, mixtures, MetricConfig.from_mapping(config["metrics"]))
    require(state_sha256(_MODEL.state_dict()) == fingerprint == _PLAN["model"]["model_state_sha256"]
            and torch.equal(torch.get_rng_state(), rng) and not torch.cuda.is_initialized()
            and all(_PLAN["source_bindings"][str(p)] == sha(p) for p in paths),
            "Diagnostic changed model, RNG, source audio or CPU scope")
    return index, {"name": track["name"], "index": index, "intervals": intervals,
                   "views": scores, "stream": metadata, "elapsed_seconds": time.monotonic() - began,
                   "worker_pid": os.getpid(), "source_audio_unchanged": True,
                   "model_state_sha256": fingerprint, "cpu_rng_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Vocal quarter_controlled plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-quarter-controlled-views-evaluation-plan-v1" and plan["workers"] == 2
            and plan["track_indices"] == list(range(14)) and plan["audio_export"] is False
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require the complete panel with two CUDA-hidden CPU1 workers and metrics-only output")
    verify_inputs(plan)
    for key in ("manifest", "config", "qualification", "qualification_execution"):
        binding = plan[key]
        require(sha(binding["path"]) == binding["sha256"], "Diagnostic prerequisite changed: " + key)
    qualification = read(plan["qualification"]["path"])
    execution = read(plan["qualification_execution"]["path"])
    require(qualification["status"] == "pass" and qualification["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Source-view implementation is not qualified")
    reservation = plan["reservation"]
    require(sha(reservation["path"]) == reservation["sha256"], "Combined reservation changed")
    allocation = read(reservation["path"])
    require(allocation["new_artifact_allowance_bytes"] == 10_000_000
            and allocation["training_reserve_bytes"] == 350_000_000
            and allocation["counted_bytes_at_preparation"] + 360_000_000 < plan["stop_counted_bytes"]
            and plan["output_directory"] in allocation["evaluation_directories"],
            "Diagnostic lacks a reservation separate from active pilot completion")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve diagnostic result")
    # A concurrent pilot can consume its checkpoint reservation while this
    # read-only diagnostic runs. Do not reserve that same space twice.
    before = require_space(plan, 5_000_000)
    began, reports = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(plan,)) as pool:
        futures = {pool.submit(score_track, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, row = future.result()
            require(index == futures[future] and index not in reports, "Diagnostic track index differs")
            reports[index] = row
            summary = {"event": "track", "index": index, "track": row["name"],
                       "instrumental_vocal_output_dbfs": row["views"]["instrumental"]["native_output_levels"]["vocals"]["output_rms_dbfs"]}
            progress.write(json.dumps(summary, allow_nan=False) + "\n")
            print(summary, flush=True)
    require(set(reports) == set(plan["track_indices"]), "Incomplete vocal diagnostic panel")
    from research.metrics import SOURCE_ORDER, mean_or_none
    from research.direct.latency58_vocal_views import VIEWS, VERSION
    aggregate = {}
    for view in VIEWS:
        aggregate[view] = {"tracks": 14, "input_active_windows": sum(
            r["views"][view]["input_active_windows"] for r in reports.values()), "per_stem": {}}
        for stem in SOURCE_ORDER:
            levels = [r["views"][view]["native_output_levels"][stem] for r in reports.values()]
            scores = [r["views"][view]["standard_scores_on_remixed_references"]["per_stem"][stem] for r in reports.values()]
            aggregate[view]["per_stem"][stem] = {
                "off_target": levels[0]["off_target"],
                **{field: mean_or_none(r[field] for r in levels) for field in
                   ("output_rms_dbfs", "output_to_input_db", "signed_desired_projection_gain")},
                "desired_full_sdr_db": mean_or_none(r["full_sdr_db"] for r in scores),
                "desired_low_sdr_db": mean_or_none(r["band_sdr_db"]["low_20_250"] for r in scores)}
    verify_inputs(plan)
    result = {"schema": "latency58-vocal-views-evaluation-result-v1", "status": "pass", "version": VERSION,
              "plan_sha256": args.plan_sha256, "model": plan["model"], "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "tracks": [reports[i] for i in plan["track_indices"]],
              "aggregate": aggregate, "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
              "combined_reservation": reservation,
              "cuda_initialized": False, "training_updates_executed": 0,
              "audio_exported": False, "quality_selected": False, "confirmation_material_used": False,
              "limitations": ["Counterfactual remixes of existing development sources; not an unseen test or the original full-mixture score.",
                              "Natural stem recordings may already contain bleed; source removal is defined by dataset stem assignment.",
                              "Source-isolated behavior does not establish spill under a full mix; retain original metrics and listening.",
                              "Leakage uses input-active one-second windows, mean dB within track and equal track weighting.",
                              "Desired-source SDR, low-band SDR and signed gain accompany leakage; zero output is not successful separation.",
                              "No human listening or M4 runtime conclusion follows from this diagnostic."]}
    current_bytes = sum(p.stat().st_size for directory in allocation["evaluation_directories"]
                        for p in Path(directory).rglob("*") if p.is_file())
    require(current_bytes + len(json.dumps(result, indent=2, allow_nan=False).encode("utf-8")) + 1_000_000
            < allocation["new_artifact_allowance_bytes"], "Combined diagnostic artifact allowance exceeded")
    write(out / "result.json", result)
    print({"status": "pass", "aggregate": aggregate}, flush=True)


if __name__ == "__main__":
    main()
