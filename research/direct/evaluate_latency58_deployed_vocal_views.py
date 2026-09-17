"""Measure supplementary vocal views for the exact released ONNX graph."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs

_SESSION = _PLAN = None


def initialize_worker(plan):
    global _SESSION, _PLAN
    from research.direct.run_latency58_deployed_vocal_views import require_cpu
    from research.direct.latency58_attention_int8_verify import session_for
    require_cpu(plan)
    data = Path(plan["checkpoint"]["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == plan["checkpoint"]["sha256"]
            and len(data) == plan["checkpoint"]["bytes"], "Worker graph bytes changed")
    _SESSION, _PLAN = session_for(data, plan["interface"]), plan


def score_track(index):
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import plan_latency58_stream
    from research.direct.latency58_onnx_vocal_views import stream_onnx_views
    from research.direct.latency58_vocal_views import score_views
    from research.metrics import MetricConfig, SOURCE_ORDER
    manifest, config = read(_PLAN["manifest"]["path"]), read(_PLAN["config"]["path"])
    tracks, config = shared.select_panel(manifest, config, panel="full", track_indices=[index],
        excerpt_starts=None, duration=15., alignment_samples=128)
    require(len(tracks) == 1, "Expected exactly one validation track")
    track = tracks[0]
    intervals = legacy._reference_intervals(track, config)
    require(intervals == _PLAN["track_intervals"][str(index)]["intervals"]
            and track["name"] == _PLAN["track_intervals"][str(index)]["name"], "Physical panel changed")
    paths = [legacy._safe_dataset_path(Path(manifest["root"]), track["stems"][s]) for s in SOURCE_ORDER]
    require(all(sha(p) == _PLAN["source_bindings"][str(p)] for p in paths), "Source audio changed")
    began = time.monotonic()
    stream_plan = plan_latency58_stream(intervals, int(track["frames"]), unroll_hops=1, io_block_hops=64)
    heartbeat = Path(_PLAN["output_directory"]) / ("track-%02d-progress.jsonl" % index)
    with heartbeat.open("x", buffering=1) as log:
        def progress(calls, total):
            log.write(json.dumps({"calls_per_view": calls, "total_calls_per_view": total,
                "elapsed_seconds": time.monotonic() - began, "worker_pid": os.getpid()}) + "\n")
        refs, outputs, mixtures, metadata = stream_onnx_views(_SESSION, _PLAN["interface"], paths,
                                                            stream_plan, progress=progress)
    for captured, row in zip(refs, intervals, strict=True):
        independent = np.stack([legacy._read_excerpt(p, row["reference_start"], row["reference_end"],
            expected_frames=int(track["frames"])) for p in paths]).astype(np.float32)
        require(np.array_equal(captured, independent), "Independent physical source read differs")
    scores = score_views(track["name"], intervals, refs, outputs, mixtures,
                         MetricConfig.from_mapping(config["metrics"]))
    require(not torch.cuda.is_initialized()
            and all(sha(p) == _PLAN["source_bindings"][str(p)] for p in paths), "CPU scope or source bytes changed")
    return index, {"name": track["name"], "index": index, "intervals": intervals, "views": scores,
        "stream": metadata, "elapsed_seconds": time.monotonic() - began, "worker_pid": os.getpid(),
        "source_audio_unchanged": True, "independent_physical_references_verified": True}


def aggregate_reports(reports):
    from research.direct.latency58_vocal_views import VIEWS
    from research.metrics import SOURCE_ORDER, mean_or_none
    aggregate = {}
    for view in VIEWS:
        aggregate[view] = {"tracks": len(reports), "input_active_windows": sum(
            r["views"][view]["input_active_windows"] for r in reports), "per_stem": {}}
        for stem in SOURCE_ORDER:
            levels = [r["views"][view]["native_output_levels"][stem] for r in reports]
            scores = [r["views"][view]["standard_scores_on_remixed_references"]["per_stem"][stem] for r in reports]
            aggregate[view]["per_stem"][stem] = {"off_target": levels[0]["off_target"],
                **{field: mean_or_none(r[field] for r in levels) for field in
                    ("output_rms_dbfs", "output_to_input_db", "signed_desired_projection_gain")},
                "desired_full_sdr_db": mean_or_none(r["full_sdr_db"] for r in scores),
                "desired_low_sdr_db": mean_or_none(r["band_sdr_db"]["low_20_250"] for r in scores)}
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--qualification-execution-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Evaluation plan changed")
    plan = read(args.plan)
    from research.direct.run_latency58_deployed_vocal_views import require_cpu, budget_snapshot
    require_cpu(plan)
    require(plan["schema"] == "latency58-deployed-vocal-views-plan-v1" and plan["workers"] == 2
            and plan["track_indices"] == list(range(14)) and not plan["audio_export"], "Evaluation scope changed")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    qpath, epath = out / "qualification/result.json", out / "qualification/execution.json"
    require(sha(qpath) == args.qualification_sha256 and sha(epath) == args.qualification_execution_sha256,
            "Transport qualification changed")
    q, e = read(qpath), read(epath)
    require(q["status"] == "pass" and q["source_bindings_unchanged"]
            and q["plan_sha256"] == args.plan_sha256 and q["graph_sha256"] == plan["checkpoint"]["sha256"]
            and e["actual_exit_code"] == 0 and not e["timed_out"] and e["source_bindings_unchanged"],
            "Transport qualification did not finish successfully")
    before, began, reports = budget_snapshot(plan), time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as log, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker, initargs=(plan,)) as pool:
        futures = {pool.submit(score_track, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, row = future.result()
            require(index == futures[future] and index not in reports, "Wrong or repeated track")
            reports[index] = row
            # Keep completed track measurements if a later worker fails.
            write(out / ("track-%02d.json" % index), row)
            summary = {"index": index, "track": row["name"], "elapsed_seconds": time.monotonic() - began,
                "instrumental_vocals": row["views"]["instrumental"]["native_output_levels"]["vocals"],
                "vocals_only_vocals": row["views"]["vocals_only"]["native_output_levels"]["vocals"]}
            log.write(json.dumps(summary, allow_nan=False) + "\n")
            print(json.dumps(summary, allow_nan=False), flush=True)
    require(set(reports) == set(plan["track_indices"]), "Incomplete fixed panel")
    ordered = [reports[i] for i in plan["track_indices"]]
    aggregate = aggregate_reports(ordered)
    windows = [{"track": r["name"], "excerpt_index": w["excerpt_index"],
        "physical_start": w["physical_start"], "physical_end": w["physical_end"],
        "input_rms_dbfs": w["input_rms_dbfs"], **w["per_stem"]["vocals"]}
        for r in ordered for w in r["views"]["instrumental"]["windows"] if w["input_active"]]
    verify_inputs(plan)
    result = {"schema": "latency58-deployed-vocal-views-result-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "checkpoint": plan["checkpoint"], "release": plan["release"],
        "runtime": plan["runtime"], "track_count": 14, "excerpt_count_per_view": 28,
        "tracks": ordered, "aggregate": aggregate,
        "worst_instrumental_vocal_windows_by_dbfs": sorted(windows, key=lambda w: w["output_rms_dbfs"], reverse=True)[:20],
        "worst_instrumental_vocal_windows_relative_to_mix": sorted(windows, key=lambda w: w["output_to_input_db"], reverse=True)[:20],
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "qualification_sha256": args.qualification_sha256,
        "qualification_execution_sha256": args.qualification_execution_sha256,
        "budget_before": before, "budget_after": budget_snapshot(plan),
        "elapsed_seconds": time.monotonic() - began, "gpu_used": False, "audio_exported": False,
        "training_updates_executed": 0, "quality_selected": False, "host_qualified": False,
        "limitations": ["Supplementary remixes of development sources, not the original full-mixture score or an unseen test.",
            "Source removal follows dataset assignments; recordings may contain bleed.",
            "Levels use input-active one-second windows, mean dB within track and equal track weighting.",
            "Desired SDR and signed gain accompany leakage; muting is not successful separation.",
            "This baseline does not establish human listening acceptance or M4 timing."]}
    encoded_size = len(json.dumps(result, indent=2, allow_nan=False).encode())
    existing_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    require(existing_size + encoded_size + 500_000 < plan["diagnostic_artifact_allowance_bytes"],
            "Diagnostic artifact allowance exceeded")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
