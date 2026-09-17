"""Compare completed grouped-vocal endpoints with their retained training parent.

This report reuses completed full14 and continuous source-view measurements.
It performs no inference, training, checkpoint selection or plugin replacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

TRAINING = PHASE / "branch-grouped-vocal-012"
PARENT_VIEWS = PHASE / "paired-vocal-long-context-006"
CANDIDATE_VIEWS = PHASE / "paired-vocal-grouped-012"


def window_summary(windows):
    instrumental = [row for row in windows["all_windows"]
                    if row["view"] == "instrumental" and row["input_active"]]
    vocals = [row for row in windows["all_windows"]
              if row["view"] == "vocals_only" and row["input_active"]
              and row["per_stem"]["vocals"]["desired_active"]]
    quiet = [row for row in vocals if row["input_rms_dbfs"] < -35]

    def metric(row, name, side):
        return row["per_stem"]["vocals"]["metrics"][name][side]

    def compact(row):
        return {key: row[key] for key in
                ("track", "physical_start", "physical_end", "input_rms_dbfs")} | {
                    "vocal_metrics": row["per_stem"]["vocals"]["metrics"]}

    require(len(instrumental) == 420 and len(vocals) == 319 and len(quiet) == 15,
            "Established instrumental or desired-vocal support changed")
    return {"instrumental_active_windows": len(instrumental),
        "instrumental_vocal_output_increased_windows": sum(
            metric(row, "output_rms_dbfs", "delta") > 0 for row in instrumental),
        "instrumental_vocals_within_10_db_of_input": {
            side: sum(metric(row, "output_to_input_db", side) >= -10 for row in instrumental)
            for side in ("reference", "candidate")},
        "worst_candidate_relative_instrumental_windows": [compact(row) for row in sorted(
            instrumental, key=lambda row: metric(row, "output_to_input_db", "candidate"), reverse=True)[:10]],
        "largest_instrumental_vocal_increases": [compact(row) for row in sorted(
            instrumental, key=lambda row: metric(row, "output_rms_dbfs", "delta"), reverse=True)[:10]],
        "quiet_vocal_input_threshold_dbfs": -35, "quiet_vocal_active_windows": len(quiet),
        "quiet_vocal_mean_signed_gain": {
            side: mean(metric(row, "signed_desired_projection_gain", side) for row in quiet)
            for side in ("reference", "candidate")},
        "quiet_vocal_gain_decreased_windows": sum(
            metric(row, "signed_desired_projection_gain", "delta") < 0 for row in quiet),
        "worst_candidate_quiet_vocal_gain": [compact(row) for row in sorted(
            quiet, key=lambda row: metric(row, "signed_desired_projection_gain", "candidate"))[:10]],
        "worst_candidate_active_vocal_gain": [compact(row) for row in sorted(
            vocals, key=lambda row: metric(row, "signed_desired_projection_gain", "candidate"))[:10]]}


def music_summary(cells):
    result = {}
    for metric in next(iter(next(iter(cells.values())).values()))["metrics"]:
        rows = [{"track": track, "stem": stem, **cell["metrics"][metric]}
                for track, stems in cells.items() for stem, cell in stems.items()
                if cell["metrics"][metric]["delta"] is not None]
        lower_is_better = metric == "absent_fp_dbfs"
        regressions = [row for row in rows if (row["delta"] > 0 if lower_is_better else row["delta"] < 0)]
        result[metric] = {"eligible_cells": len(rows), "regressed_cells": len(regressions),
                          "worst_regressions": sorted(regressions, key=lambda row: row["delta"],
                                                      reverse=lower_is_better)[:10]}
    return result


def merge(bindings, incoming):
    for path, digest in incoming.items():
        require(path not in bindings or bindings[path] == digest, "Conflicting evidence: " + path)
        bindings[path] = digest


def bind(bindings, paths):
    merge(bindings, {str(path): sha(path) for path in paths})


def paired_root(directory, bindings):
    paths = [directory / name for name in ("plan.json", "result.json", "root-execution.json", "root-command.json")]
    plan, result, execution, _ = map(read, paths)
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == result["plan_sha256"] == sha(paths[0])
            and execution["result_sha256"] == sha(paths[1])
            and execution["root_command_sha256"] == sha(paths[3])
            and result["track_count_per_endpoint"] == 14
            and result["excerpt_count_per_view_per_endpoint"] == 28,
            "Paired source-view root is incomplete: " + str(directory))
    verify_inputs(plan)
    merge(bindings, plan["source_bindings"])
    bind(bindings, paths)
    return plan, result


def full_report(directory, checkpoint, state_digest, bindings):
    paths = [directory / name for name in ("plan.json", "result.json", "execution.json")]
    plan, result, execution = map(read, paths)
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and result["plan_sha256"] == execution["plan_sha256"] == sha(paths[0])
            and result["track_count"] == 14 and result["excerpt_count"] == 28
            and result["graph_delay_samples"] == result["host_queue_samples"] == 128,
            "Full14 endpoint is incomplete: " + str(directory))
    report = result["results"][0]
    require(plan["checkpoint"] == report["checkpoint"] == checkpoint
            and report["model"]["model_state_sha256_after"] == state_digest
            and len(report["tracks"]) == 14, "Full14 and source-view model identities differ")
    # The completed training/scoring receipts own training-corpus verification.
    # This report binds those receipts, the saved inference bytes, and all code;
    # source-view plans below authenticate the fixed validation audio again.
    merge(bindings, {path: digest for path, digest in plan["source_bindings"].items()
                     if Path(path).suffix == ".py"})
    merge(bindings, {checkpoint["path"]: checkpoint["sha256"]})
    bind(bindings, paths)
    return report, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-prefix", default="grouped-vocal-parent-review-012")
    args = parser.parse_args()
    required = [directory / name for directory in (PARENT_VIEWS, CANDIDATE_VIEWS)
                for name in ("plan.json", "result.json", "root-execution.json", "root-command.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if args.preflight:
        print(json.dumps({"status": "pending" if missing else "files_present_unverified", "missing": missing}))
        return
    require(not missing, "Finish both source-view endpoints and record their actual root exit first: " + str(missing))
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    require(args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid output prefix")
    out = PHASE / args.output_prefix
    require(not out.exists(), "Preserve previous reviews")
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.report_latency58_vocal_focus import load_views, music_cells
    from research.direct.report_latency58_branch_gru_int8 import compare_windows
    from research.direct.report_latency58_branch_output_int8 import require_close
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

    code = ("compare.py", "compare_latency58_vocal_views.py", "report_latency58_vocal_focus.py",
            "report_latency58_branch_gru_int8.py", "report_latency58_branch_output_int8.py",
            "run_latency58_deployed_vocal_views.py", "run_latency58_quality.py", "train_latency58.py")
    bindings = {}
    bind(bindings, [Path(__file__).resolve(), ROOT / "research/evaluate.py", ROOT / "research/metrics.py",
                    *(ROOT / "research/direct" / name for name in code)])
    parent_plan, parent_root = paired_root(PARENT_VIEWS, bindings)
    candidate_plan, candidate_root = paired_root(CANDIDATE_VIEWS, bindings)
    source = read(TRAINING / "plan.json")
    completed, execution = read(TRAINING / "result.json"), read(TRAINING / "root-execution.json")
    require(completed["status"] == "training_audit_and_paired_full14_complete"
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"] and execution["result_sha256"] == sha(TRAINING / "result.json")
            and execution["plan_sha256"] == sha(TRAINING / "plan.json")
            and execution["root_command_sha256"] == sha(TRAINING / "root-command.json")
            and candidate_plan["source_training_root"] == str(TRAINING), "Training root is incomplete")
    require(all(parent_plan[key] == candidate_plan[key] for key in
                ("manifest", "config", "track_indices", "track_intervals", "protocol_version")),
            "Parent and candidate source-view protocols differ")
    parent = load_views(PARENT_VIEWS / "ema", bindings)
    require(parent["model"]["checkpoint"] == source["parent_checkpoint"], "Wrong retained training parent")
    bind(bindings, [TRAINING / name for name in ("plan.json", "result.json", "root-execution.json", "root-command.json")])
    parent_full_path = Path(parent["model"]["original_full_mixture_report"]["path"])
    require(sha(parent_full_path) == parent["model"]["original_full_mixture_report"]["sha256"]
            and str(parent_full_path) == source["reference_result"], "Parent full14 report changed")
    parent_full, _ = full_report(parent_full_path.parent, source["parent_checkpoint"],
                                 parent["model"]["model_state_sha256"], bindings)
    require_close(parent_full["aggregate"], legacy._aggregate_tracks(parent_full["tracks"]), "parent aggregate")
    comparisons = {}
    for role in ("raw", "ema"):
        views = load_views(CANDIDATE_VIEWS / role, bindings)
        require(views["model"] == candidate_root["models"][role]
                and views["model"]["checkpoint"] == completed["checkpoints"][role], "Candidate role changed")
        full, original = full_report(TRAINING / ("full14-" + role), views["model"]["checkpoint"],
                                     views["model"]["model_state_sha256"], bindings)
        require_close(full["aggregate"], legacy._aggregate_tracks(full["tracks"]), role + " aggregate")
        music, cells = compare(parent_full, full), music_cells(parent_full, full)
        require_close(music, original["comparison"], role + " stored parent comparison")
        require_close(cells, original["all_track_stem_cells"], role + " stored music cells")
        windows = compare_windows(parent["tracks"], views["tracks"])
        require(len(windows["all_windows"]) == 840, "Incomplete paired windows")
        comparisons[role] = {"full_mixture": music, "all_track_stem_cells": cells,
                             "source_views": compare_reports(parent, views), "paired_windows": windows,
                             "music_regressions": music_summary(cells), "window_summary": window_summary(windows)}
    before = budget_snapshot(source["storage_budget"])
    allowance = 15_000_000
    require(before["headroom_bytes"] > allowance, "Reserve the complete report before writing")
    verify_inputs({"source_bindings": bindings})
    plan = {"schema": "latency58-grouped-parent-review-plan-v1", "source_bindings": bindings,
            "source_training_root": str(TRAINING), "parent_source_views": str(PARENT_VIEWS),
            "candidate_source_views": str(CANDIDATE_VIEWS), "budget_before": before,
            "output_allowance_bytes": allowance, "quality_selected": False}
    out.mkdir()
    write(out / "plan.json", plan)
    result = {"schema": "latency58-grouped-parent-review-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(out / "plan.json"),
        "source_bindings_unchanged": True, "comparisons": comparisons,
        "retained_parent": parent["model"], "parent_training_history": {
            key: parent_plan.get(key) for key in ("original_training_monitor_successful", "host_stability_proven")},
        "gpu_used": False, "inference_executed": False, "quality_selected": False,
        "plugin_replaced": False, "overall_goal_complete": False,
        "budget_after": budget_snapshot(source["storage_budget"]),
        "limitations": ["Existing development measurements only; no unseen confirmation material.",
            "Read lower unwanted vocal output alongside desired vocal gain and Other quality.",
            "The recovered parent's historical monitoring failure remains recorded separately from its saved-model quality.",
            "No deployment graph, physical M4 or human-listening acceptance follows from this report."]}
    require(len(json.dumps(result, indent=2, allow_nan=False).encode()) + (out / "plan.json").stat().st_size
            + 100_000 < allowance, "Report allowance exceeded")
    verify_inputs(plan)
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "parent_full_sdr_db": parent_full["aggregate"]["full_sdr_db"],
        "endpoints": {role: {"full_sdr_db": value["full_mixture"]["metrics"]["full_sdr_db"],
            "instrumental_vocals": value["source_views"]["views"]["instrumental"]["per_stem"]["vocals"]}
            for role, value in comparisons.items()}, "quality_selected": False}))


if __name__ == "__main__":
    main()
