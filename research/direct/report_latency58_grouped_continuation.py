"""Review complete grouped endpoints against their starting parent and retained best.

Reuses authenticated measurements and existing comparison functions. No inference,
training, checkpoint selection or plugin replacement is performed by this report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_grouped_vocal_parent import (
    bind, full_report, music_summary, paired_root, window_summary)


ROLES = ("raw", "ema")
REFERENCE_ROLES = ("starting_parent", "retained_best")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def compare_endpoint(reference, candidate):
    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.report_latency58_branch_gru_int8 import compare_windows

    music = compare(reference["full"], candidate["full"])
    cells = music_cells(reference["full"], candidate["full"])
    windows = compare_windows(reference["views"]["tracks"], candidate["views"]["tracks"])
    require(len(cells) == 14 and sum(len(stems) for stems in cells.values()) == 56
            and len(windows["all_windows"]) == 840, "Incomplete full14 or paired-window review")
    return {"full_mixture": music, "all_track_stem_cells": cells,
            "source_views": compare_reports(reference["views"], candidate["views"]),
            "paired_windows": windows, "music_regressions": music_summary(cells),
            "window_summary": window_summary(windows)}


def collect(training, candidate_directory, reference_directories):
    from research import evaluate as legacy
    from research.direct.report_latency58_vocal_focus import load_views
    from research.direct.report_latency58_branch_output_int8 import require_close

    bindings = {}
    code = ("report_latency58_grouped_vocal_parent.py", "compare.py", "compare_latency58_vocal_views.py",
            "report_latency58_vocal_focus.py", "report_latency58_branch_gru_int8.py",
            "report_latency58_branch_output_int8.py", "run_latency58_deployed_vocal_views.py",
            "run_latency58_quality.py", "train_latency58.py")
    bind(bindings, [Path(__file__).resolve(), ROOT / "research/evaluate.py", ROOT / "research/metrics.py",
                    *(ROOT / "research/direct" / name for name in code)])
    candidate_plan, candidate_root = paired_root(candidate_directory, bindings)
    source = read(training / "plan.json")
    completed, execution = read(training / "result.json"), read(training / "root-execution.json")
    require(completed["status"] == "training_audit_and_paired_full14_complete"
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"] and execution["result_sha256"] == sha(training / "result.json")
            and execution["plan_sha256"] == sha(training / "plan.json")
            and execution["root_command_sha256"] == sha(training / "root-command.json")
            and candidate_plan["source_training_root"] == str(training)
            and set(completed["checkpoints"]) == set(ROLES), "Training root is incomplete or mismatched")
    bind(bindings, [training / name for name in
                    ("plan.json", "result.json", "root-execution.json", "root-command.json")])

    expected = {
        "starting_parent": (source["parent_checkpoint"], source["reference_result"]),
        "retained_best": (source.get("retained_best_research_checkpoint", source["parent_checkpoint"]),
                          source.get("retained_best_research_reference_result", source["reference_result"]))}
    references, cache = {}, {}
    for label in REFERENCE_ROLES:
        directory = reference_directories[label]
        if directory not in cache:
            ref_plan, ref_root = paired_root(directory, bindings)
            require(all(ref_plan[key] == candidate_plan[key] for key in
                        ("manifest", "config", "track_indices", "track_intervals", "protocol_version")),
                    "Reference and candidate physical source-view protocols differ")
            views = load_views(directory / "ema", bindings)
            require(views["model"] == ref_root["models"]["ema"], "Reference endpoint role changed")
            full_path = Path(views["model"]["original_full_mixture_report"]["path"])
            require(sha(full_path) == views["model"]["original_full_mixture_report"]["sha256"],
                    "Reference full14 evidence changed")
            full, _ = full_report(full_path.parent, views["model"]["checkpoint"],
                                  views["model"]["model_state_sha256"], bindings)
            require_close(full["aggregate"], legacy._aggregate_tracks(full["tracks"]), label + " aggregate")
            cache[directory] = {"views": views, "full": full, "full_path": full_path,
                                "plan": ref_plan, "root": ref_root}
        reference = cache[directory]
        checkpoint, result_path = expected[label]
        require(reference["views"]["model"]["checkpoint"] == checkpoint
                and str(reference["full_path"]) == result_path, "Wrong " + label + " identity")
        references[label] = reference
    if "retained_best_research_full_sdr_db" in source:
        require_close(references["retained_best"]["full"]["aggregate"]["full_sdr_db"],
                      source["retained_best_research_full_sdr_db"], "retained best score")

    candidates = {}
    for role in ROLES:
        views = load_views(candidate_directory / role, bindings)
        require(views["model"] == candidate_root["models"][role]
                and views["model"]["checkpoint"] == completed["checkpoints"][role], "Candidate role changed")
        full, original = full_report(training / ("full14-" + role), views["model"]["checkpoint"],
                                     views["model"]["model_state_sha256"], bindings)
        require_close(full["aggregate"], legacy._aggregate_tracks(full["tracks"]), role + " aggregate")
        candidates[role] = {"views": views, "full": full, "original": original}

    comparisons = {label: {role: compare_endpoint(references[label], candidates[role]) for role in ROLES}
                   for label in REFERENCE_ROLES}
    for role in ROLES:
        value = comparisons["starting_parent"][role]
        original = candidates[role]["original"]
        require_close(value["full_mixture"], original["comparison"], role + " stored parent comparison")
        require_close(value["all_track_stem_cells"], original["all_track_stem_cells"], role + " stored cells")

    peer = compare_endpoint(candidates["raw"], candidates["ema"])
    peer_binding = completed["paired_comparison"]
    peer_path = Path(peer_binding["path"])
    require(sha(peer_path) == peer_binding["sha256"], "Stored raw/EMA full14 comparison changed")
    stored_peer = read(peer_path)
    bind(bindings, [peer_path])
    require_close(peer["full_mixture"], stored_peer["comparison"], "stored raw/EMA full14 comparison")
    require_close(peer["all_track_stem_cells"], stored_peer["all_track_stem_cells"], "stored raw/EMA cells")
    require_close(peer["source_views"], candidate_root["comparisons"]["ema_vs_raw"]["aggregate"],
                  "stored raw/EMA source-view aggregate")
    require_close(peer["paired_windows"], candidate_root["comparisons"]["ema_vs_raw"]["windows"],
                  "stored raw/EMA source-view windows")
    comparisons["ema_vs_raw"] = peer
    verify_inputs({"source_bindings": bindings})
    return source, bindings, comparisons, references, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--candidate-views", type=Path, required=True)
    parser.add_argument("--starting-parent-views", type=Path, required=True)
    parser.add_argument("--retained-best-views", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--qualify-against", type=Path)
    args = parser.parse_args()
    training, candidate = args.quality_root.resolve(), args.candidate_views.resolve()
    directories = {"starting_parent": args.starting_parent_views.resolve(),
                   "retained_best": args.retained_best_views.resolve()}
    require(all(path.is_relative_to(PHASE) for path in (training, candidate, *directories.values())),
            "Use retained local completed evidence")
    required = [directory / name for directory in (training, candidate, *directories.values())
                for name in ("plan.json", "result.json", "root-execution.json", "root-command.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if args.preflight:
        print(json.dumps({"status": "pending" if missing else "files_present_unverified", "missing": missing}))
        return
    require(not missing, "Complete training, full14 and both source views with actual root exits first: " + str(missing))
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    require(args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid output prefix")
    out = PHASE / args.output_prefix
    require(not out.exists(), "Preserve completed comparisons")
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    from research.direct.report_latency58_branch_output_int8 import require_close

    source, bindings, comparisons, references, candidates = collect(training, candidate, directories)
    control = None
    if args.qualify_against:
        control_dir = args.qualify_against.resolve(strict=True)
        control_plan, control_result, control_execution = (read(control_dir / name) for name in
                                                         ("plan.json", "result.json", "execution.json"))
        require(control_result["status"] == "pass" and control_execution["actual_exit_code"] == 0
                and control_execution["source_bindings_unchanged"] and not control_execution["timed_out"]
                and control_execution["result_sha256"] == sha(control_dir / "result.json")
                and control_execution["plan_sha256"] == control_result["plan_sha256"] == sha(control_dir / "plan.json")
                and control_plan["source_training_root"] == str(training), "Existing control review is incomplete")
        verify_inputs(control_plan)
        bind(bindings, [control_dir / name for name in ("plan.json", "result.json", "execution.json")])
        require_close(comparisons["starting_parent"], control_result["comparisons"], "complete previous review")
        require(candidates["raw"]["views"]["model"]["model_state_sha256"]
                != candidates["ema"]["views"]["model"]["model_state_sha256"], "Control must include distinct endpoints")
        control = {"existing_review_result": {"path": str(control_dir / "result.json"),
                                             "sha256": sha(control_dir / "result.json")},
                   "all_parent_metrics_cells_and_windows_reproduced": True,
                   "distinct_raw_ema_full14_and_source_views_reproduced": True,
                   "comparison_fingerprints": {label: fingerprint(value) for label, value in comparisons.items()}}

    allowance = 2_000_000 if control else 25_000_000
    before = budget_snapshot(source["storage_budget"])
    require(before["headroom_bytes"] > allowance, "Reserve the report before writing")
    verify_inputs({"source_bindings": bindings})
    plan = {"schema": "latency58-grouped-continuation-review-plan-v1", "source_bindings": bindings,
            "source_training_root": str(training), "candidate_source_views": str(candidate),
            "reference_source_views": {key: str(value) for key, value in directories.items()},
            "budget_before": before, "output_allowance_bytes": allowance,
            "qualification_only": control is not None, "quality_selected": False}
    out.mkdir()
    write(out / "plan.json", plan)
    result = {"schema": "latency58-grouped-continuation-review-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(out / "plan.json"),
        "source_bindings_unchanged": True, "qualification_only": control is not None,
        "comparisons": None if control else comparisons, "control": control,
        "reference_models": {label: reference["views"]["model"] for label, reference in references.items()},
        "reference_training_history": {label: {key: reference["plan"].get(key) for key in
            ("original_training_monitor_successful", "host_stability_proven")} for label, reference in references.items()},
        "candidate_models": {role: endpoint["views"]["model"] for role, endpoint in candidates.items()},
        "candidate_full_sdr_db": {role: endpoint["full"]["aggregate"]["full_sdr_db"]
                                  for role, endpoint in candidates.items()},
        "gpu_used": False, "inference_executed": False, "quality_selected": False,
        "plugin_replaced": False, "overall_goal_complete": False,
        "budget_after": budget_snapshot(source["storage_budget"]),
        "limitations": ["Existing development measurements; no unseen confirmation or seed uncertainty.",
            "Consider unwanted vocal output alongside desired vocal gain, Other and every full-mixture stem.",
            "Historical monitor failures remain recorded separately from saved-model quality.",
            "No deployment graph, physical-M4 timing or human-listening acceptance follows from this report."]}
    require(len(json.dumps(result, indent=2, allow_nan=False).encode()) + (out / "plan.json").stat().st_size
            + 100_000 < allowance, "Report exceeded its allowance")
    verify_inputs(plan)
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "qualification_only": control is not None,
        "candidate_full_sdr_db": result["candidate_full_sdr_db"], "quality_selected": False,
        "result_sha256": sha(out / "result.json")}))


if __name__ == "__main__":
    main()
