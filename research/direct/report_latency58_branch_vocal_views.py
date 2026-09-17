"""Pair completed branch-memory vocal views with all windows and full-mixture costs."""
import argparse
import gzip
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def load_endpoint(directory):
    from research.direct.report_latency58_vocal_focus import load_views
    evidence = {}
    views = load_views(directory, evidence)
    plan = read(directory / "plan.json")
    completion = read(directory / "actual-root-tool-completion.json")
    root = read(directory / "root-execution.json")
    require(plan["schema"] == "latency58-branch-vocal-views-evaluation-plan-v1"
            and views["model"]["kind"] == "branch_memory"
            and root["actual_exit_code"] == completion["actual_exit_code"] == 0
            and root["actual_root_session"] == completion["actual_root_session"]
            and root["actual_tool_output_chunk"] == completion["actual_tool_output_chunk"]
            and root["source_bindings_unchanged"]
            and root["execution_sha256"] == sha(directory / "execution.json"),
            "Require the actual completed vocal-view launcher and its unchanged child execution")
    original = views["model"]["original_full_mixture_report"]
    require(sha(original["path"]) == original["sha256"], "Original full-mixture result changed")
    music = read(original["path"])
    require(music["status"] == "pass" and music["source_bindings_unchanged"]
            and music["track_count"] == 14 and music["excerpt_count"] == 28
            and music["graph_delay_samples"] == music["host_queue_samples"] == 128
            and music["results"][0]["checkpoint"] == views["model"]["checkpoint"]
            and music["results"][0]["model"]["model_state_sha256_after"] == views["model"]["model_state_sha256"],
            "Original full mixture and source views must use the same saved model")
    verify_inputs(music)
    for name in ("root-execution.json", "actual-root-tool-completion.json"):
        evidence[str(directory / name)] = sha(directory / name)
    evidence[original["path"]] = original["sha256"]
    evidence.update(plan["source_bindings"])
    return plan, views, music["results"][0], evidence


def main():
    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.report_latency58_vocal_focus_windows import window_comparison, quiet_comparison
    from research.direct.latency58_sdr_checkpoint import require_space
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a CUDA-hidden CPU1 comparison")
    reference_dir, candidate_dir = args.reference_directory.resolve(strict=True), args.candidate_directory.resolve(strict=True)
    out = args.output_directory.resolve()
    require(reference_dir.parent == candidate_dir.parent == out.parent == PHASE
            and reference_dir != candidate_dir and not out.exists(), "Preserve distinct completed endpoints and prior reviews")
    reference_plan, reference, reference_music, bindings = load_endpoint(reference_dir)
    plan, candidate, candidate_music, candidate_bindings = load_endpoint(candidate_dir)
    require(all(bindings.get(path, digest) == digest for path, digest in candidate_bindings.items()),
            "Compared endpoints bind different bytes at a shared path")
    bindings.update(candidate_bindings)
    for key in ("manifest", "config", "protocol_template", "track_indices", "track_intervals"):
        require(reference_plan[key] == plan[key], "Paired vocal-view protocol differs: " + key)
    require(reference["model"]["model_state_sha256"] != candidate["model"]["model_state_sha256"],
            "Compare distinct saved model states")
    for name in ("report_latency58_branch_vocal_views.py", "compare_latency58_vocal_views.py",
                 "report_latency58_vocal_focus_windows.py", "report_latency58_vocal_focus.py", "compare.py"):
        path = ROOT / "research/direct" / name
        bindings[str(path)] = sha(path)
    verify_inputs({"source_bindings": bindings})
    require(plan["stop_counted_bytes"] + plan["outside_roots_reservation_bytes"] == 90_000_000_000,
            "Keep the current artifact cap")
    counted = require_space(plan, 5_000_000)
    comparison = compare_reports(reference, candidate)
    windows, extremes = window_comparison(reference, candidate)
    original_comparison = compare(reference_music, candidate_music)
    original_cells = music_cells(reference_music, candidate_music)
    quiet = quiet_comparison(reference_music, candidate_music)
    document = {"schema": "latency58-branch-vocal-view-window-pairs-v1",
                "pair_order": ["reference", "candidate", "delta"],
                "reference_model": reference["model"], "candidate_model": candidate["model"], "windows": windows}
    compressed = gzip.compress(json.dumps(document, allow_nan=False, separators=(",", ":")).encode(), mtime=0)
    require(json.loads(gzip.decompress(compressed)) == document, "Window archive did not round-trip exactly")
    result = {"schema": "latency58-branch-vocal-view-review-v1", "status": "pass",
              "reference_directory": str(reference_dir), "candidate_directory": str(candidate_dir),
              "reference_model": reference["model"], "candidate_model": candidate["model"],
              "source_bindings": bindings, "source_bindings_unchanged": True,
              "vocal_views": comparison, "window_extremes": extremes,
              "full_mixture_comparison": original_comparison, "full_mixture_track_stem_cells": original_cells,
              "quiet_reference_comparison": quiet, "all_840_window_pairs_retained": True,
              "all_stored_track_and_aggregate_levels_recomputed": True,
              "compressed_window_bytes": len(compressed), "compression_round_trip_exact": True,
              "counted_bytes_before": counted, "artifact_cap_bytes": 90_000_000_000,
              "quality_selected": False, "training_updates_executed": 0, "human_listening_completed": False,
              "limitations": ["Source views are controlled remixes of development stems; their recordings may contain bleed.",
                              "Read unwanted output levels alongside wanted-source SDR, signed gain and original full-mixture quality.",
                              "A signed gain increase is not automatically an improvement; assess its distance from one.",
                              "Bootstrap intervals cover paired-track sampling, not training-seed uncertainty or repeated selection.",
                              "No new native timing, M4 qualification or human listening is established by this report."]}
    require(len(compressed) + len(json.dumps(result, indent=2, allow_nan=False).encode()) < 4_000_000,
            "Paired review exceeds its reserved size")
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    write(out / "plan.json", {"reference_directory": str(reference_dir), "candidate_directory": str(candidate_dir),
                              "source_bindings": bindings, "artifact_cap_bytes": 90_000_000_000})
    path = out / "window-comparisons.json.gz"
    with path.open("xb") as stream:
        stream.write(compressed)
    result["compressed_windows_sha256"] = sha(path)
    result["plan_sha256"] = sha(out / "plan.json")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "output_directory": str(out), "window_pairs": 840,
                      "full_mixture_sdr_delta": original_comparison["metrics"]["full_sdr_db"]["delta"]}), flush=True)


if __name__ == "__main__":
    main()
