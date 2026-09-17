"""Supplement signed vocal-view gain with windowwise absolute error from unity."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_vocal_focus import load_views


def compare_gain(reference, candidate):
    import numpy as np
    windows, tracks = [], []
    for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
        require(left["name"] == right["name"] and left["intervals"] == right["intervals"]
                and left["stream"]["input_stream_sha256"] == right["stream"]["input_stream_sha256"],
                "Different complete source-view inputs")
        current = []
        for a, b in zip(left["views"]["vocals_only"]["windows"], right["views"]["vocals_only"]["windows"], strict=True):
            require(all(a[k] == b[k] for k in ("excerpt_index", "physical_start", "physical_end", "input_rms_dbfs", "input_active"))
                    and a["per_stem"]["vocals"]["desired_active"] == b["per_stem"]["vocals"]["desired_active"],
                    "Different physical window or source support")
            if not a["input_active"]:
                continue
            require(a["per_stem"]["vocals"]["desired_active"], "Active vocal-only input lacks active desired vocal")
            x, y = (v["per_stem"]["vocals"]["signed_desired_projection_gain"] for v in (a, b))
            require(math.isfinite(x) and math.isfinite(y), "Nonfinite desired gain")
            row = {"track": right["name"], "excerpt_index": a["excerpt_index"],
                   "physical_start": a["physical_start"], "physical_end": a["physical_end"],
                   "reference_gain": x, "candidate_gain": y,
                   "reference_error": abs(x - 1), "candidate_error": abs(y - 1),
                   "delta": abs(y - 1) - abs(x - 1)}
            current.append(row)
            windows.append(row)
        require(current, "Expected active vocal support on every primary track")
        tracks.append({"track": right["name"], "windows": len(current), **{
            k: statistics.mean(row[k] for row in current) for k in ("reference_error", "candidate_error", "delta")}})
    require(len(windows) == 319 and len(tracks) == 14, "Incomplete original vocal-only support")
    differences = np.array([row["delta"] for row in tracks], dtype=np.float64)
    indices = np.random.default_rng(91).integers(0, 14, (20000, 14))
    return {"reference_model": reference["model"], "candidate_model": candidate["model"],
            "equal_track_means": {k: statistics.mean(row[k] for row in tracks)
                                  for k in ("reference_error", "candidate_error", "delta")},
            "paired_track_bootstrap_95_percent": np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist(),
            "better_windows": sum(row["delta"] < 0 for row in windows),
            "worse_windows": sum(row["delta"] > 0 for row in windows),
            "better_tracks": sum(row["delta"] < 0 for row in tracks),
            "worse_tracks": sum(row["delta"] > 0 for row in tracks),
            "tracks": tracks, "windows": windows,
            "largest_error_increases": sorted(windows, key=lambda row: row["delta"], reverse=True)[:5],
            "all_input_streams_and_window_support_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 gain supplement")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-gain-error-plan-v1"
            and set(plan["references"]) == {"working", "focused_control", "drum500", "drum1000"}, "Different comparison scope")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve gain supplement")
    from research.direct.latency58_sdr_checkpoint import require_space
    before = require_space(plan, 2_000_000)
    evidence = {}
    candidate = load_views(Path(plan["candidate_directory"]), evidence)
    comparisons = {}
    for label, directory in plan["references"].items():
        reference = load_views(Path(directory), evidence)
        require(reference["model"]["model_state_sha256"] != candidate["model"]["model_state_sha256"], "Same compared model")
        comparisons[label] = compare_gain(reference, candidate)
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound gain inputs")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-vocal-gain-error-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "definition": "Mean of abs(signed reference projection gain - 1) within each track, then equal-track mean.",
          "comparisons": comparisons, "bootstrap_samples": 20000, "bootstrap_seed": 91,
          "primary_protocol_changed": False, "inference_executed": False, "training_updates_executed": 0,
          "quality_selected": False, "confirmation_material_used": False, "human_listening_completed": False,
          "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
          "limitations": ["Projection gain can contain correlated error and must be read with SDR, leakage and listening.",
                          "Full-band active vocal-only windows; natural quiet sources have a separate report.",
                          "Track intervals omit training-seed uncertainty and repeated-selection effects."]})
    print({"status": "pass", "references": list(comparisons), "active_windows_per_comparison": 319}, flush=True)


if __name__ == "__main__":
    main()
