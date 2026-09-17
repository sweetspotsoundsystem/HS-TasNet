"""Preserve aligned vocal-window costs and quiet-output ratios for endpoint review."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_vocal_focus import REFERENCES, STEMS, load_views
from research.direct.compare_latency58_vocal_views import LEVEL_FIELDS, compare_reports
from research.direct.train_latency58 import verify_inputs


def pair(left, right):
    require((left is None) == (right is None)
            and (left is None or (math.isfinite(left) and math.isfinite(right))),
            "Metric support differs or a value is nonfinite")
    return [left, right, None if left is None else right - left]


def window_comparison(reference, candidate):
    """Return every window, plus native-level and wanted-gain extremes."""
    all_windows, summaries = {}, {}
    for view in ("vocals_only", "instrumental"):
        records = []
        for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
            require(left["name"] == right["name"], "Track order differs")
            for a, b in zip(left["views"][view]["windows"], right["views"][view]["windows"], strict=True):
                geometry = ("excerpt_index", "physical_start", "physical_end", "input_rms_dbfs", "input_active")
                require(all(a[k] == b[k] for k in geometry), "Window geometry or input activity differs")
                cells = {}
                for stem in STEMS:
                    x, y = a["per_stem"][stem], b["per_stem"][stem]
                    require(all(x[k] == y[k] for k in ("desired_active", "off_target")), "Desired support differs")
                    cells[stem] = {"desired_active": y["desired_active"], "off_target": y["off_target"],
                                   "metrics": {k: pair(x[k], y[k]) for k in LEVEL_FIELDS}}
                records.append({"track": left["name"], **{k: a[k] for k in geometry}, "per_stem": cells})
        require(len(records) == 420 and sum(w["input_active"] for w in records)
                == candidate["aggregate"][view]["input_active_windows"], "Incomplete window inventory")
        all_windows[view] = records
        summaries[view] = {}
        for stem in STEMS:
            summaries[view][stem] = {}
            for field in LEVEL_FIELDS:
                eligible = []
                for w in records:
                    values = w["per_stem"][stem]["metrics"][field]
                    if w["input_active"] and values[0] is not None:
                        eligible.append({"track": w["track"], "excerpt_index": w["excerpt_index"],
                                         "physical_start": w["physical_start"], "physical_end": w["physical_end"],
                                         "reference": values[0], "candidate": values[1], "delta": values[2]})
                summaries[view][stem][field] = {
                    "eligible_windows": len(eligible),
                    "positive_windows": sum(w["delta"] > 0 for w in eligible),
                    "negative_windows": sum(w["delta"] < 0 for w in eligible),
                    "unchanged_windows": sum(w["delta"] == 0 for w in eligible),
                    "largest_increases": sorted(eligible, key=lambda w: w["delta"], reverse=True)[:5],
                    "largest_decreases": sorted(eligible, key=lambda w: w["delta"])[:5],
                    "highest_candidate_values": sorted(eligible, key=lambda w: w["candidate"], reverse=True)[:5],
                    "lowest_candidate_values": sorted(eligible, key=lambda w: w["candidate"])[:5],
                }
    return all_windows, summaries


def quiet_comparison(reference, candidate):
    cells, errors = [], []
    for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
        require(left["name"] == right["name"] and left["excerpts"] == right["excerpts"], "Music intervals differ")
        for stem in STEMS:
            a, b = left["per_stem"][stem], right["per_stem"][stem]
            require(a["absent_windows"] == b["absent_windows"], "Quiet-reference support differs")
            native = pair(a["absent_fp_dbfs"], b["absent_fp_dbfs"])
            ratio = pair(a["absent_fp_ratio_db"], b["absent_fp_ratio_db"])
            require((native[0] is None) == (ratio[0] is None), "Quiet metric support differs")
            if native[0] is not None:
                errors.append(abs(native[2] - ratio[2]))
            cells.append({"track": left["name"], "stem": stem, "absent_windows": b["absent_windows"],
                          "absent_fp_dbfs": native, "absent_fp_ratio_db": ratio})
    require(len(cells) == 56, "Incomplete quiet-reference inventory")
    return {"all_track_stem_cells": cells, "eligible_cells": len(errors),
            "max_native_vs_ratio_delta_difference_db": max(errors, default=0.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 window review")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-window-review-plan-v1", "Unknown review plan")
    verify_inputs(plan)
    prefix = plan["candidate_prefix"]
    require(prefix in {"vocal-focus-" + arm + "-250" for arm in ("original", "focused", "focused-mixer")},
            "Different pilot endpoint")
    summary_dir = PHASE / (prefix + "-summary-001")
    summary, execution = read(summary_dir / "result.json"), read(summary_dir / "summary-execution.json")
    verify_inputs(summary)
    require(summary["schema"] == "latency58-vocal-focus-quality-summary-v1" and summary["status"] == "pass"
            and summary["source_bindings_unchanged"] and summary["all_metrics_compared"] and summary["step"] == 250
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and summary["plan_sha256"] == execution["plan_sha256"] == sha(summary_dir / "plan.json"),
            "The complete endpoint summary has not passed")
    evidence = {str(summary_dir / n): sha(summary_dir / n)
                for n in ("plan.json", "result.json", "summary-execution.json")}
    evidence.update(summary["source_bindings"])
    _, music = load_completed(PHASE / (prefix + "-full14-001"), evidence)
    views = load_views(PHASE / (prefix + "-views-001"), evidence)
    require(music["results"][0]["model"]["model_state_sha256"] == views["model"]["model_state_sha256"]
            == summary["model_state_sha256"], "Mixed candidate endpoints")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists()
            and not (out / "window-comparisons.json.gz").exists(), "Preserve window review")
    from research.direct.latency58_vocal_focus_checkpoint import require_space
    counted = require_space(plan, 3_000_000)
    detailed, compared = {}, {}
    for label, reference_prefix in REFERENCES.items():
        _, reference_music = load_completed(PHASE / (reference_prefix + "-full14-001"), evidence,
                                            canonical_baseline=label == "working")
        reference_views = load_views(PHASE / ("vocal-views-" + label + "-001"), evidence)
        require(reference_views["model"]["model_state_sha256"]
                == reference_music["results"][0]["model"]["model_state_sha256"]
                == summary["reference_model_states"][label], "Mixed reference endpoints")
        require(compare_reports(reference_views, views) == summary["comparisons"][label]["vocal_views"],
                "Recomputed track-level view comparison differs from the reviewed summary")
        detailed[label], local = window_comparison(reference_views, views)
        compared[label] = {"window_extremes": local,
                           "quiet_reference": quiet_comparison(reference_music["results"][0], music["results"][0])}
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound completed evidence")
    document = {"schema": "latency58-vocal-focus-window-comparisons-v1", "pair_order": ["reference", "candidate", "delta"],
                "candidate_model_state_sha256": summary["model_state_sha256"],
                "reference_model_states": summary["reference_model_states"], "comparisons": detailed}
    compressed = gzip.compress(json.dumps(document, allow_nan=False, separators=(",", ":")).encode(), mtime=0)
    require(json.loads(gzip.decompress(compressed)) == document, "Compressed window records do not round-trip exactly")
    result = {"schema": "latency58-vocal-focus-window-review-v1", "status": "pass", "plan_sha256": args.plan_sha256,
              "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
              "candidate_prefix": prefix, "candidate_model_state_sha256": summary["model_state_sha256"],
              "comparisons": compared, "compressed_windows_bytes": len(compressed),
              "all_window_records_retained": True, "all_track_view_comparisons_exact": True,
              "compression_round_trip_exact": True, "current_counted_bytes": counted,
              "quality_selected": False, "human_listening_completed": False,
              "limitations": ["Extremes describe observed development windows, without sampling uncertainty.",
                              "Read native leakage alongside wanted gain, fidelity and full-mixture scores.",
                              "No inference, training, normalization, confirmation or human listening is performed."]}
    require(len(compressed) + len(json.dumps(result, indent=2)) < 2_900_000, "Report exceeds its storage reserve")
    verify_inputs(plan)
    path = out / "window-comparisons.json.gz"
    with path.open("xb") as stream:
        stream.write(compressed)
    result["compressed_windows_sha256"] = sha(path)
    write(out / "result.json", result)
    print({"status": "pass", "candidate_prefix": prefix, "compressed_windows_bytes": len(compressed),
           "complete_windows_per_view_and_reference": len(detailed["working"]["vocals_only"])}, flush=True)


if __name__ == "__main__":
    main()
