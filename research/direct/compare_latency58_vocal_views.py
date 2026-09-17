"""Compare authenticated vocal-view reports without rerendering or selecting weights."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs

STEMS = ("drums", "bass", "vocals", "other")
LEVEL_FIELDS = ("output_rms_dbfs", "output_to_input_db", "signed_desired_projection_gain")


def compare_reports(reference, candidate):
    import numpy as np
    from research.metrics import mean_or_none
    require(reference["version"] == candidate["version"]
            and [t["name"] for t in reference["tracks"]] == [t["name"] for t in candidate["tracks"]]
            and len(reference["tracks"]) == len(candidate["tracks"]) == 14, "Vocal diagnostic panels differ")
    views, names = {}, [t["name"] for t in reference["tracks"]]
    indices = np.random.default_rng(91).integers(0, 14, (20000, 14))
    for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
        require(left["index"] == right["index"] and left["intervals"] == right["intervals"]
                and left["stream"]["input_stream_sha256"] == right["stream"]["input_stream_sha256"],
                "Physical source intervals or complete remixed input stream differ")
    for view in ("vocals_only", "instrumental"):
        tracks = [[], []]
        for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
            a, b = left["views"][view], right["views"][view]
            require(a["desired_stems"] == b["desired_stems"] and a["input_active_windows"] == b["input_active_windows"]
                    and a["total_windows"] == b["total_windows"], "View or activity support differs")
            for x, y in zip(a["windows"], b["windows"], strict=True):
                require(all(x[k] == y[k] for k in
                            ("excerpt_index", "physical_start", "physical_end", "input_rms_dbfs", "input_active"))
                        and all(x["per_stem"][s]["desired_active"] == y["per_stem"][s]["desired_active"]
                                for s in STEMS), "Input-window geometry or desired-source support differs")
            for destination, current in zip(tracks, (a, b), strict=True):
                cells = {}
                for stem in STEMS:
                    levels = current["native_output_levels"][stem]
                    active = [w["per_stem"][stem] for w in current["windows"] if w["input_active"]]
                    require(levels["input_active_windows"] == len(active)
                            and all(levels[k] == mean_or_none(c[k] for c in active) for k in LEVEL_FIELDS),
                            "Stored track level does not match its complete window support")
                    scores = current["standard_scores_on_remixed_references"]["per_stem"][stem]
                    cells[stem] = {**{k: levels[k] for k in LEVEL_FIELDS},
                                   "desired_full_sdr_db": scores["full_sdr_db"],
                                   "desired_low_sdr_db": scores["band_sdr_db"]["low_20_250"]}
                destination.append(cells)
        stem_results = {}
        for stem in STEMS:
            fields = {}
            for field in (*LEVEL_FIELDS, "desired_full_sdr_db", "desired_low_sdr_db"):
                values = [[t[stem][field] for t in panel] for panel in tracks]
                require([v is None for v in values[0]] == [v is None for v in values[1]],
                        "Desired-source metric eligibility differs")
                eligibility = [i for i, value in enumerate(values[0]) if value is not None]
                for report, sequence in zip((reference, candidate), values, strict=True):
                    stored = report["aggregate"][view]["per_stem"][stem][field]
                    recomputed = mean_or_none(sequence)
                    require((stored is None and recomputed is None) or
                            (stored is not None and recomputed is not None and abs(stored - recomputed) <= 1e-12),
                            "Stored aggregate is not the equal-track mean")
                if not eligibility:
                    fields[field] = None
                    continue
                require(all(math.isfinite(v) for sequence in values for v in sequence if v is not None),
                        "Nonfinite diagnostic metric")
                differences = np.array([values[1][i] - values[0][i] if i in eligibility else np.nan
                                        for i in range(14)], dtype=np.float64)
                sampled = differences[indices]
                valid = np.isfinite(sampled).sum(axis=1)
                means = np.nansum(sampled, axis=1)[valid > 0] / valid[valid > 0]
                fields[field] = {
                    "reference": mean_or_none(values[0]), "candidate": mean_or_none(values[1]),
                    "delta": float(np.nanmean(differences)),
                    "paired_track_bootstrap_95_percent": np.quantile(means, [.025, .975]).tolist(),
                    "eligible_tracks": len(eligibility), "negative_tracks": int(np.sum(differences < 0)),
                    "positive_tracks": int(np.sum(differences > 0)),
                    "per_track_delta": {name: None if not np.isfinite(d) else float(d)
                                        for name, d in zip(names, differences, strict=True)}}
            stem_results[stem] = {"off_target": reference["aggregate"][view]["per_stem"][stem]["off_target"],
                                  "metrics": fields}
        views[view] = {"input_active_windows": reference["aggregate"][view]["input_active_windows"],
                       "per_stem": stem_results}
    return {"reference_model": reference["model"], "candidate_model": candidate["model"],
            "all_physical_inputs_and_window_support_exact": True, "all_stored_level_aggregates_recomputed": True,
            "aggregate_reduction_order_tolerance": 1e-12,
            "views": views, "bootstrap_samples": 20000, "bootstrap_seed": 91,
            "uncertainty_scope": "Paired track sampling only; no training-seed uncertainty or correction for repeated selection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Vocal comparison plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-views-comparison-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Require CUDA-hidden CPU1")
    verify_inputs(plan)
    from research.direct.latency58_sdr_checkpoint import require_space
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve comparison result")
    began, before, reports = time.monotonic(), require_space(plan, 3_000_000), {}
    for label, spec in plan["reports"].items():
        for binding in spec.values():
            require(sha(binding["path"]) == binding["sha256"], "Diagnostic comparison input changed")
        result, execution, original = (read(spec[k]["path"]) for k in ("result", "execution", "plan"))
        require(result["schema"] == "latency58-vocal-views-evaluation-result-v1" and result["status"] == "pass"
                and result["source_bindings_unchanged"] and not result["cuda_initialized"]
                and result["training_updates_executed"] == 0 and not result["confirmation_material_used"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"]
                and result["plan_sha256"] == execution["plan_sha256"] == spec["plan"]["sha256"]
                and result["model"] == original["model"], "Diagnostic is not completely authenticated")
        verify_inputs(original)
        reports[label] = result
    comparisons = {key: compare_reports(reports[value["reference"]], reports[value["candidate"]])
                   for key, value in plan["comparisons"].items()}
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-vocal-views-comparison-result-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "comparisons": comparisons, "elapsed_seconds": time.monotonic() - began,
        "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
        "training_updates_executed": 0, "quality_selected": False, "confirmation_material_used": False,
        "limitations": ["Controlled development remixes; retain the original mixture metrics and native listening.",
                        "Native output levels and ratios must be read alongside desired-source SDR and signed gain."]})
    print({"status": "pass", "comparisons": list(comparisons)}, flush=True)


if __name__ == "__main__":
    main()
