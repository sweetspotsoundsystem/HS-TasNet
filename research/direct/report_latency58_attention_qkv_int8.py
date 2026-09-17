"""Review completed seventeen-projection quality, vocal views and native cost.

This is a report-only operation. It neither rerenders audio nor selects a release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_m4_followup_budget import POLICY, snapshot

FOLLOWUP = ROOT / "research/m4_followup_20260916"

RELEASED = "c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61"
CANDIDATE = "08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748"
STAGES = {"full_reference": str(PHASE / "branch-output-int8-quality-001"),
          "vocal_reference": str(PHASE / "branch-output-int8-quality-001"),
          "candidate": str(FOLLOWUP / "attention-qkv-int8-quality-001"),
          "native": str(FOLLOWUP / "attention-qkv-native-001")}


def stage_paths(stage):
    root = Path(stage)
    receipt = FOLLOWUP / "attention-qkv-native-execution.json" if stage == STAGES["native"] else root / "execution.json"
    return [root / "plan.json", root / "result.json", receipt]

STEMS = ("drums", "bass", "vocals", "other")


def require_close(stored, computed, location="aggregate"):
    """Compare complete nested summaries, including null support and finiteness."""
    if isinstance(computed, dict):
        require(isinstance(stored, dict) and stored.keys() == computed.keys(), location + " keys differ")
        for key in computed:
            require_close(stored[key], computed[key], location + "." + key)
    elif isinstance(computed, list):
        require(isinstance(stored, list) and len(stored) == len(computed), location + " length differs")
        for index, (a, b) in enumerate(zip(stored, computed, strict=True)):
            require_close(a, b, location + "[%d]" % index)
    elif isinstance(computed, (int, float)) and not isinstance(computed, bool):
        require(isinstance(stored, (int, float)) and math.isfinite(stored) and math.isfinite(computed)
                and abs(stored - computed) <= 1e-12, location + " value differs")
    else:
        require(stored == computed, location + " support/value differs")


def missing_inputs():
    return [str(path) for stage in STAGES.values() for path in stage_paths(stage) if not path.is_file()]


def load_completed(stage, bindings):
    paths = stage_paths(stage)
    plan, result, execution = map(read, paths)
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and result["plan_sha256"] == sha(paths[0]),
            "Incomplete or changed execution: " + stage)
    verify_inputs(plan)
    bindings.update(plan["source_bindings"])
    bindings.update({str(p): sha(p) for p in paths})
    return plan, result


def compare_windows(reference, candidate):
    """Keep every paired one-second cell; make absolute levels and tails explicit."""
    rows = []
    for a, b in zip(reference, candidate, strict=True):
        require(a["index"] == b["index"] and a["name"] == b["name"] and a["intervals"] == b["intervals"],
                "Window track differs")
        for view in ("vocals_only", "instrumental"):
            for x, y in zip(a["views"][view]["windows"], b["views"][view]["windows"], strict=True):
                geometry = ("excerpt_index", "physical_start", "physical_end", "input_rms_dbfs", "input_active")
                require(all(x[k] == y[k] for k in geometry), "Window input or support differs")
                cells = {}
                for stem in STEMS:
                    left, right = x["per_stem"][stem], y["per_stem"][stem]
                    require(all(left[k] == right[k] for k in ("desired_active", "off_target")),
                            "Window desired-source support differs")
                    metrics = {}
                    for field in ("output_rms_dbfs", "output_to_input_db", "signed_desired_projection_gain"):
                        u, v = left[field], right[field]
                        require((u is None) == (v is None) and (u is None or (math.isfinite(u) and math.isfinite(v))),
                                "Window metric support differs or is nonfinite")
                        metrics[field] = {"reference": u, "candidate": v, "delta": None if u is None else v - u}
                    cells[stem] = {"desired_active": left["desired_active"], "off_target": left["off_target"],
                                   "metrics": metrics}
                rows.append({"index": a["index"], "track": a["name"], "view": view,
                             **{k: x[k] for k in geometry}, "per_stem": cells})
    eligible = [r for r in rows if r["view"] == "instrumental" and r["input_active"]]
    worst = {}
    for field in ("output_rms_dbfs", "output_to_input_db"):
        for column in ("reference", "candidate", "delta"):
            worst[field + "_" + column] = sorted(eligible,
                key=lambda r: r["per_stem"]["vocals"]["metrics"][field][column], reverse=True)[:20]
    return {"all_windows": rows, "instrumental_input_active_windows": len(eligible),
            "worst_instrumental_vocal_windows": worst,
            "delta_sign": "Candidate minus reference; positive unwanted output means more leakage.",
            "window_units": "Physical start/end are samples at 44100 Hz; native output is dBFS."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="List missing evidence without creating files")
    parser.add_argument("--output-prefix", default="attention-qkv-int8-review-001")
    args = parser.parse_args()
    missing = missing_inputs()
    if args.preflight:
        print(json.dumps({"status": "pending" if missing else "files_present_unverified", "missing": missing}))
        return
    require(not missing, "Finish all evaluation processes and record their actual exits first: " + str(missing))
    require(args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid prefix")
    out = FOLLOWUP / args.output_prefix
    require(not out.exists(), "Preserve prior reviews")
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.evaluate_latency58_deployed_vocal_views import aggregate_reports
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.run_latency58_deployed_vocal_views import require_cpu
    require_cpu()
    paths = [Path(__file__).resolve(), POLICY, ROOT / "research/evaluate.py", ROOT / "research/metrics.py", ROOT / "research/direct/latency58_m4_followup_budget.py", ROOT / "research/direct/latency58_weighted_storage.py"]
    paths.extend(ROOT / "research/direct" / name for name in (
        "compare.py", "compare_latency58_vocal_views.py", "evaluate_latency58_deployed_vocal_views.py",
        "report_latency58_vocal_focus.py", "run_latency58_deployed_vocal_views.py",
        "run_latency58_quality.py", "train_latency58.py"))
    bindings = {str(p): sha(p) for p in paths}
    completed = {stage: load_completed(stage, bindings) for stage in dict.fromkeys(STAGES.values())}
    stages = {key: completed[stage] for key, stage in STAGES.items()}
    fplan, original = stages["full_reference"]
    vplan, prior_views = stages["vocal_reference"]
    baseline = {**prior_views, "tracks": prior_views["vocal_views"]["tracks"],
        "aggregate": prior_views["vocal_views"]["aggregate"],
        "excerpt_count_per_view": prior_views["counterfactual_excerpt_count_per_view"]}
    cplan, changed = stages["candidate"]
    nplan, native = stages["native"]
    require(original["graph_sha256"] == baseline["checkpoint"]["sha256"] == RELEASED
            and changed["graph_sha256"] == CANDIDATE
            and nplan["models"]["released"]["sha256"] == RELEASED
            and nplan["models"]["candidate"]["sha256"] == CANDIDATE, "Evidence graph identity differs")
    require(all(vplan[k] == cplan[k] for k in ("manifest", "config", "track_indices", "track_intervals",
                "protocol_version", "interface", "views", "host_queue_samples", "graph_delay_samples"))
            and cplan["host_queue_samples"] + cplan["graph_delay_samples"] == 256,
            "Scoring or latency protocol differs")
    reference, candidate = original["results"][0], changed["results"][0]
    require(reference["checkpoint"] == fplan["checkpoint"] and candidate["checkpoint"] == cplan["checkpoint"],
            "Result graph differs from plan")
    require(original["track_count"] == changed["track_count"] == baseline["track_count"] == 14
            and original["excerpt_count"] == changed["excerpt_count"] == baseline["excerpt_count_per_view"] == 28,
            "Incomplete original protocol")
    require(len(reference["tracks"]) == len(candidate["tracks"]) == len(baseline["tracks"])
            == len(changed["vocal_views"]["tracks"]) == 14, "Incomplete track arrays")
    for current in (reference, candidate):
        require_close(current["aggregate"], legacy._aggregate_tracks(current["tracks"]))
    views = []
    for plan, tracks, aggregate in ((vplan, baseline["tracks"], baseline["aggregate"]),
            (cplan, changed["vocal_views"]["tracks"], changed["vocal_views"]["aggregate"])):
        require([t["index"] for t in tracks] == list(range(14)), "Wrong vocal track order")
        require_close(aggregate, aggregate_reports(tracks))
        views.append({"version": plan["protocol_version"], "model": plan["checkpoint"],
                      "tracks": tracks, "aggregate": aggregate})
    for index, (a, b) in enumerate(zip(reference["tracks"], candidate["tracks"], strict=True)):
        require(a["excerpts"] == b["excerpts"] == cplan["track_intervals"][str(index)]["intervals"],
                "Original full-mixture interval or alignment changed")
        for key, expected in (("vocal_reference", baseline["tracks"][index]), ("candidate", None)):
            path = PHASE / STAGES[key] / ("track-%02d.json" % index)
            track = read(path)
            if key == "candidate":
                require(track["full_mixture"] == b and track["vocal_views"] == views[1]["tracks"][index],
                        "Candidate per-track evidence differs from final report")
            else:
                require(track["full_mixture"] == a and track["vocal_views"] == expected, "Baseline per-track evidence differs from final report")
            bindings[str(path)] = sha(path)
    full = compare(reference, candidate)
    cells = music_cells(reference, candidate)
    require_close(full, changed["parent_sixteen_projection_comparison"], "stored parent comparison")
    require_close(cells, changed["parent_sixteen_projection_all_track_stem_cells"], "stored parent cells")
    vocal = compare_reports(*views)
    windows = compare_windows(views[0]["tracks"], views[1]["tracks"])
    summary = {}
    for metric in next(iter(next(iter(cells.values())).values()))["metrics"]:
        eligible = [{"track": track, "stem": stem, **cell["metrics"][metric]}
                    for track, stems in cells.items() for stem, cell in stems.items()
                    if cell["metrics"][metric]["delta"] is not None]
        more_is_worse = metric == "absent_fp_dbfs"
        summary[metric] = {"eligible_cells": len(eligible),
            "regressing_cells": sum((r["delta"] > 0 if more_is_worse else r["delta"] < 0) for r in eligible),
            "maximum_absolute_change": max((abs(r["delta"]) for r in eligible), default=None),
            "worst_changes": sorted(eligible, key=lambda r: r["delta"], reverse=more_is_worse)[:10]}
    require(len(windows["all_windows"]) == 840 and sum(len(stems) for stems in cells.values()) == 56,
            "Incomplete 56-cell or 840-window comparison")
    before = snapshot()
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    plan = {"schema": "latency58-attention-qkv-int8-review-plan-v1", "source_bindings": bindings,
            "input_stages": STAGES, "budget_before": before, "quality_selection": False,
            "output_allowance_bytes": 5_000_000}
    write(out / "plan.json", plan)
    result = {"schema": "latency58-attention-qkv-int8-review-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(out / "plan.json"),
        "source_bindings_unchanged": True, "full_mixture": full, "all_track_stem_band_absence_cells": cells,
        "music_regression_summary": summary, "vocal_views": vocal, "paired_windows": windows,
        "native_cost": {**{k: native[k] for k in ("median_of_block_p50_ms", "candidate_over_released_ratio",
            "system", "cpu", "background_load_declared", "limitations", "relative_speed_gate")}},
        "full_band_target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.,
        "target_gap_db": max(0., 5. - candidate["aggregate"]["full_sdr_db"]),
        "graph_plus_host_delay_samples": 256, "quality_selected": False, "native_host_qualified": False,
        "human_listening_completed": False, "goal_complete": False, "gpu_used": False,
        "budget_before": before, "budget_after": snapshot(),
        "limitations": ["The pass status describes completed evidence checks, not goal acceptance.",
            "All quality observations use the existing development panel, without new confirmation material.",
            "Counterfactual references follow source assignments and may contain recording bleed.",
            "Read unwanted vocal levels with desired-vocal gain and Other quality; attenuation alone is insufficient.",
            "Native cost was measured locally under concurrent load; target-M4 AU and DAW evidence is outstanding."]}
    require(len(json.dumps(result, indent=2).encode()) + (out / "plan.json").stat().st_size + 100_000
            < plan["output_allowance_bytes"], "Review output allowance exceeded")
    verify_inputs(plan)
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_mixture": full["metrics"]["full_sdr_db"],
                      "instrumental_vocals": vocal["views"]["instrumental"]["per_stem"]["vocals"],
                      "quality_selected": False, "native_host_qualified": False}))


if __name__ == "__main__":
    main()
