"""Compare complete vocal-focus quality evidence with three retained baselines."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed, probe_comparison
from research.direct.train_latency58 import verify_inputs

REFERENCES = {"working": "teacher-half250", "drum500": "sdr-drum-accum-500", "drum1000": "sdr-drum-accum-1000"}
STEMS = ("drums", "bass", "vocals", "other")


def music_cells(reference, candidate):
    cells = {}
    for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
        require(left["name"] == right["name"] and left["excerpts"] == right["excerpts"], "Physical excerpts differ")
        stems = {}
        for stem in STEMS:
            a, b = left["per_stem"][stem], right["per_stem"][stem]
            require(all(a[k] == b[k] for k in ("active_windows", "absent_windows"))
                    and a["band_sdr_db"].keys() == b["band_sdr_db"].keys(), "Source activity or band support differs")
            pairs = {k: (a[k], b[k]) for k in ("full_sdr_db", "sir_db", "absent_fp_dbfs")}
            pairs.update({band: (a["band_sdr_db"][band], b["band_sdr_db"][band]) for band in a["band_sdr_db"]})
            values = {}
            for metric, (x, y) in pairs.items():
                require((x is None) == (y is None) and (x is None or (math.isfinite(x) and math.isfinite(y))),
                        "Metric support differs or is nonfinite")
                values[metric] = {"reference": x, "candidate": y, "delta": None if x is None else y - x}
            stems[stem] = {"active_windows": b["active_windows"], "absent_windows": b["absent_windows"], "metrics": values}
        require(left["name"] not in cells, "Duplicate music track")
        cells[left["name"]] = stems
    return cells


def load_views(directory, evidence):
    paths = [directory / name for name in ("plan.json", "result.json", "execution.json")]
    plan, report, execution = [read(p) for p in paths]
    require(report["schema"] == "latency58-vocal-views-evaluation-result-v1" and report["status"] == "pass"
            and report["source_bindings_unchanged"] and not report["cuda_initialized"]
            and report["training_updates_executed"] == 0 and not report["confirmation_material_used"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and report["plan_sha256"] == execution["plan_sha256"] == sha(paths[0])
            and report["model"] == plan["model"], "Vocal views are not completely authenticated")
    verify_inputs(plan)
    evidence.update({str(p): sha(p) for p in paths})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require frozen CPU1 summary")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-vocal-focus-summary-plan-v1", "Unexpected summary plan")
    prefix = plan["candidate_prefix"]
    require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid candidate prefix")
    out = Path(plan["output_directory"])
    require(out.is_relative_to(PHASE) and out.is_dir() and not (out / "result.json").exists(), "Preserve summary")
    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.latency58_vocal_focus_checkpoint import read_generation, require_space
    evidence, candidates, quality_plans = {}, {}, {}
    for mode in ("full14", "actions60", "probes"):
        quality_plans[mode], candidates[mode] = load_completed(PHASE / (prefix + "-" + mode + "-001"), evidence)
    training_binding = quality_plans["full14"]["training_plan"]
    require(sha(training_binding["path"]) == training_binding["sha256"], "Training plan changed")
    training = read(training_binding["path"])
    require(training["schema"] == "latency58-vocal-focus-training-v1" and not training["resource_only"]
            and all(q["training_plan"] == training_binding and q["step"] == 250
                    and q["generation"] == quality_plans["full14"]["generation"] for q in quality_plans.values()),
            "Quality bundle contains different pilot endpoints")
    verify_inputs(training)
    before = require_space(training, 3_000_000)
    receipt = read_generation(quality_plans["full14"]["generation"], expected_plan_sha=training_binding["sha256"],
                              require_optimizer=False)
    candidate_views = load_views(PHASE / (prefix + "-views-001"), evidence)
    fingerprint = receipt["model_state_sha256"]
    require(candidate_views["model"]["model_state_sha256"] == candidates["probes"]["model_state_sha256"] == fingerprint
            and all(candidates[m]["results"][0]["model"]["model_state_sha256"] == fingerprint
                    for m in ("full14", "actions60")), "Pilot model identity differs across metrics")
    comparisons, reference_states = {}, {}
    for label, reference_prefix in REFERENCES.items():
        compared, states = {}, set()
        for mode in ("full14", "actions60", "probes"):
            _, reference = load_completed(PHASE / (reference_prefix + "-" + mode + "-001"), evidence,
                                          canonical_baseline=label == "working")
            candidate = candidates[mode]
            if mode == "probes":
                require(candidate["status"] == reference["status"] == "pass"
                        and candidate["source_bindings_unchanged"] and reference["source_bindings_unchanged"], "Probe failed")
                compared[mode] = probe_comparison(reference, candidate)
                states.add(reference["model_state_sha256"])
            else:
                require(candidate["inputs_unchanged"] and reference["inputs_unchanged"]
                        and all(candidate[k] == reference[k] for k in
                                ("manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                        "Music protocol differs")
                compared[mode] = compare(reference["results"][0], candidate["results"][0])
                compared[mode + "_all_track_stem_band_absence"] = music_cells(reference["results"][0], candidate["results"][0])
                states.add(reference["results"][0]["model"]["model_state_sha256"])
        views = load_views(PHASE / ("vocal-views-" + label + "-001"), evidence)
        states.add(views["model"]["model_state_sha256"])
        require(len(states) == 1, "Reference bundle contains different models")
        reference_states[label] = next(iter(states))
        compared["vocal_views"] = compare_reports(views, candidate_views)
        comparisons[label] = compared
    require(reference_states["working"] == training["parent"]["model_state_sha256"], "Working reference differs from parent")
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound completed score evidence")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-vocal-focus-quality-summary-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "training_plan": training_binding, "arm": training["arm"], "step": 250,
        "model_state_sha256": fingerprint, "reference_model_states": reference_states,
        "full_mixture_aggregate": candidates["full14"]["results"][0]["aggregate"],
        "vocal_views_aggregate": candidate_views["aggregate"], "comparisons": comparisons,
        "all_metrics_compared": True, "quality_selected": False, "human_listening_completed": False,
        "counted_bytes_before": before, "counted_bytes_after": require_space(training, 0),
        "limitations": ["Development-panel comparisons; reserved confirmation material is excluded.",
                        "Vocal counterfactuals do not establish behavior under full mixtures or human listening quality.",
                        "Native unwanted levels must be considered with wanted-source fidelity and gain; attenuation is not success.",
                        "One seed; track intervals omit seed uncertainty and repeated selection effects.",
                        "One Actions excerpt gives no sampling-uncertainty estimate; probe metrics do not establish audibility."]})
    print({"status": "pass", "arm": training["arm"], "step": 250, "model_state_sha256": fingerprint}, flush=True)


if __name__ == "__main__":
    main()
