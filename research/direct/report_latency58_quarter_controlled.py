"""Compare reduced forced views with the matched control and completed rate sweep."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed, probe_comparison
from research.direct.train_latency58 import verify_inputs

from research.direct.report_latency58_vocal_focus import music_cells, load_views

REFERENCES = {"working": "teacher-half250", "parent": "leader-cleanup-250",
              "matched_control": "cleanup-successor-250", "higher_rate": "cleanup-rebound-250",
              "lower_rate": "cleanup-lr3e6-250"}


def reference_views(label):
    return PHASE / {"working": "vocal-views-working-001", "parent": "leader-cleanup-250-views-001",
                    "matched_control": "cleanup-successor-250-views-001", "higher_rate": "cleanup-rebound-250-views-001",
                    "lower_rate": "cleanup-lr3e6-250-views-001"}[label]


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
    require(plan["schema"] == "latency58-quarter-controlled-summary-plan-v1", "Unexpected summary plan")
    prefix = plan["candidate_prefix"]
    require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid candidate prefix")
    out = Path(plan["output_directory"])
    require(out.is_relative_to(PHASE) and out.is_dir() and not (out / "result.json").exists(), "Preserve summary")
    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.latency58_quarter_controlled_checkpoint import read_generation, require_space
    evidence, candidates, quality_plans = {}, {}, {}
    for mode in ("full14", "actions60", "probes"):
        quality_plans[mode], candidates[mode] = load_completed(PHASE / (prefix + "-" + mode + "-001"), evidence)
    training_binding = quality_plans["full14"]["training_plan"]
    require(sha(training_binding["path"]) == training_binding["sha256"], "Training plan changed")
    training = read(training_binding["path"])
    require(training["schema"] == "latency58-quarter-controlled-training-v1" and not training["resource_only"]
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
            reference_plan, reference = load_completed(PHASE / (reference_prefix + "-" + mode + "-001"), evidence,
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
        views = load_views(reference_views(label), evidence)
        states.add(views["model"]["model_state_sha256"])
        require(len(states) == 1, "Reference bundle contains different models")
        reference_states[label] = next(iter(states))
        compared["vocal_views"] = compare_reports(views, candidate_views)
        comparisons[label] = compared
    require(reference_states["parent"] == training["parent"]["model_state_sha256"],
            "Selected-parent reference differs from training initialization")
    from research.direct.latency58_quarter_controlled_checkpoint import validate_journal
    from research.direct.train_latency58 import load_source
    helpers = load_source("quarter_controlled_summary_helpers", training["helper_source"])
    journal_path = Path(quality_plans["full14"]["generation"]) / "metrics.jsonl"
    matched_rows = validate_journal(journal_path.read_bytes(), 250, training, helpers)
    control_training = read(training["matched_control_training_plan"]["path"])
    control_receipt = read(Path(control_training["run_dir"]) / "checkpoints/step-000250/receipt.json")
    require(reference_states["matched_control"] == control_receipt["model_state_sha256"],
            "View-frequency reference is not the matched training control")
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound completed score evidence")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-quarter-controlled-quality-summary-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "training_plan": training_binding, "arm": training["arm"], "step": 250,
        "teacher_mode": training["teacher_mode"], "additional_loss_weight": training["additional_loss_weight"],
        "additional_loss_version": training["additional_loss_version"],
        "model_state_sha256": fingerprint, "reference_model_states": reference_states,
        "full_mixture_aggregate": candidates["full14"]["results"][0]["aggregate"],
        "vocal_views_aggregate": candidate_views["aggregate"], "comparisons": comparisons,
        "all_metrics_compared": True, "comparison_variable": "matched_controlled_view_frequency",
        "controlled_examples_per_update": 4, "ordinary_examples_per_update": 12,
        "shared_input_and_teacher_microbatches": len(matched_rows) * 2,
        "restored_ordinary_microbatches": len(matched_rows) * 2,
        "matched_training_effect_claimed": True, "matched_comparison_scope": "Controlled-view frequency, including its teacher and auxiliary participation; one seed and the same 4000 pristine crops and original augmentation RNG", "matched_pristine_original_augmentation_rng_updates": len(matched_rows), "quality_selected": False, "human_listening_completed": False,
        "counted_bytes_before": before, "counted_bytes_after": require_space(training, 0),
        "limitations": ["Development-panel comparisons; old confirmation intervals have already informed this research and cannot serve as unseen confirmation.",
                        "Vocal counterfactuals do not establish behavior under full mixtures or human listening quality.",
                        "Native unwanted levels must be considered with wanted-source fidelity and gain; attenuation is not success.",
                        "One seed; track intervals omit seed uncertainty and repeated selection effects.",
                        "One Actions excerpt gives no sampling-uncertainty estimate; probe metrics do not establish audibility."]})
    print({"status": "pass", "arm": training["arm"], "step": 250, "model_state_sha256": fingerprint}, flush=True)


if __name__ == "__main__":
    main()
