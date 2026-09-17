"""Record a complete pilot quality review and close its fixed training schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def completed_endpoint(plan):
    """Authenticate the original training, saved-state audit and quality summary."""
    from research.direct.latency58_vocal_focus_checkpoint import read_generation, validate_recipe
    for key in ("training_plan", "summary", "summary_execution", "audit", "audit_execution", "training_execution"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound endpoint prerequisite")
    training = read(plan["training_plan"]["path"])
    validate_recipe(training)
    verify_inputs(training)
    require(not training["resource_only"] and training["arm"] == plan["arm"], "Different production pilot")
    generation = Path(training["run_dir"]) / "checkpoints/step-000250"
    require(generation.is_relative_to(PHASE), "Different generation root")
    receipt = read_generation(generation, expected_plan_sha=plan["training_plan"]["sha256"], require_optimizer=False)
    summary, summary_execution, audit, audit_execution, execution = (
        read(plan[key]["path"]) for key in ("summary", "summary_execution", "audit", "audit_execution", "training_execution"))
    summary_plan_path = Path(plan["summary"]["path"]).parent / "plan.json"
    require(plan["source_bindings"].get(str(summary_plan_path)) == sha(summary_plan_path), "Unbound summary plan")
    summary_plan = read(summary_plan_path)
    verify_inputs(summary_plan)
    verify_inputs(summary)
    require(summary["schema"] == "latency58-vocal-focus-quality-summary-v1" and summary["status"] == "pass"
            and summary["source_bindings_unchanged"] and summary["all_metrics_compared"]
            and summary["arm"] == training["arm"] and summary["step"] == 250
            and summary["training_plan"] == plan["training_plan"]
            and summary["model_state_sha256"] == receipt["model_state_sha256"]
            and not summary["quality_selected"] and not summary["human_listening_completed"]
            and set(summary["comparisons"]) == {"working", "drum500", "drum1000"}
            and summary_execution["actual_exit_code"] == 0 and not summary_execution["timed_out"]
            and summary_execution["source_bindings_unchanged"]
            and summary_execution["plan_sha256"] == summary["plan_sha256"] == sha(summary_plan_path),
            "Complete quality summary did not finish successfully")
    require(all(plan["source_bindings"].get(p) == s for p, s in summary["source_bindings"].items()),
            "Review omitted completed quality evidence")
    for comparison in summary["comparisons"].values():
        require(set(comparison) == {"full14", "actions60", "probes", "vocal_views",
                                    "full14_all_track_stem_band_absence", "actions60_all_track_stem_band_absence"}
                and len(comparison["full14_all_track_stem_band_absence"]) == 14
                and len(comparison["actions60_all_track_stem_band_absence"]) == 1,
                "Incomplete primary or diagnostic comparison")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    status_path = Path(training["run_dir"]) / "status.json"
    status = read(status_path)
    require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and audit["arm"] == training["arm"] and audit["step"] == receipt["step"] == 250
            and audit["matched_input_journal_verified"] and audit["normalized_drum_objective_journal_verified"]
            and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit["model_state_sha256"] == receipt["model_state_sha256"]
            and audit["plan_sha256"] == audit_execution["plan_sha256"] == execution["plan_sha256"]
            == plan["training_plan"]["sha256"]
            and audit_execution["actual_exit_code"] == execution["actual_exit_code"] == 0
            and not audit_execution["timed_out"] and audit_execution["source_bindings_unchanged"]
            and execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"]
            and status["status"] == "complete" and status["step"] == 250,
            "Pilot lacks its original saved-state audit or clean monitored completion")
    for path in (monitor_path, status_path, *(generation / name for name in ("receipt.json", "model.pt", "rng.pt", "metrics.jsonl"))):
        require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound retained endpoint")
    return training, generation, receipt, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Review plan or cwd changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-review-plan-v1", "Unexpected review plan")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve endpoint review")
    review_fields = ("full_mixture", "per_track_stem_bands_and_absence", "probes_and_dc", "actions",
                     "instrumental_vocal_output", "vocals_only_other_output", "wanted_fidelity_and_gain",
                     "local_windows", "interpretation", "limitations")
    require(set(plan["review"]) == set(review_fields)
            and all(isinstance(plan["review"][key], str) and plan["review"][key].strip() for key in review_fields)
            and plan["all_quality_metrics_reviewed"] is True
            and plan["further_optimizer_updates"] == 0 and not plan["quality_selected"],
            "Missing substantive complete review or different training decision")
    training, generation, receipt, summary = completed_endpoint(plan)
    from research.direct.latency58_vocal_focus_checkpoint import require_space
    counted = require_space(training, 3_000_000)
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-vocal-focus-review-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": {**plan["source_bindings"], str(args.plan): args.plan_sha256},
        "source_bindings_unchanged": True, "arm": training["arm"], "training_plan": plan["training_plan"],
        "generation": str(generation), "model_state_sha256": receipt["model_state_sha256"],
        "completed_step": 250, "original_maximum_step": 250, "training_closed": True,
        "further_optimizer_updates": 0, "summary": plan["summary"], "comparisons": summary["comparisons"],
        "review": plan["review"], "all_quality_metrics_reviewed": True,
        "decision": "Close this pilot at its original 250-update horizon; retain its inference model for matched comparison.",
        "quality_selected": False, "human_listening_completed": False, "goal_complete": False,
        "current_counted_bytes": counted})
    print({"status": "pass", "arm": training["arm"], "training_closed": True, "quality_selected": False}, flush=True)


if __name__ == "__main__":
    main()
