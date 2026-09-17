"""Record a complete pilot quality review and close its fixed training schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def completed_endpoint(plan):
    """Authenticate the original training, saved-state audit and quality summary."""
    from research.direct.latency58_controlled_deployed_checkpoint import read_generation, validate_recipe
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
    require(summary["schema"] == "latency58-controlled-deployed-quality-summary-v1" and summary["status"] == "pass"
            and summary["source_bindings_unchanged"] and summary["all_metrics_compared"]
            and summary["arm"] == training["arm"] and summary["step"] == 250
            and summary["training_plan"] == plan["training_plan"]
            and summary["model_state_sha256"] == receipt["model_state_sha256"]
            and not summary["quality_selected"] and not summary["human_listening_completed"]
            and set(summary["comparisons"]) == {"working", "drum500", "drum1000", "focused_control", "ordinary_control"}
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
            and audit["selected_teacher_participation_and_batch_divisor_verified"]
            and audit["teacher_mode"] == summary["teacher_mode"] == training["teacher_mode"] == "ordinary_only"
            and audit["controlled_deployed_truth_and_full_batch_divisor_verified"]
            and audit["additional_loss_weight"] == summary["additional_loss_weight"] == training["additional_loss_weight"] == .5
            and audit["additional_loss_version"] == summary["additional_loss_version"] == training["additional_loss_version"]
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
    completed_supplements(plan, summary, receipt)
    completed_playback(plan, receipt)
    return training, generation, receipt, summary


def completed_supplements(plan, summary, receipt):
    """Require the actual matched training result and all three complete quiet comparisons."""
    from research.direct.audit_latency58_controlled_deployed_training_match import load_completed_match
    from research.direct.compare_latency58_quiet_wanted import load_completed, compare_reports
    evidence = {}
    matched = load_completed_match(summary["training_match"], evidence)
    require(matched["model_states"] == {"reference": summary["reference_model_states"]["ordinary_control"],
                                        "candidate": receipt["model_state_sha256"]},
            "Review does not describe the authenticated matched pair")
    require(set(plan["quiet_comparisons"]) == {"working", "focused_control", "ordinary_control"}, "Incomplete quiet reference scope")
    for label, binding in plan["quiet_comparisons"].items():
        directory = Path(binding["path"]).parent
        paths = [directory / name for name in ("plan.json", "result.json", "comparison-execution.json")]
        require(binding["path"] == str(paths[1]) and binding["sha256"] == sha(paths[1]), "Quiet comparison changed")
        comparison_plan, result, execution = [read(path) for path in paths]
        require(comparison_plan["schema"] == "latency58-quiet-wanted-comparison-plan-v1"
                and result["schema"] == "latency58-quiet-wanted-comparison-v1" and result["status"] == "pass"
                and result["source_bindings_unchanged"] and result["source_bindings"] == comparison_plan["source_bindings"]
                and result["all_1680_source_windows_matched"] and not result["inference_executed"]
                and not result["primary_protocol_changed"] and not result["confirmation_material_used"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"]
                and execution["plan_sha256"] == result["plan_sha256"] == sha(paths[0]),
                "Quiet comparison did not complete under the frozen protocol")
        verify_inputs(comparison_plan)
        evidence.update(comparison_plan["source_bindings"])
        evidence.update({str(path): sha(path) for path in paths})
        reference = load_completed(Path(comparison_plan["reference_directory"]), evidence)
        candidate = load_completed(Path(comparison_plan["candidate_directory"]), evidence)
        require(reference["model"]["model_state_sha256"] == summary["reference_model_states"][label]
                and candidate["model"]["model_state_sha256"] == receipt["model_state_sha256"], "Different quiet model pair")
        independently_compared = compare_reports(reference, candidate)
        require(all(result[key] == value for key, value in independently_compared.items()),
                "Quiet paired values do not reproduce their complete raw measurements")
    gain = plan["gain_supplement"]
    gain_directory = Path(gain["path"]).parent
    gain_paths = [gain_directory / name for name in ("plan.json", "result.json", "gain-error-execution.json")]
    require(gain["path"] == str(gain_paths[1]) and gain["sha256"] == sha(gain_paths[1]), "Gain supplement changed")
    gain_plan, gain_result, gain_execution = [read(path) for path in gain_paths]
    require(gain_result["schema"] == "latency58-controlled-deployed-gain-error-v1" and gain_result["status"] == "pass"
            and gain_result["source_bindings_unchanged"] and gain_result["source_bindings"] == gain_plan["source_bindings"]
            and gain_execution["actual_exit_code"] == 0 and not gain_execution["timed_out"]
            and gain_execution["source_bindings_unchanged"]
            and gain_execution["plan_sha256"] == gain_result["plan_sha256"] == sha(gain_paths[0])
            and not gain_result["inference_executed"] and not gain_result["primary_protocol_changed"]
            and set(gain_result["comparisons"]) == set(summary["reference_model_states"]), "Incomplete gain-error supplement")
    for label, comparison in gain_result["comparisons"].items():
        require(comparison["candidate_model"]["model_state_sha256"] == receipt["model_state_sha256"]
                and comparison["reference_model"]["model_state_sha256"] == summary["reference_model_states"][label]
                and comparison["all_input_streams_and_window_support_exact"]
                and len(comparison["tracks"]) == 14 and len(comparison["windows"]) == 319,
                "Gain supplement describes a different model or window scope")
    verify_inputs(gain_plan)
    evidence.update(gain_plan["source_bindings"])
    evidence.update({str(path): sha(path) for path in gain_paths})
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound supplementary evidence")


def completed_playback(plan, receipt):
    """Verify the Actions workflow actually passed native decoding and browser replay."""
    evidence = {}
    item = plan["listening_workflow"]
    paths = [Path(item["path"]).parent / name for name in ("plan.json", "result.json", "workflow-execution.json")]
    require(item["path"] == str(paths[1]) and item["sha256"] == sha(paths[1]), "Listening workflow changed")
    workflow_plan, workflow, execution = [read(p) for p in paths]
    require(workflow["schema"] == "latency58-controlled-deployed-listening-workflow-v1"
            and workflow["status"] == "pass" and workflow["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == workflow["plan_sha256"] == sha(paths[0]),
            "Native listening workflow did not complete")
    verify_inputs(workflow_plan)
    evidence.update(workflow_plan["source_bindings"])
    evidence.update({str(p): sha(p) for p in paths})
    for key in ("player", "browser", "inventory"):
        bound = workflow[key]
        require(sha(bound["path"]) == bound["sha256"], "Native playback evidence changed")
        evidence[bound["path"]] = bound["sha256"]
    player, browser, inventory = [read(workflow[k]["path"]) for k in ("player", "browser", "inventory")]
    require(player["status"] == browser["status"] == inventory["status"] == "pass"
            and player["models"]["controlled_deployed"]["model_state_sha256"] == receipt["model_state_sha256"]
            and player["all_decoded_track_and_aggregate_scores_exact"] and player["shared_reference_samples_exact"]
            and len(inventory["files"]) == len(player["audio_files"]) == 17
            and inventory["all_served_bytes_exact"] and inventory["all_byte_ranges_exact"]
            and browser["temporary_profile_removed"] and browser["checks"]["browser_audio_muted"]
            and not workflow["human_listening_completed"] and not workflow["quality_selected"],
            "Native playback did not preserve source samples or complete its controls")
    verify_inputs(inventory)
    evidence.update(inventory["source_bindings"])
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Review omitted native playback inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Review plan or cwd changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-controlled-deployed-review-plan-v1", "Unexpected review plan")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve endpoint review")
    review_fields = ("full_mixture", "per_track_stem_bands_and_absence", "probes_and_dc", "actions",
                     "instrumental_vocal_output", "vocals_only_other_output", "wanted_fidelity_and_gain",
                     "local_windows", "quiet_wanted_sources", "matched_deployed_truth_change", "native_actions_playback", "interpretation", "limitations")
    require(set(plan["review"]) == set(review_fields)
            and all(isinstance(plan["review"][key], str) and plan["review"][key].strip() for key in review_fields)
            and plan["all_quality_metrics_reviewed"] is True
            and plan["further_optimizer_updates"] == 0 and not plan["quality_selected"],
            "Missing substantive complete review or different training decision")
    training, generation, receipt, summary = completed_endpoint(plan)
    from research.direct.latency58_controlled_deployed_checkpoint import require_space
    counted = require_space(training, 3_000_000)
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-controlled-deployed-review-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": {**plan["source_bindings"], str(args.plan): args.plan_sha256},
        "source_bindings_unchanged": True, "arm": training["arm"], "training_plan": plan["training_plan"],
        "generation": str(generation), "model_state_sha256": receipt["model_state_sha256"],
        "completed_step": 250, "original_maximum_step": 250, "training_closed": True,
        "further_optimizer_updates": 0, "teacher_mode": training["teacher_mode"],
        "additional_loss_weight": training["additional_loss_weight"],
        "additional_loss_version": training["additional_loss_version"],
        "listening_workflow": plan["listening_workflow"],
        "summary": plan["summary"], "comparisons": summary["comparisons"],
        "quiet_comparisons": plan["quiet_comparisons"], "training_match": summary["training_match"],
        "gain_supplement": plan["gain_supplement"],
        "review": plan["review"], "all_quality_metrics_reviewed": True,
        "decision": "Close this pilot at its original 250-update horizon; retain its inference model for matched comparison.",
        "quality_selected": False, "human_listening_completed": False, "goal_complete": False,
        "current_counted_bytes": counted})
    print({"status": "pass", "arm": training["arm"], "training_closed": True, "quality_selected": False}, flush=True)


if __name__ == "__main__":
    main()
