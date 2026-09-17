"""Retire a reviewed terminal Adam file and four rejected, reproducible estimate WAVs."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.record_latency58_leader_cleanup_review import completed_endpoint
from research.direct.latency58_leader_cleanup_checkpoint_v2 import read_generation, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Retirement plan or cwd changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-leader-cleanup-artifact-retirement-plan-v1"
            and plan["arm"] == "focused" and plan["teacher_mode"] == "ordinary_only"
            and plan["additional_loss_weight"] == .5
            and plan["reserve_after_bytes"] == 250_000_000,
            "Different bounded pilot retirement")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "receipt.json").exists(), "Preserve retirement evidence")
    for key in ("review_plan", "review", "review_execution", "active_decision"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound retirement prerequisite")
    review_plan, review, execution = (read(plan[key]["path"]) for key in ("review_plan", "review", "review_execution"))
    verify_inputs(review_plan)
    verify_inputs(review)
    require(review["schema"] == "latency58-leader-cleanup-review-v1" and review["status"] == "pass"
            and review["source_bindings_unchanged"] and review["training_closed"]
            and review["completed_step"] == review["original_maximum_step"] == 250
            and review["further_optimizer_updates"] == 0 and review["all_quality_metrics_reviewed"]
            and not review["quality_selected"] and review["review"] == review_plan["review"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == review["plan_sha256"] == plan["review_plan"]["sha256"],
            "Complete quality review has not closed this fixed training schedule")
    require(all(plan["source_bindings"].get(p) == s for p, s in review["source_bindings"].items()),
            "Retirement omitted reviewed evidence")
    training, generation, receipt, _ = completed_endpoint(review_plan)
    require(review["teacher_mode"] == plan["teacher_mode"] == training["teacher_mode"]
            and review["arm"] == plan["arm"] == training["arm"]
            and review["additional_loss_weight"] == plan["additional_loss_weight"] == training["additional_loss_weight"]
            and review["additional_loss_version"] == training["additional_loss_version"]
            and review["model_state_sha256"] == receipt["model_state_sha256"]
            and review["generation"] == str(generation), "Different reviewed endpoint")
    # Read with the optimizer required before deleting it; the original CPU audit
    # has already checked all saved Adam tensors against this same receipt.
    require(read_generation(generation, expected_plan_sha=review["training_plan"]["sha256"]) == receipt,
            "Terminal Adam or retained generation changed")
    active = read(plan["active_decision"]["path"])
    verify_inputs(active)
    require(active["schema"] == "latency58-leader-cleanup-decision-v1"
            and active["quality_endpoints"] == [250] and not active["automatic_continuation"]
            and active["maximum_production_updates"] == 250
            and active["additional_loss_weight"] == plan["additional_loss_weight"]
            and active["additional_loss_version"] == training["additional_loss_version"]
            and active["reference_training_plan"] == training["reference_training_plan"]
            and active["comparison_variable"] == training["comparison_variable"] == "training_parent"
            and active["matched_loss_effect_from_leader_claimed"] is False
            and active["config"] == training["config"]
            and active["counted_roots"] == training["counted_roots"]
            and active["stop_counted_bytes"] == training["stop_counted_bytes"]
            and training["preparation_decision"] == plan["active_decision"], "Different active pilot decision")
    queue_modules = {
        "leader-cleanup-250-queued-quality-001":
            ("research.direct.queue_latency58_leader_cleanup_quality", "queue-execution.json"),
        "leader-cleanup-supplements-001":
            ("research.direct.run_latency58_leader_cleanup_supplements", "supplements-execution.json"),
        "leader-cleanup-listening-workflow-001":
            ("research.direct.run_latency58_leader_cleanup_listening", "workflow-execution.json"),
    }
    require(set(plan["completed_queues"]) == set(queue_modules),
            "All quality, quiet, gain and native playback work must complete before retirement")
    for name, (module, execution_name) in queue_modules.items():
        directory = PHASE / name
        queue_paths = [directory / leaf for leaf in ("plan.json", "result.json", execution_name)]
        require(plan["completed_queues"][name] == {"path": str(queue_paths[2]), "sha256": sha(queue_paths[2])}
                and all(plan["source_bindings"].get(str(p)) == sha(p) for p in queue_paths), "Unbound queue completion")
        queued_plan, queued_result, queued_execution = [read(p) for p in queue_paths]
        command = queued_execution["argv"]
        require(queued_execution["actual_exit_code"] == 0 and not queued_execution["timed_out"]
                and queued_execution["source_bindings_unchanged"] and queued_result["status"] == "pass"
                and queued_result["source_bindings_unchanged"]
                and command[command.index("-m") + 1] == module
                and queued_execution["plan_sha256"] == queued_result["plan_sha256"] == sha(queue_paths[0]),
                "A scoring queue has not completed under its original plan")
        verify_inputs(queued_plan)
    optimizer = generation / "optimizer.pt"
    expected = plan["optimizer"]
    require(expected["path"] == str(optimizer) and expected["file_binding"] == receipt["files"]["optimizer.pt"],
            "Only this exact terminal Adam file may be retired")
    for document in (plan, review_plan, review, training, active):
        require(str(optimizer) not in document["source_bindings"], "Adam remains an active input")
    protected = {str(generation / name): sha(generation / name)
                 for name in ("model.pt", "rng.pt", "metrics.jsonl", "receipt.json")}
    require(all(plan["source_bindings"].get(p) == s for p, s in protected.items()), "Unbound retained inference state")
    from research.direct.report_latency58_sdr import load_completed
    rejected_dir = PHASE / "sdr-log-relative-250-actions60-001"
    rejected_evidence = {}
    rejected_plan, rejected = load_completed(rejected_dir, rejected_evidence)
    rejected_evidence.update(rejected_plan["source_bindings"])
    decision_path = PHASE / "sdr-log-relative-250-decision-001/decision.json"
    rejected_decision = read(decision_path)
    verify_inputs(rejected_decision)
    rejected_evidence.update(rejected_decision["source_bindings"])
    rejected_evidence[str(decision_path)] = sha(decision_path)
    require(rejected_decision["status"] == "stop_recipe_at_250_without_adoption"
            and not rejected_decision["quality_selected"]
            and rejected_decision["authorized_further_updates_for_this_recipe"] == 0
            and rejected_decision["checkpoint_state_sha256"]
            == rejected["results"][0]["model"]["model_state_sha256"]
            and rejected["inputs_unchanged"] and rejected["retained_audio"]
            and rejected_plan["mode"] == "actions60" and rejected_plan["step"] == 250
            and rejected_plan["physical_reference_intervals_seconds"] == [[60, 75]]
            and plan["rejected_audio_decision"] == {"path": str(decision_path), "sha256": sha(decision_path)}
            and all(plan["source_bindings"].get(p) == s for p, s in rejected_evidence.items()),
            "Recreatable estimates do not belong to the completed, rejected trial")
    estimates = plan["rejected_estimates"]
    expected_paths = {str(rejected_dir / "audio" / ("estimate-" + stem + ".wav"))
                      for stem in ("drums", "bass", "vocals", "other")}
    require(set(estimates) == expected_paths and len(estimates) == 4,
            "Only four rejected estimate WAVs may be retired; retain all source audio")
    import soundfile as sf
    for path, binding in estimates.items():
        file = Path(path)
        info = sf.info(file)
        require(file.is_file() and not file.is_symlink()
                and file.stat().st_size == binding["bytes"] == 5_292_088
                and sha(file) == binding["sha256"]
                and info.samplerate == 44100 and info.channels == 2
                and info.frames == 661500 and info.subtype == "FLOAT",
                "Rejected estimate identity or native format changed")
    retiring = {str(optimizer), *estimates}
    # Completed outputs are recorded by identity, never converted into active
    # source bindings that will break on this authorized terminal retirement.
    for document in (plan, review_plan, review, training, active):
        require(retiring.isdisjoint(document["source_bindings"]), "A retirement target remains an active input")
    freed_bytes = expected["file_binding"]["bytes"] + sum(item["bytes"] for item in estimates.values())
    before = require_space(active, 0)
    require(before - freed_bytes + plan["reserve_after_bytes"] < active["stop_counted_bytes"],
            "Retirement would not leave the bounded listening and export reserve")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "optimizer": expected,
                               "protected_files": protected, "rejected_estimates": estimates,
                               "recreation_plan": str(rejected_dir / "plan.json"), "counted_bytes_before": before})
    require(optimizer.is_file() and not optimizer.is_symlink()
            and optimizer.stat().st_size == expected["file_binding"]["bytes"]
            and sha(optimizer) == expected["file_binding"]["sha256"], "Adam changed immediately before retirement")
    for path, binding in estimates.items():
        require(Path(path).is_file() and not Path(path).is_symlink()
                and sha(path) == binding["sha256"], "Estimate changed immediately before retirement")
    optimizer.unlink()
    for path in sorted(estimates):
        Path(path).unlink()
    verify_inputs(plan)
    verify_inputs(review_plan)
    verify_inputs(active)
    verify_inputs(rejected_plan)
    verify_inputs(rejected_decision)
    require(all(not Path(p).exists() for p in retiring)
            and all(sha(p) == s for p, s in protected.items()), "Retained state changed")
    after = require_space(active, plan["reserve_after_bytes"])
    write(out / "receipt.json", {
        "schema": "latency58-leader-cleanup-artifact-retirement-v1", "status": "complete",
        "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"),
        "arm": plan["arm"], "review": plan["review"], "retired_path": str(optimizer),
        "freed_bytes": freed_bytes, "optimizer_bytes": expected["file_binding"]["bytes"],
        "retired_estimates": estimates, "rejected_audio_recreation_plan": str(rejected_dir / "plan.json"),
        "rejected_trial_inputs_unchanged": True, "protected_files": protected,
        "source_bindings_unchanged": True, "active_training_inputs_unchanged": True,
        "counted_bytes_after": after, "headroom_before_stop_bytes": active["stop_counted_bytes"] - after,
        "reserved_bytes": plan["reserve_after_bytes"],
        "reserve_purpose": "bounded native listening captures and candidate export",
        "additional_loss_weight": plan["additional_loss_weight"], "quality_selected": False,
        "training_updates_executed": 0, "next_training_arm_launched": False})
    print({"status": "complete", "arm": plan["arm"], "freed_bytes": freed_bytes,
           "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
