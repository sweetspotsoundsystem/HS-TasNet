"""Run the matched 250-update controlled deployed-truth trial and audit its saved endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import PYTHON, execute, write
from research.direct.train_latency58 import ROOT, load_source, read, require, sha, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stop-step", type=int, choices=(250,), required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-training-v1", "Unknown training plan")
    from research.direct.latency58_leader_cleanup_checkpoint import require_space, validate_recipe
    validate_recipe(plan)
    require_space(plan, 400_000_000)
    require(not plan["resource_only"] and plan["additional_loss_weight"] == .5, "Require the fixed half-weight production trial")
    out, run = args.output_directory.absolute(), Path(plan["run_dir"])
    require(out.is_relative_to(ROOT / "research/direct/runs/latency58") and not out.exists(), "Use a fresh stage directory")
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"]
            and Path(execution["monitor_result"]).resolve() == args.previous_monitor.resolve(),
            "Previous GPU stage did not finish successfully or belongs to a different monitor")
    for binding in plan["functional_proofs"]:
        proof, proof_execution = read(binding["result"]), read(binding["execution"])
        require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
                and proof_execution["actual_exit_code"] == 0 and not proof_execution["timed_out"]
                and proof_execution["source_bindings_unchanged"], "Preserved context qualification failed")
    decision_binding = plan["preparation_decision"]
    require(sha(decision_binding["path"]) == decision_binding["sha256"], "Preparation decision changed")
    decision = read(decision_binding["path"])
    verify_inputs(decision)
    require(decision["schema"] == "latency58-leader-cleanup-decision-v1"
            and decision["status"] == "prepare_fixed_parent_transfer"
            and decision["training_parent"] == plan["parent"] and decision["config"] == plan["config"]
            and decision["maximum_production_updates"] == 250
            and decision["additional_loss_weight"] == plan["additional_loss_weight"] == .5
            and not decision["quality_selected"] and not decision["goal_complete"], "Different research decision")
    resource = read(plan["full_resource"]["path"])
    resource_execution = read(plan["full_resource_execution"]["path"])
    resource_monitor = read(resource_execution["monitor_result"])
    resource_plan = read(plan["resource_plan"]["path"])
    shared = ("config", "parent", "warmup_samples", "scored_samples", "carry_state",
              "teacher_kind", "teacher_weight", "teacher_model_state_sha256", "teacher",
              "precision_policy", "torch_version", "environment",
              "helper_source", "watchdog_source", "manifest_sha256", "functional_proofs",
              "microbatch_size", "accumulation_steps", "accumulation_version",
              "drum_weight", "objective_version",
              "arm", "focused_augmentation", "local_mask_mixer", "augmentation_version",
              "parameter_tensors", "mixer_initialization_seed", "initialized_model_state_sha256",
              "architecture", "automatic_continuation", "preparation_decision",
              "teacher_mode", "counterfactual_version", "reference_training_plan",
              "comparison_variable", "checkpoint_and_quality_reserve_bytes", "additional_loss_weight", "additional_loss_version")
    require(resource["schema"] == "latency58-leader-cleanup-resource-result-v1" and resource["status"] == "pass"
            and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
            and resource["source_bindings_unchanged"] and resource["all_parameter_gradients_present"]
            and resource["fixed_buffers_unchanged"] and resource["teacher_unchanged"]
            and resource["initial_model_state_sha256"] == plan["initialized_model_state_sha256"]
            and resource["teacher_model_state_sha256"] == plan["teacher_model_state_sha256"]
            and resource["teacher_mode"] == plan["teacher_mode"] == "ordinary_only"
            and resource["counterfactual_version"] == plan["counterfactual_version"]
            and resource["additional_loss_weight"] == plan["additional_loss_weight"]
            and resource["additional_loss_version"] == plan["additional_loss_version"]
            and resource["config"] == plan["config"] and resource["carry_state"]
            and resource["microbatch_size"] == plan["microbatch_size"]
            and resource["accumulation_steps"] == plan["accumulation_steps"]
            and resource["accumulation_version"] == plan["accumulation_version"]
            and resource["drum_weight"] == plan["drum_weight"]
            and resource["objective_version"] == plan["objective_version"]
            and resource["augmented_examples_executed"] == 32
            and resource_plan["resource_only"] and all(resource_plan[k] == plan[k] for k in shared)
            and resource_execution["actual_exit_code"] == 0 and resource_execution["source_bindings_unchanged"]
            and resource_execution["plan_sha256"] == resource["plan_sha256"] == sha(plan["resource_plan"]["path"])
            and resource_monitor["status"] == resource_monitor["supervisor_health"] == "pass"
            and resource_monitor["child_exit_code"] == 0 and resource_monitor["post_exit_quiet_completed"],
            "Full GPU resource proof differs from the training recipe")
    comparison_binding = plan["resource_comparison"]
    comparison_execution_binding = plan["resource_comparison_execution"]
    require(sha(comparison_binding["path"]) == comparison_binding["sha256"]
            and sha(comparison_execution_binding["path"]) == comparison_execution_binding["sha256"],
            "Parent resource comparison changed")
    comparison, comparison_execution = read(comparison_binding["path"]), read(comparison_execution_binding["path"])
    require(comparison["schema"] == "latency58-leader-cleanup-resource-comparison-v1" and comparison["status"] == "pass"
            and comparison["updates_per_parent"] == 2 and comparison["microbatches_per_parent"] == 8
            and comparison["operational_recipe_exact"] and comparison["all_inputs_and_teacher_targets_exact"]
            and comparison["final_rng_states_exact"] and comparison["comparison_variable"] == "training_parent"
            and comparison["source_bindings_unchanged"] and not comparison["matched_loss_effect_from_leader_claimed"]
            and comparison["resource"] == plan["full_resource"] and comparison["training_plan"] == plan["resource_plan"]
            and comparison_execution["actual_exit_code"] == 0 and not comparison_execution["timed_out"]
            and comparison_execution["source_bindings_unchanged"], "Leader resource comparison is not qualified")
    verify_inputs(comparison)
    start = 0
    require(not run.exists() and args.stop_step == 250,
            "Only a fresh complete 250-update pilot is allowed")
    bind = [args.plan, args.previous_monitor, args.previous_execution]
    stage = {"schema": "latency58-leader-cleanup-stage-v1", "plan_sha256": args.plan_sha256,
             "start_step": start, "stop_step": args.stop_step, "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"],
             "previous_monitor": {"path": str(args.previous_monitor.resolve()), "sha256": sha(args.previous_monitor)},
             "source_bindings": {str(p.resolve()): sha(p) for p in bind}}
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve previous GPU monitor output")
    out.mkdir()
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_leader_cleanup", "--plan", str(args.plan.resolve()),
                     "--plan-sha256", args.plan_sha256, "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    timeout = max(240, (args.stop_step - start) * 8 * plan["accumulation_steps"] + 180)
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(timeout), "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "cwd": str(ROOT), "plan_sha256": args.plan_sha256,
                                 "stage_sha256": sha(out / "stage.json")})
    print(json.dumps({"event": "training_launch", "teacher": plan["teacher_kind"],
                      "start_step": start, "stop_step": args.stop_step, "carry_state": plan["carry_state"],
                      "teacher_mode": plan["teacher_mode"]}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": all(sha(p) == s for p, s in
                                                  {**plan["source_bindings"], **stage["source_bindings"]}.items()),
                 "plan_sha256": args.plan_sha256, "stage_sha256": sha(out / "stage.json"),
                 "monitor_result": str(monitor_out / "result.json")}
    write(out / "execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and execution["source_bindings_unchanged"], "Training stage failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"] and read(run / "status.json")["step"] == args.stop_step,
            "Training endpoint or monitor did not pass")
    generation = Path(read(run / "latest.json")["generation"])
    bindings = {**plan["source_bindings"], str(args.plan.resolve()): args.plan_sha256}
    bindings.update({str(p): sha(p) for p in generation.iterdir() if p.is_file()})
    argv = [PYTHON, "-u", "-m", "research.direct.audit_latency58_leader_cleanup", "--plan", str(args.plan.resolve()),
            "--plan-sha256", args.plan_sha256, "--generation", str(generation), "--output", str(out / "audit.json")]
    execute(argv, out, "audit", 180, bindings, {"plan_sha256": args.plan_sha256})
    audit = read(out / "audit.json")
    require(audit["status"] == "pass" and audit["step"] == args.stop_step, "Saved-state audit failed")
    helpers = load_source("latency58_sdr_stage_helpers", plan["helper_source"])
    helpers.atomic_json(run / "audit-latest.json", {
        "step": args.stop_step, "generation_receipt_sha256": sha(generation / "receipt.json"),
        "audit": {"path": str(out / "audit.json"), "sha256": sha(out / "audit.json")},
        "execution": {"path": str(out / "audit-execution.json"), "sha256": sha(out / "audit-execution.json")}})
    print(json.dumps({"event": "training_and_audit_pass", "step": args.stop_step, "teacher": plan["teacher_kind"],
                      "carry_state": plan["carry_state"]}), flush=True)


if __name__ == "__main__":
    main()
