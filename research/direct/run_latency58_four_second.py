"""Run the selected four-second trial with the unchanged health supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_four_second import validate_recipe, runtime_policy, budget_snapshot as storage_snapshot
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.recover_latency58_nvml_guard_idle import require_monitor_qualification, WATCHDOG_SHA
from research.direct.run_latency58_weighted_vocal import require_monitor_closed, binding


def require_cpu_evidence(plan):
    entries = ((PHASE / "four-second-model-cpu-002", "four_second_model_cpu_qualification"),
               (ARTIFACT_ROOT / "restart-cpu-001", "four_second_restart_cpu_qualification"),
               (ARTIFACT_ROOT / "selected-data-prefix-002", "four_second_prefix_qualification"))
    results = []
    for root, key in entries:
        prepared, result, execution = (read(root / name) for name in ("plan.json", "result.json", "root-execution.json"))
        require(result["status"] == "pass" and not result["gpu_used"] and result["source_bindings_unchanged"]
                and type(execution["actual_exit_code"]) is int and execution["actual_exit_code"] == 0
                and execution["timed_out"] is False and execution["source_bindings_unchanged"]
                and execution["result_sha256"] == sha(root / "result.json")
                and execution["plan_sha256"] == result["plan_sha256"] == sha(root / "plan.json")
                and plan[key] == binding(root / "result.json"), "Four-second CPU evidence is incomplete")
        verify_inputs(prepared); results.append(result)
    model, recovery, data = results
    gradients = model["whole_group_neural_gradient_comparison"]
    require(model["fixture_model_state_sha256"] == plan["parent_model_state_sha256"]
            and model["warmup_samples"] == 88064 and model["scored_samples"] == 176512
            and model["ordinary_microbatch"] == gradients["ordinary_microbatch"] == 2
            and len(gradients["all_40_gradients"]) == 40
            and all(row["bitwise_equal"] for row in gradients["all_40_gradients"].values())
            and model["model_weights_unchanged"] and model["rng_unchanged"],
            "Complete four-second CPU model gradients are not qualified")
    recovered = recovery["restart"]
    require(recovered["status"] == "pass" and recovered["parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and recovered["ordinary_microbatch"] == recovered["auxiliary_microbatch"] == 2
            and recovered["planned_stop_step"] == plan["config"]["steps"] == 2000
            and recovered["tensor_count"] == 216 and recovered["all_tensor_and_metadata_bytes_exact"]
            and recovered["third_weighted_b2_update_and_accounting_bit_exact"]
            and recovered["all_40_adam_states_bit_exact"] and recovered["all_cpu_rng_streams_replayed"]
            and recovered["interrupted_publication_retains_loadable_previous_generation"]
            and recovered["production_budget_preflight_qualified"] and recovered["finalization_reuses_inode"]
            and recovered["parent_unchanged"] and recovery["global_rng_restored"]
            and all(row["algorithmic_latency_samples"] == 256
                    and row["all_six_outputs_and_eight_states_bit_exact"]
                    for row in recovered["final_raw_and_ema_parity"].values()),
            "Selected B2 weighted update or new packed disk recovery is not qualified")
    require(data["qualified_data_prefix"] == plan["qualified_data_prefix"]
            and data["ordinary_complete_windows_per_stem"] == 64
            and data["auxiliary_complete_windows_per_stem"] == 8
            and data["reproduces_prior_two_worker_proof"], "Selected four-second training data differs")


def require_base_resource_result(result, parity, plan, plan_sha):
    require(parity["grouped_parameter_gradients"]["accumulation_policy"] == plan["accumulation_policy"]
            and parity["grouped_restart"]["accumulation_policy"] == plan["accumulation_policy"]
            and parity["grouped_parameter_gradients"]["canonical_replay_outputs_bit_exact"]
            and parity["grouped_restart"]["noncontiguous_step_rejected_before_gradients"]
            and len(parity["grouped_restart"]["interrupted_accumulations"]) == 2
            and all(row["weights_adam_ema_unchanged"] for row in parity["grouped_restart"]["interrupted_accumulations"])
            and all(row["accumulation_policy"] == plan["accumulation_policy"]
                    and all(g["replay_outputs_bit_exact"] for g in row["groups"].values())
                    for row in result["matching_production_updates"]), "Canonical resource evidence differs")
    require(result["status"] == "pass" and result["updates"] == 2 and not result["checkpoint_written"]
            and result["source_bindings_unchanged"] and result["plan_sha256"] == plan_sha
            and result["ema_updates"] == 2 and result["ema_policy"] == plan["ema"]
            and len(result["matching_production_updates"]) == 2
            and parity["status"] == "pass" and parity["trained_parent_weights_unchanged"]
            and parity["cpu_and_cuda_rng_restored"] and parity["training_optimizer_updates"] == 0
            and parity["original_model_state_sha256"] == plan["initialized_model_state_sha256"],
            "Grouped resource rehearsal did not reach a matching raw/EMA endpoint")
    ordinary, auxiliary = parity["ordinary_context"], parity["auxiliary_context"]
    for check in (ordinary, auxiliary):
        require(check["status"] == "pass" and check["scored_samples"] == 176512
                and len(check["all_40_gradients"]) == 40
                and all(row["maximum_error"] == 0 and row["reference_norm"] > 0 for row in check["all_40_gradients"].values())
                and len(check["completed_pass_tensors_released"]) == 2
                and all(row["audio_output_and_loss_released"] for row in check["completed_pass_tensors_released"]),
                "Ordinary or auxiliary GPU warmup/context qualification failed")
    require(ordinary["microbatch_size"] == 2 and ordinary["logical_batch_loss"]["status"] == "pass",
            "Ordinary B16/B2 loss qualification failed")
    gradients, restart = parity["grouped_parameter_gradients"], parity["grouped_restart"]
    require(gradients["status"] == "pass" and gradients["device"].startswith("cuda")
            and gradients["precision"] == "bf16" and gradients["model_state_sha256"] == plan["initialized_model_state_sha256"]
            and gradients["ordinary_microbatch"] == 2 and gradients["auxiliary_microbatch"] == 2
            and gradients["warmup_samples"] == 88064 and gradients["scored_samples"] == 176512
            and len(gradients["all_40_gradients"]) == 40 and gradients["weights_unchanged"]
            and gradients["warmup_input_gradients_zero"] and gradients["optimizer_updates"] == 0
            and gradients["absolute_gradient_tolerance"] == 1e-7 and gradients["relative_gradient_tolerance"] == 1e-4
            and gradients["relative_l2_tolerance"] == 5e-5 and gradients["loss_absolute_tolerance"] == 3e-6
            and all(row["relative_l2_error"] < 5e-5 and row["reference_norm"] > 0
                    for row in gradients["all_40_gradients"].values()), "Grouped GPU whole-objective gradients failed")
    require(restart["status"] == "pass" and restart["device"].startswith("cuda") and restart["precision"] == "bf16"
            and restart["parent_model_state_sha256"] == plan["initialized_model_state_sha256"]
            and restart["all_40_adam_states_checked"] and restart["third_update_raw_adam_ema_and_accounting_bit_exact"]
            and restart["parent_weights_unchanged"] and not restart["checkpoint_files_written"],
            "Grouped BF16 serialization/restart failed")
    require(len(result["resource_ema_arithmetic_checks"]) == 2
            and all(check["status"] == "pass" and check["step"] == index + 1
                    and check["raw_device"].startswith("cuda") and len(check["parameter_errors"]) == 40
                    and check["maximum_absolute_error"] < check["absolute_tolerance"] == 2e-6
                    for index, check in enumerate(result["resource_ema_arithmetic_checks"])),
            "Actual grouped Adam updates failed EMA arithmetic qualification")

def require_resource_result(result, parity, plan, plan_sha):
    require_base_resource_result(result, parity, plan, plan_sha)
    require(parity["grouped_parameter_gradients"]["ordinary_all_40_gradients_bit_exact_against_unmodified_reference"],
            "Weighted GPU qualification changed the ordinary gradients")
    recovered = read(Path(plan["output_directory"]) / "resource-run/recovery-gpu-parity.json")
    require(recovered["status"] == "pass" and recovered["device"].startswith("cuda") and recovered["precision"] == "bf16"
            and recovered["parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and recovered["parent_weights_unchanged"] and recovered["tensor_count"] == 217
            and recovered["all_tensor_and_metadata_values_exact"] and recovered["all_40_adam_states_bit_exact"]
            and recovered["third_stochastic_update_raw_adam_ema_bit_exact"] and recovered["all_rng_streams_replayed"]
            and recovered["production_rng_restored"] and recovered["training_schedule_steps"] == 2000
            and recovered["data_cursor_and_journal_exact"] and recovered["production_filesystem_wrapper"] is not None
            and recovered["production_filesystem_wrapper"]["all_tensor_and_metadata_values_exact"]
            and recovered["production_filesystem_wrapper"]["complete_production_wrapper_save_seconds"]
                < runtime_policy()["save_seconds_per_generation"]
            and recovered["complete_pack_and_audit_seconds"] < runtime_policy()["save_seconds_per_generation"],
            "Selected BF16 device lacks exact, bounded packed recovery")

def launch(plan_path, previous_execution_path, *, resource):
    plan = read(plan_path)
    validate_recipe(plan)
    require(not plan.get("resume_checkpoint") and plan["optimizer_initialization"] == "fresh_adam",
            "This controller executes the selected fresh controlled trial")
    require_cpu_evidence(plan)
    require_monitor_qualification()
    require(sha(plan["watchdog_source"]) == WATCHDOG_SHA and plan["supervision"] == {
        "version": "persistent-nvml-planned-final-step-v1", "finalization_timeout_seconds": 180,
        "watchdog_sha256": WATCHDOG_SHA}, "Keep the qualified GPU health supervisor")
    verify_inputs(plan)
    budget = storage_snapshot(plan)
    root = Path(plan["output_directory"])
    require(previous_execution_path == (Path(plan["previous_execution_for_resource"]) if resource
                                        else root / "resource-stage/execution.json"), "Wrong GPU continuity predecessor")
    if not resource:
        enclosing = read(root / "resource-root-execution.json")
        require(type(enclosing["actual_exit_code"]) is int and enclosing["actual_exit_code"] == 0
                and enclosing["timed_out"] is False and enclosing["source_bindings_unchanged"]
                and enclosing["plan_sha256"] == sha(plan_path)
                and enclosing["stage_execution_sha256"] == sha(root / "resource-stage/execution.json")
                and enclosing["result_sha256"] == sha(root / "resource-run/result.json"),
                "Observe actual resource-controller completion before production")
    previous_execution = read(previous_execution_path)
    previous_path = Path(previous_execution["monitor_result"])
    previous = read(previous_path)
    require_monitor_closed(previous_execution, previous)
    name = "resource" if resource else "production"
    out, run = root / (name + "-stage"), root / (name + "-run")
    require(not out.exists() and not run.exists(), "Preserve existing weighted training stages")
    stop = 2 if resource else plan["config"]["steps"]
    stage = {"schema": "latency58-direct-sdr-stage-v1", "plan_sha256": sha(plan_path), "resource_only": resource,
             "stop_step": stop, "run_directory": str(run), "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"], "previous_monitor": binding(previous_path),
             "source_bindings": {str(path): sha(path) for path in (plan_path, previous_path, previous_execution_path)},
             "storage_before": budget}
    if not resource:
        result_path, parity_path = root / "resource-run/result.json", root / "resource-run/grouped-vocal-gpu-parity.json"
        require_resource_result(read(result_path), read(parity_path), plan, sha(plan_path))
        stage["resource_result"] = binding(result_path)
        for path in (result_path, parity_path, root / "resource-run/recovery-gpu-parity.json"):
            stage["source_bindings"][str(path)] = sha(path)
    out.mkdir(); write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "expected_final_step": stop,
            "cwd": str(ROOT), "environment": plan["environment"], "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_four_second", "--plan", str(plan_path),
                     "--plan-sha256", sha(plan_path), "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = ARTIFACT_ROOT / "monitors" / (root.name + "-" + name)
    monitor_out.parent.mkdir(exist_ok=True)
    require(not monitor_out.exists(), "Preserve existing GPU monitor evidence")
    runtime = runtime_policy()
    maximum_seconds = (runtime["resource_max_seconds"] if resource else stop * runtime["production_seconds_per_update"]
        + (stop // plan["recovery_checkpoint"]["interval_updates"]) * runtime["save_seconds_per_generation"]
        + runtime["production_fixed_allowance_seconds"])
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(maximum_seconds), "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "600" if resource else "120", "--progress-timeout-seconds", "120",
            "--finalization-timeout-seconds", "180", "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10",
            "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    command = {"argv": argv, "runtime_policy": runtime,
               "progress_window_reason": "Allow a bounded CPU packed save followed by the next complete compute step; GPU health polling and limits remain fixed"}
    write(out / "command.json", command)
    print(json.dumps({"event": "launch", "stage": name, "updates": stop, "maximum_seconds": maximum_seconds,
                      "plan_sha256": sha(plan_path)}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as log:
        child = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(path) == digest for path, digest in {**plan["source_bindings"], **stage["source_bindings"]}.items())
    execution = {"actual_exit_code": child.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": unchanged, "plan_sha256": sha(plan_path),
                 "monitor_result": str(monitor_out / "result.json"), "command_sha256": sha(out / "command.json")}
    write(out / "execution.json", execution)
    require(child.returncode == 0 and unchanged, "Monitored weighted stage failed")
    terminal = read(monitor_out / "result.json")
    require_monitor_closed(execution, terminal, final_step=stop)
    if resource:
        require_resource_result(read(run / "result.json"), read(run / "grouped-vocal-gpu-parity.json"), plan, sha(plan_path))
    write(out / "storage-after.json", storage_snapshot(plan))
    print(json.dumps({"event": "stage_pass", "stage": name, "updates": stop,
                      "execution_sha256": sha(out / "execution.json")}), flush=True)
    return out / "execution.json"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--stage", choices=("resource", "production"), required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, "Require the shared project working directory")
    launch(args.plan, args.previous_execution, resource=args.stage == "resource")


if __name__ == "__main__":
    main()
