"""Run the selected four-second trial with the unchanged health supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_four_second_shared import validate_recipe, runtime_policy, budget_snapshot as storage_snapshot
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.latency58_four_second_monitor import (require_monitor_qualification, WATCHDOG_SHA, require_monitor_closed, require_original_monitor_closed)
from research.direct.run_latency58_weighted_vocal import binding


from research.direct.latency58_four_second_shared_qualification import require_cpu_evidence,require_gpu_evidence

def require_base_resource_result(result, parity, plan, plan_sha):
    require(parity == require_gpu_evidence(plan), "Actual resource parent did not bind the completed CUDA geometry proof")
    require(result["status"] == "pass" and result["updates"] == 2 and not result["checkpoint_written"]
            and result["source_bindings_unchanged"] and result["plan_sha256"] == plan_sha
            and result["ema_updates"] == 2 and result["ema_policy"] == plan["ema"]
            and len(result["matching_production_updates"]) == 2,
            "Shared resource rehearsal did not reach a matching raw/EMA endpoint")
    for row in result["matching_production_updates"]:
        require(row["accumulation_policy"] == plan["accumulation_policy"]
                and row["adam_steps_this_update"] == row["gradient_clips_this_update"] == 1
                and len(row["parameter_gradient_norms"]) == 40
                and all(v > 0 for v in row["parameter_gradient_norms"].values())
                and all(g["replay_outputs_bit_exact"] for g in row["groups"].values())
                and row["groups"]["ordinary"]["examples"] == 16
                and row["groups"]["auxiliary"]["examples"] == 2
                and len(row["groups"]["ordinary"]["microbatches"]) == 1
                and len(row["groups"]["auxiliary"]["microbatches"]) == 1
                and row["groups"]["auxiliary"]["view_contribution_multipliers"] == [1., .25],
                "Actual B16/B2 weighted update geometry or complete-group replay differs")
    require(len(result["resource_ema_arithmetic_checks"]) == 2
            and all(check["status"] == "pass" and check["step"] == index + 1
                    and check["raw_device"].startswith("cuda") and len(check["parameter_errors"]) == 40
                    and check["maximum_absolute_error"] < check["absolute_tolerance"] == 2e-6
                    for index, check in enumerate(result["resource_ema_arithmetic_checks"])),
            "Actual shared-storage Adam updates failed EMA arithmetic qualification")

def require_resource_result(result, parity, plan, plan_sha):
    require_base_resource_result(result, parity, plan, plan_sha)
    recovered = read(Path(plan["output_directory"]) / "resource-run-005/recovery-gpu-parity.json")
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

    root = Path(plan["output_directory"])
    allocator = read(root / "resource-run-005/allocator-after-qualification.json")
    require(allocator["allocator_settings"]["expandable_segments"] is True
            and allocator["expandable_segment_count"] > 0 and allocator["configured_process_fraction"] == .75,
            "Selected memory allocator was not observed during full qualification")
    rows = [json.loads(line) for line in (root / "resource-run-005/metrics.jsonl").read_text().splitlines()]
    slowest_update = max(row["compute_and_audit_seconds"] + row["data_wait_seconds"] for row in rows)
    measured_save = recovered["production_filesystem_wrapper"]["complete_production_wrapper_save_seconds"]
    require(len(rows) == 2 and slowest_update < runtime_policy()["production_seconds_per_update"]
            and slowest_update + measured_save + 15 < runtime_policy()["progress_timeout_seconds"],
            "Measured B16/B2 update and checkpoint save exceed the prepared production allowances")


def launch(plan_path, previous_execution_path, *, resource):
    plan = read(plan_path)
    validate_recipe(plan)
    require(plan["environment"].get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "Allocator retry setting changed")
    require(not plan.get("resume_checkpoint") and plan["optimizer_initialization"] == "fresh_adam",
            "This controller executes the selected fresh controlled trial")
    require_cpu_evidence(plan)
    require_gpu_evidence(plan)
    require_monitor_qualification()
    require(sha(plan["watchdog_source"]) == WATCHDOG_SHA and plan["supervision"] == {
        "version": "persistent-nvml-planned-final-step-v1", "finalization_timeout_seconds": 180,
        "watchdog_sha256": WATCHDOG_SHA}, "Keep the qualified GPU health supervisor")
    verify_inputs(plan)
    budget = storage_snapshot(plan)
    root = Path(plan["output_directory"])
    require(previous_execution_path == (Path(plan["previous_execution_for_resource"]) if resource
                                        else root / "resource-stage-005/execution.json"), "Wrong GPU continuity predecessor")
    if not resource:
        enclosing = read(root / "resource-root-execution-005.json")
        require(type(enclosing["actual_exit_code"]) is int and enclosing["actual_exit_code"] == 0
                and enclosing["timed_out"] is False and enclosing["source_bindings_unchanged"]
                and enclosing["plan_sha256"] == sha(plan_path)
                and enclosing["stage_execution_sha256"] == sha(root / "resource-stage-005/execution.json")
                and enclosing["result_sha256"] == sha(root / "resource-run-005/result.json"),
                "Observe actual resource-controller completion before production")
    previous_execution = read(previous_execution_path)
    previous_path = Path(previous_execution["monitor_result"])
    previous = read(previous_path)
    require_monitor_closed(previous_execution, previous, final_step=(12 if resource else 2))
    name = "resource" if resource else "production"
    out, run = root / (name + ("-stage-005" if resource else "-stage")), root / (name + ("-run-005" if resource else "-run"))
    require(not out.exists() and not run.exists(), "Preserve existing weighted training stages")
    stop = 2 if resource else plan["config"]["steps"]
    stage = {"schema": "latency58-direct-sdr-stage-v1", "plan_sha256": sha(plan_path), "resource_only": resource,
             "stop_step": stop, "run_directory": str(run), "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"], "previous_monitor": binding(previous_path),
             "source_bindings": {str(path): sha(path) for path in (plan_path, previous_path, previous_execution_path)},
             "storage_before": budget}
    if not resource:
        result_path, parity_path = root / "resource-run-005/result.json", root / "resource-run-005/grouped-vocal-gpu-parity.json"
        require_resource_result(read(result_path), read(parity_path), plan, sha(plan_path))
        stage["resource_result"] = binding(result_path)
        for path in (result_path, parity_path, root / "resource-run-005/recovery-gpu-parity.json"):
            stage["source_bindings"][str(path)] = sha(path)
    out.mkdir(); write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "expected_final_step": stop,
            "cwd": str(ROOT), "environment": plan["environment"], "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_four_second_shared", "--plan", str(plan_path),
                     "--plan-sha256", sha(plan_path), "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = ARTIFACT_ROOT / "monitors" / (root.name + "-" + name + ("-005" if resource else ""))
    monitor_out.parent.mkdir(exist_ok=True)
    require(not monitor_out.exists(), "Preserve existing GPU monitor evidence")
    runtime = runtime_policy()
    maximum_seconds = (runtime["resource_max_seconds"] if resource else stop * runtime["production_seconds_per_update"]
        + (stop // plan["recovery_checkpoint"]["interval_updates"]) * runtime["save_seconds_per_generation"]
        + runtime["production_fixed_allowance_seconds"])
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(maximum_seconds), "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", str(runtime["resource_startup_grace_seconds"] if resource
                else runtime["production_startup_grace_seconds"]), "--progress-timeout-seconds", str(runtime["progress_timeout_seconds"]),
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
