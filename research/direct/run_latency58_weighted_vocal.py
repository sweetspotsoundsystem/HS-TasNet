"""Run the controlled weighted trial under the qualified NVML supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_weighted_vocal import validate_recipe, runtime_policy
from research.direct.latency58_weighted_storage import snapshot as storage_snapshot
from research.direct.recover_latency58_nvml_guard_idle import require_monitor_qualification, WATCHDOG_SHA


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def require_cpu_evidence(plan):
    entries = ((PHASE / "weighted-vocal-quarter-cpu-002", "weighted_cpu_qualification"),
               (PHASE / "lossless-recovery-cpu-003", "packed_cpu_qualification"),
               (Path(plan["output_directory"]) / "cpu-integration", "packed_integration_cpu_qualification"))
    results = []
    for root, key in entries:
        prepared, result, execution = (read(root / name) for name in ("plan.json", "result.json", "execution.json"))
        require(result["status"] == "pass" and not result["gpu_used"] and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"] and execution["result_sha256"] == sha(root / "result.json")
                and execution["plan_sha256"] == result["plan_sha256"] == sha(root / "plan.json")
                and plan[key] == binding(root / "result.json"), "CPU qualification identity or completion differs")
        verify_inputs(prepared); results.append(result)
    weighted, packed, integrated = results
    require(weighted["independent_scalar_and_neural_references_pass"]
            and weighted["ordinary_all_40_gradients_bit_exact"] and weighted["canonical_combined_all_40_gradients_bit_exact"]
            and weighted["third_update_raw_adam_ema_and_accounting_bit_exact"] and weighted["parent_and_rng_unchanged"],
            "Weighted CPU objective qualification is incomplete")
    require(packed["tensor_count"] == 216 and packed["all_tensor_and_metadata_bytes_exact"]
            and packed["third_stochastic_update_raw_adam_ema_bit_exact"]
            and packed["interrupted_publication_retains_loadable_previous_file"]
            and packed["raw_and_ema_native_parity_and_256_sample_latency"]
            and packed["finalization_reuses_same_inode_without_tensor_duplicate"], "Packed CPU recovery qualification is incomplete")
    require(integrated["tensor_count"] == 216 and integrated["training_schedule_steps"] == plan["config"]["steps"] == 1000
            and integrated["parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and integrated["production_storage_preflight_exercised_with_real_packed_file"]
            and integrated["third_stochastic_update_raw_adam_ema_bit_exact"] and integrated["production_rng_restored"]
            and integrated["production_filesystem_wrapper"]["complete_production_wrapper_save_seconds"]
                < runtime_policy()["save_seconds_per_generation"], "Production filesystem adapter is not qualified")


def require_resource_result(result, parity, plan, plan_sha):
    from research.direct.run_latency58_grouped_vocal_runtime import require_resource_result as original_check
    original_check(result, parity, plan, plan_sha)
    require(parity["grouped_parameter_gradients"]["ordinary_all_40_gradients_bit_exact_against_unmodified_reference"],
            "Weighted GPU qualification changed the ordinary gradients")
    recovered = read(Path(plan["output_directory"]) / "resource-run/recovery-gpu-parity.json")
    require(recovered["status"] == "pass" and recovered["device"].startswith("cuda") and recovered["precision"] == "bf16"
            and recovered["parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and recovered["parent_weights_unchanged"] and recovered["tensor_count"] == 217
            and recovered["all_tensor_and_metadata_values_exact"] and recovered["all_40_adam_states_bit_exact"]
            and recovered["third_stochastic_update_raw_adam_ema_bit_exact"] and recovered["all_rng_streams_replayed"]
            and recovered["production_rng_restored"] and recovered["training_schedule_steps"] == 1000
            and recovered["data_cursor_and_journal_exact"] and recovered["production_filesystem_wrapper"] is None
            and recovered["complete_pack_and_audit_seconds"] < runtime_policy()["save_seconds_per_generation"],
            "Selected BF16 device lacks exact, bounded packed recovery")


def require_monitor_closed(execution, monitor, *, final_step=None):
    require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["post_exit_quiet_completed"] and monitor["identities_unchanged"]
            and monitor["source_sha256"] == monitor["source_sha256_after"] == WATCHDOG_SHA
            and monitor["finalization_started"] and monitor["finalization_timeout_seconds"] == 180,
            "The preceding GPU supervisor did not complete successfully")
    for name in ("event_worker_close", "gpu_worker_close"):
        value = monitor[name]
        require(value["closed"] and value["actual_exit_code"] == 0 and not value["forced"],
                "A telemetry worker did not close normally")
    require(monitor["gpu_worker_close"]["identities_unchanged"], "GPU telemetry identities changed")
    if final_step is not None:
        require(monitor["expected_final_step"] == monitor["latest_completed_step_seen"] == final_step,
                "GPU monitor did not observe the requested endpoint")


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
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_weighted_vocal", "--plan", str(plan_path),
                     "--plan-sha256", sha(plan_path), "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / (root.name + "-" + name)
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
    parser.add_argument("--prepared-plan", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--after-resource", action="store_true")
    args = parser.parse_args()
    require(not (args.resource_only and args.after_resource) and ((args.previous_execution is None) == args.after_resource),
            "Provide the predecessor only when starting the resource stage")
    plan_path = args.prepared_plan.resolve(strict=True)
    preceding = (plan_path.parent / "resource-stage/execution.json" if args.after_resource
                 else launch(plan_path, args.previous_execution.resolve(strict=True), resource=True))
    if args.resource_only:
        print(json.dumps({"event": "resource_complete", "execution": str(preceding)}), flush=True)
        return
    execution = launch(plan_path, preceding, resource=False)
    print(json.dumps({"event": "production_complete_requires_saved_quality_evaluation", "execution": str(execution)}), flush=True)


if __name__ == "__main__":
    main()
