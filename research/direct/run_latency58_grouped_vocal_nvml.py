"""Run the qualified NVML monitor, fresh resource check, and recovered trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_branch_pitch_ema import audit_and_score
from research.direct.train_latency58_grouped_vocal_nvml import validate_recipe
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.recover_latency58_nvml_idle import require_monitor_qualification, WATCHDOG_SHA


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def require_recovery_check(result):
    require(result["status"] == "pass" and result["parent_weights_unchanged"]
            and result["all_40_adam_states_bit_exact"]
            and result["third_stochastic_update_raw_adam_ema_bit_exact"]
            and result["all_rng_streams_replayed"]
            and result["interrupted_publication_retains_loadable_previous_file"]
            and result["successful_atomic_replacement_and_receipt_history"]
            and result["training_schedule_still_500"] and result["ephemeral_tensor_files_removed"]
            and {row["case"] for row in result["rejected_cases"]} == {
                "wrong_file_hash", "wrong_plan_hash", "changed_full_schedule", "changed_cursor",
                "truncated_journal", "EMA_used_as_raw_optimizer_owner", "wrong_numpy_rng",
                "interrupted_publication", "pending_file_preserved", "stale_binding_after_replacement"}
            and result["first_publication_seconds"] < 10
            and result["replacement_publication_seconds"] < 10,
            "Rolling recovery did not pass exact restart, interruption or publication timing checks")


def require_resource_result(result, parity, plan, plan_sha):
    from research.direct.run_latency58_grouped_vocal_runtime import require_resource_result as original_check
    original_check(result, parity, plan, plan_sha)
    recovery_path = Path(plan["output_directory"]) / "resource-run/recovery-gpu-parity.json"
    recovery = read(recovery_path)
    require_recovery_check(recovery)
    require(recovery["device"].startswith("cuda") and recovery["precision"] == "bf16"
            and recovery["production_rng_restored"]
            and len(recovery["complete_successful_save_seconds"]) == 2
            and max(recovery["complete_successful_save_seconds"]) < 10,
            "Disk recovery lacks a bounded BF16 device rehearsal")


def launch(plan_path, previous_execution_path, *, resource, prefix_only=False):
    require(not (resource and prefix_only), "Resource and resumed-prefix stages are separate")
    plan = read(plan_path)
    recovery_cpu = PHASE / "grouped-vocal-recovery-cpu-001"
    recovery_result, recovery_execution, recovery_root = (read(recovery_cpu / name) for name in
        ("result.json", "execution.json", "root-execution.json"))
    require_recovery_check(recovery_result)
    require(recovery_result["device"] == "cpu" and recovery_result["gpu_used"] is False
            and recovery_result["source_bindings_unchanged"]
            and recovery_execution["actual_exit_code"] == recovery_root["actual_exit_code"] == 0
            and recovery_execution["source_bindings_unchanged"] and not recovery_execution["timed_out"]
            and recovery_execution["result_sha256"] == recovery_root["result_sha256"] == sha(recovery_cpu / "result.json")
            and recovery_root["execution_sha256"] == sha(recovery_cpu / "execution.json")
            and recovery_root["actual_session_id"] == 64727 and recovery_root["actual_tool_chunk_id"] == "70e696",
            "Rolling recovery CPU qualification is incomplete")
    verify_inputs(read(recovery_cpu / "plan.json"))
    require(plan.get("runtime_allowance") == {
        "version": "canonical-observed-runtime-per-update60-plus600-v1",
        "resource_max_seconds": 1200, "production_seconds_per_update": 60,
        "production_fixed_allowance_seconds": 600}, "Require the prepared canonical runtime allowance")
    cpu_root = PHASE / "grouped-vocal-canonical-cpu-001"
    cpu_plan, cpu_result, cpu_execution = (read(cpu_root / (n + ".json")) for n in ("plan", "result", "execution"))
    require(cpu_result["status"] == "pass" and cpu_result["source_bindings_unchanged"]
            and cpu_execution["actual_exit_code"] == cpu_execution["actual_enclosing_exit_code"] == 0
            and cpu_execution["source_bindings_unchanged"] and not cpu_execution["timed_out"]
            and cpu_result["plan_sha256"] == cpu_execution["plan_sha256"] == sha(cpu_root / "plan.json")
            and cpu_execution["result_sha256"] == sha(cpu_root / "result.json")
            and cpu_plan["fixture_model_state_sha256"] == plan["initialized_model_state_sha256"]
            and cpu_result["accumulation_policy"] == plan["accumulation_policy"]
            and cpu_result["all_40_parameter_gradients_match_independent_whole_objective"]
            and cpu_result["third_update_raw_adam_ema_and_accounting_bit_exact"]
            and cpu_result["parent_and_rng_unchanged"] and not cpu_result["gpu_used"],
            "Canonical CPU qualification is incomplete")
    require_monitor_qualification()
    monitor_sha = WATCHDOG_SHA
    require(sha(plan["watchdog_source"]) == monitor_sha
            and plan["supervision"] == {"version": "persistent-nvml-planned-final-step-v1",
                                       "finalization_timeout_seconds": 180,
                                       "watchdog_sha256": monitor_sha},
            "Require the qualified finalization monitor and its fixed 180-second bound")
    validate_recipe(plan)
    verify_inputs(plan)
    budget_snapshot(plan["storage_budget"])
    previous_execution = read(previous_execution_path)
    previous_path = Path(previous_execution["monitor_result"])
    previous = read(previous_path)
    require(previous_execution["actual_exit_code"] == 0 and previous_execution["source_bindings_unchanged"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"]
            and previous["identities_unchanged"] and previous["source_sha256"] == monitor_sha
            and previous["finalization_started"] and previous["event_worker_close"]["closed"]
            and previous["event_worker_close"]["actual_exit_code"] == 0
            and not previous["event_worker_close"]["forced"]
            and previous["gpu_worker_close"]["closed"] and previous["gpu_worker_close"]["identities_unchanged"]
            and previous["gpu_worker_close"]["actual_exit_code"] == 0
            and not previous["gpu_worker_close"]["forced"], "Previous GPU monitor failed")
    root = Path(plan["output_directory"])
    name = "resource" if resource else "resume-prefix" if prefix_only else "production"
    out, run = root / (name + "-stage"), root / (name + "-run")
    require(not out.exists() and not run.exists(), "Preserve existing monitored stages")
    out.mkdir()
    stop = 2 if resource else 103 if prefix_only else plan["config"]["steps"]
    stage = {"schema": "latency58-direct-sdr-stage-v1", "plan_sha256": sha(plan_path), "resource_only": resource,
             "resume_prefix_only": prefix_only,
             "stop_step": stop, "run_directory": str(run), "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"], "previous_monitor": binding(previous_path),
             "source_bindings": {str(p): sha(p) for p in (plan_path, previous_path, previous_execution_path)}}
    if not resource:
        path = root / "resource-run/result.json"
        result = read(path)
        parity_path = root / "resource-run/grouped-vocal-gpu-parity.json"
        parity = read(parity_path)
        require_resource_result(result, parity, plan, sha(plan_path))
        stage["resource_result"] = binding(path)
        stage["source_bindings"][str(path)] = sha(path)
        stage["source_bindings"][str(parity_path)] = sha(parity_path)
        disk_parity = root / "resource-run/recovery-gpu-parity.json"
        stage["source_bindings"][str(disk_parity)] = sha(disk_parity)
        if not prefix_only:
            prefix_path = root / "resume-prefix-run/result.json"
            prefix = read(prefix_path)
            require(previous_execution_path.resolve() == (root / "resume-prefix-stage/execution.json").resolve()
                    and prefix["status"] == "pass" and prefix["source_bindings_unchanged"]
                    and prefix["plan_sha256"] == sha(plan_path) and prefix["resumed_from_step"] == 100
                    and prefix["updates"] == prefix["ema_updates"] == 103 and prefix["resume_prefix_only"]
                    and prefix["full_schedule_steps"] == 500 and not prefix["checkpoint_written"]
                    and prefix["replayed_steps_matching_retained_raw_ema_data_loss_gradients"] == [101, 102, 103],
                    "Actual restored training prefix is not qualified")
            stage["resume_prefix_result"] = binding(prefix_path)
            stage["source_bindings"][str(prefix_path)] = sha(prefix_path)
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "expected_final_step": stop,
            "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_grouped_vocal_nvml", "--plan", str(plan_path),
                     "--plan-sha256", sha(plan_path), "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / (root.name + "-" + name)
    require(not monitor_out.exists(), "Preserve monitor output")
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(1200 if resource else 780 if prefix_only else stop * 60 + 600), "--poll-seconds", "2",
            "--query-timeout-seconds", "10", "--startup-grace-seconds", "600" if resource else "120",
            "--progress-timeout-seconds", "120" if resource else "60",
            "--finalization-timeout-seconds", "180",
            "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80",
            "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv})
    print(json.dumps({"event": "launch", "stage": name, "updates": stop}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as log:
        child = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(p) == h for p, h in {**plan["source_bindings"], **stage["source_bindings"]}.items())
    write(out / "execution.json", {"actual_exit_code": child.returncode, "elapsed_seconds": time.monotonic() - began,
          "source_bindings_unchanged": unchanged, "plan_sha256": sha(plan_path), "monitor_result": str(monitor_out / "result.json")})
    require(child.returncode == 0 and unchanged, "Monitored direct-SDR stage failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"] and terminal["expected_final_step"] == stop
            and terminal["latest_completed_step_seen"] == stop and terminal["finalization_started"]
            and terminal["finalization_timeout_seconds"] == 180
            and terminal["gpu_worker_close"]["closed"] and terminal["gpu_worker_close"]["identities_unchanged"]
            and terminal["gpu_worker_close"]["actual_exit_code"] == 0
            and not terminal["gpu_worker_close"]["forced"], "GPU supervisor did not close cleanly")
    print(json.dumps({"event": "stage_pass", "stage": name, "updates": stop}), flush=True)
    return out / "execution.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-plan", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--after-resource", action="store_true")
    parser.add_argument("--resume-prefix-only", action="store_true")
    parser.add_argument("--after-resume-prefix", action="store_true")
    args = parser.parse_args()
    require(not (args.resource_only and (args.after_resource or args.after_resume_prefix or args.resume_prefix_only))
            and not (args.after_resume_prefix and (args.after_resource or args.resume_prefix_only))
            and (not args.resume_prefix_only or args.after_resource)
            and ((args.previous_execution is None) == (args.after_resource or args.after_resume_prefix)),
            "Provide the previous execution for a new resource stage only")
    plan_path = args.prepared_plan.resolve(strict=True)
    resource = (plan_path.parent / "resource-stage/execution.json" if args.after_resource or args.after_resume_prefix else
                launch(plan_path, args.previous_execution.resolve(strict=True), resource=True))
    if args.resource_only:
        print(json.dumps({"event": "resource_complete", "plan": str(plan_path), "execution": str(resource)}), flush=True)
        return
    prefix = (plan_path.parent / "resume-prefix-stage/execution.json" if args.after_resume_prefix else
              launch(plan_path, resource, resource=False, prefix_only=True))
    if args.resume_prefix_only:
        print(json.dumps({"event": "resume_prefix_complete", "plan": str(plan_path), "execution": str(prefix)}), flush=True)
        return
    launch(plan_path, prefix, resource=False)
    audit_and_score(plan_path)
    terminal = read(plan_path.parent / "result.json")
    write(plan_path.parent / "acceptance-status.json", {
        "saved_full14_sdr_gate_met": terminal["target_reached"],
        "saved_full14_sdr_db": terminal["full_sdr_db"],
        "deployment_graph_quality": "requires export and exact-graph evaluation",
        "m4_playback_gate": "requires repeated measured physical-M4 playback",
        "instrumental_vocal_gate": "requires paired source-view evaluation and user listening acceptance",
        "overall_goal_complete": False, "plugin_replaced": False})


if __name__ == "__main__":
    main()
