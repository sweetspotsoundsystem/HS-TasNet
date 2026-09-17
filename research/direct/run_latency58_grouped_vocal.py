"""Monitor grouped-vocal resource/training stages and score both saved endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_branch_pitch_ema import audit_and_score
from research.direct.train_latency58_grouped_vocal import validate_recipe
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def require_resource_result(result, parity, plan, plan_sha):
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
        require(check["status"] == "pass" and check["scored_samples"] == 88320
                and len(check["all_40_gradients"]) == 40
                and all(row["maximum_error"] == 0 and row["reference_norm"] > 0 for row in check["all_40_gradients"].values())
                and len(check["completed_pass_tensors_released"]) == 2
                and all(row["audio_output_and_loss_released"] for row in check["completed_pass_tensors_released"]),
                "Ordinary or auxiliary GPU warmup/context qualification failed")
    require(ordinary["microbatch_size"] == 4 and ordinary["logical_batch_loss"]["status"] == "pass",
            "Ordinary B16/B4 loss qualification failed")
    gradients, restart = parity["grouped_parameter_gradients"], parity["grouped_restart"]
    require(gradients["status"] == "pass" and gradients["device"].startswith("cuda")
            and gradients["precision"] == "bf16" and gradients["model_state_sha256"] == plan["initialized_model_state_sha256"]
            and gradients["ordinary_microbatch"] == 4 and gradients["auxiliary_microbatch"] == 2
            and gradients["warmup_samples"] == 88064 and gradients["scored_samples"] == 88320
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


def launch(plan_path, previous_execution_path, *, resource):
    plan = read(plan_path)
    validate_recipe(plan)
    verify_inputs(plan)
    budget_snapshot(plan["storage_budget"])
    previous_execution = read(previous_execution_path)
    previous_path = Path(previous_execution["monitor_result"])
    previous = read(previous_path)
    require(previous_execution["actual_exit_code"] == 0 and previous_execution["source_bindings_unchanged"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"], "Previous GPU monitor failed")
    root = Path(plan["output_directory"])
    name = "resource" if resource else "production"
    out, run = root / (name + "-stage"), root / (name + "-run")
    require(not out.exists() and not run.exists(), "Preserve existing monitored stages")
    out.mkdir()
    stop = 2 if resource else plan["config"]["steps"]
    stage = {"schema": "latency58-direct-sdr-stage-v1", "plan_sha256": sha(plan_path), "resource_only": resource,
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
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_grouped_vocal", "--plan", str(plan_path),
                     "--plan-sha256", sha(plan_path), "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / (root.name + "-" + name)
    require(not monitor_out.exists(), "Preserve monitor output")
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(1200 if resource else max(600, stop * 40 + 300)), "--poll-seconds", "2",
            "--query-timeout-seconds", "10", "--startup-grace-seconds", "600" if resource else "120",
            "--progress-timeout-seconds", "120" if resource else "60",
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
            and terminal["post_exit_quiet_completed"], "GPU supervisor did not close cleanly")
    print(json.dumps({"event": "stage_pass", "stage": name, "updates": stop}), flush=True)
    return out / "execution.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-plan", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--after-resource", action="store_true")
    args = parser.parse_args()
    require(not (args.resource_only and args.after_resource)
            and ((args.previous_execution is None) == args.after_resource),
            "Provide the previous execution for a new resource stage only")
    plan_path = args.prepared_plan.resolve(strict=True)
    resource = (plan_path.parent / "resource-stage/execution.json" if args.after_resource else
                launch(plan_path, args.previous_execution.resolve(strict=True), resource=True))
    if args.resource_only:
        print(json.dumps({"event": "resource_complete", "plan": str(plan_path), "execution": str(resource)}), flush=True)
        return
    launch(plan_path, resource, resource=False)
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
