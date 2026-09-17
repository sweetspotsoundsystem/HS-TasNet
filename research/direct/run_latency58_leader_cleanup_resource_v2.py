"""Run exactly two monitored controlled deployed-truth rehearsal updates."""
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
    parser.add_argument("--stop-step", type=int, choices=(2,), required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-check", type=Path, required=True)
    parser.add_argument("--model-check-execution", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-training-v1", "Unknown training plan")
    from research.direct.latency58_leader_cleanup_checkpoint_v2 import require_space, validate_recipe
    validate_recipe(plan)
    require_space(plan, 400_000_000)
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
    checked, checked_execution = read(args.model_check), read(args.model_check_execution)
    checked_plan_path = checked_execution["argv"][checked_execution["argv"].index("--plan") + 1]
    checked_plan = read(checked_plan_path)
    require(checked["schema"] == "latency58-leader-cleanup-model-check-v1" and checked["status"] == "pass"
            and checked["source_bindings_unchanged"] and checked["parent_model_state_sha256"] == plan["initialized_model_state_sha256"]
            and checked["zero_weight_loss_and_all_21_parameter_gradients_exact"] and checked["forward_outputs_exact"]
            and checked["half_weight_changes_parameter_gradients"] and checked["invalid_journals_rejected"] == 7
            and checked_execution["actual_exit_code"] == 0 and not checked_execution["timed_out"]
            and checked_execution["source_bindings_unchanged"]
            and checked["plan_sha256"] == checked_execution["plan_sha256"] == sha(checked_plan_path)
            and checked_plan["recipe_plan"] == {"path": str(args.plan.resolve()), "sha256": args.plan_sha256},
            "New parent CPU model qualification is incomplete")
    verify_inputs(checked_plan)
    decision_binding = plan["preparation_decision"]
    require(sha(decision_binding["path"]) == decision_binding["sha256"], "Preparation decision changed")
    decision = read(decision_binding["path"])
    verify_inputs(decision)
    require(decision["schema"] == "latency58-leader-cleanup-decision-v1"
            and decision["status"] == "prepare_fixed_parent_transfer"
            and decision["training_parent"] == plan["parent"] and decision["config"] == plan["config"]
            and decision["maximum_production_updates"] == 250
            and not decision["quality_selected"] and not decision["goal_complete"], "Different research decision")
    require(plan["resource_only"] and args.stop_step == 2 and not run.exists(),
            "This launcher admits exactly one fresh two-update resource rehearsal")
    start = 0
    require(not run.exists() and args.stop_step == (2 if plan["resource_only"] else 250),
            "Only a fresh rehearsal or complete 250-update pilot is allowed")
    bind = [args.plan, args.previous_monitor, args.previous_execution, args.model_check, args.model_check_execution, Path(checked_plan_path)]
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
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_leader_cleanup_v2", "--plan", str(args.plan.resolve()),
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
    if plan["resource_only"]:
        result_path = run / "resource.json"
        resource = read(result_path)
        require(resource["status"] == "pass" and resource["training_updates_executed"] == 2
                and resource["plan_sha256"] == args.plan_sha256
                and not resource["checkpoint_written"] and read(run / "status.json")["status"] == "resource_complete"
                and all(sha(p) == v for p, v in resource["source_bindings"].items()),
                "Resource rehearsal did not finish cleanly")
        write(out / "resource-qualification.json", {
            "schema": "latency58-leader-cleanup-resource-qualification-v1", "status": "pass",
            "resource": {"path": str(result_path), "sha256": sha(result_path)},
            "execution": {"path": str(out / "execution.json"), "sha256": sha(out / "execution.json")},
            "monitor": {"path": str(monitor_out / "result.json"), "sha256": sha(monitor_out / "result.json")},
            "actual_exit_code": 0, "post_exit_quiet_completed": True, "checkpoint_written": False,
            "model_check": {"path": str(args.model_check.resolve()), "sha256": sha(args.model_check)},
            "model_check_execution": {"path": str(args.model_check_execution.resolve()), "sha256": sha(args.model_check_execution)},
            "source_bindings_unchanged": True, "quality_selected": False})
        print(json.dumps({"event": "resource_rehearsal_pass", "step": 2, "peak_vram_gib": resource["peak_vram_gib"]}), flush=True)
        return


if __name__ == "__main__":
    main()
