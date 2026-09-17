"""Execute one frozen teacher/control arm, then audit its endpoint on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, execute, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Frozen training plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-teacher-training-plan-v1" and plan["config"]["steps"] == 250,
            "Use one bounded teacher/control plan")
    previous = read(plan["previous_monitor"]["path"])
    previous_execution = read(plan["previous_execution"]["path"])
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and previous_execution["actual_exit_code"] == 0
            and previous_execution["source_bindings_unchanged"], "Previous monitored stage did not finish")
    resource = read(plan["full_resource_result"]["path"])
    require(resource["status"] == "pass" and resource["samples"] == 88064
            and resource["teacher_weight"] == 0.5 and resource["teacher_unchanged_and_no_gradients"]
            and resource["initial_model_state_sha256"] == plan["initial_model_state_sha256"], "Full resource proof differs")
    out = args.output_directory.absolute()
    run = Path(plan["run_dir"])
    require(out.parent == args.plan.resolve().parent and not out.exists() and not run.exists(), "Use fresh stage and run")
    require(disk_bytes(run.parent) + 350_000_000 < plan["artifact_allowance_bytes"], "Insufficient endpoint allowance")
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve prior monitor output")
    out.mkdir()
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_teacher",
                     "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256]}
    write(out / "watchdog-spec.json", spec)
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", "600", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "root-command.json", {"argv": argv, "plan_sha256": args.plan_sha256,
                                      "source_bindings": plan["source_bindings"]})
    print(json.dumps({"event": "launch", "teacher_weight": plan["teacher_weight"],
                      "stop_step": 250, "plan_sha256": args.plan_sha256}), flush=True)
    began = time.monotonic()
    with (out / "root-console.log").open("x") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(p) == h for p, h in plan["source_bindings"].items())
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": unchanged, "plan_sha256": args.plan_sha256}
    write(out / "root-execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and unchanged, "Training execution failed; inspect retained artifacts")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"] and read(run / "status.json")["status"] == "complete",
            "Bounded training or monitor did not complete")
    argv = [PYTHON, "-u", "-m", "research.direct.audit_latency58_teacher", "--plan", str(args.plan.resolve()),
            "--plan-sha256", args.plan_sha256, "--output", str(out / "checkpoint-audit.json")]
    bindings = {**plan["source_bindings"], str(args.plan.resolve()): args.plan_sha256}
    for name in ("model.pt", "optimizer.pt", "rng.pt", "receipt.json"):
        p = run / "endpoint" / name
        bindings[str(p)] = sha(p)
    execute(argv, out, "audit", 120, bindings, {"plan_sha256": args.plan_sha256})
    require(read(out / "checkpoint-audit.json")["status"] == "pass", "Separate endpoint audit failed")
    print(json.dumps({"event": "training_and_audit_pass", "step": 250, "teacher_weight": plan["teacher_weight"]}), flush=True)


if __name__ == "__main__":
    main()
