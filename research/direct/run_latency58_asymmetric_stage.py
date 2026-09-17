"""Run one requested GPU stage, observe its exit, then audit its saved state.

This is a synchronous launcher, with no schedule or automatic continuation.
The caller must serialize it with other numeric experiments. The unchanged
watchdog owns the training process and enforces its resource and time bounds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.train_latency58 import ROOT, disk_bytes, read, require, sha, verify_inputs


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--previous-stage", type=Path)
    parser.add_argument("--previous-monitor", type=Path)
    parser.add_argument("--previous-execution", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--save-inference", action="store_true")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or working directory differs")
    plan = read(args.plan)
    verify_inputs(plan)
    run = Path(plan["run_dir"])
    require(plan["schema"] == "latency58-asymmetric-training-plan-v1", "Use a follow-up plan")
    require((args.previous_stage is not None) != (args.previous_monitor is not None),
            "Supply a previous stage for resume, or the last monitor for a fresh arm")
    audit_binding = None
    if args.previous_stage is not None:
        latest = read(run / "latest.json")
        previous = args.previous_stage.resolve(strict=True)
        audit_path = previous / "checkpoint-audit.json"
        audit, audit_execution = read(audit_path), read(previous / "audit-execution.json")
        previous_command = read(previous / "root-command.json")
        previous_execution = read(previous / "root-execution.json")
        command = previous_command["argv"]
        monitor_path = Path(command[command.index("--output-dir") + 1]) / "result.json"
        require(audit_execution["actual_exit_code"] == 0 and audit["status"] == "pass"
                and audit["step"] == latest["step"]
                and sha(run / "resume.pt") == latest["resume_sha256"] == audit["resume_sha256"]
                and audit["plan_sha256"] == latest["plan_sha256"] == args.plan_sha256,
                "Current resume does not match the separately audited checkpoint")
        audit_binding = {"path": str(audit_path), "sha256": sha(audit_path)}
    else:
        require(not run.exists() and args.previous_execution is not None,
                "A fresh arm needs a fresh run directory and the actual previous monitor execution")
        latest = {"step": 0, "resume_sha256": None}
        monitor_path = args.previous_monitor.resolve(strict=True)
        previous_execution = read(args.previous_execution)
        snapshot = read(plan["parent_snapshot_receipt"]["path"])
        require(sha(plan["parent_snapshot_receipt"]["path"]) == plan["parent_snapshot_receipt"]["sha256"]
                and snapshot["status"] == "pass" and snapshot["round_trip_model_state_exact"]
                and snapshot["output"]["sha256"] == sha(plan["parent_checkpoint"]["path"])
                == plan["parent_checkpoint"]["sha256"]
                and snapshot["model_state_sha256"] == plan["parent_model_state_sha256"],
                "Fresh parent is not the verified inference snapshot")
    monitor = read(monitor_path)
    require(previous_execution["actual_exit_code"] == 0 and previous_execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Previous monitored execution did not finish successfully")
    require(0 < args.stop_step - latest["step"] <= 250 and args.stop_step <= plan["config"]["steps"],
            "Request one stage of at most 250 updates within the training schedule")
    reserve = 350_000_000 + (115_000_000 if args.save_inference else 0)
    require(disk_bytes(ROOT / "research/direct/runs/latency58") + reserve < plan["artifact_allowance_bytes"],
            "Insufficient room for the requested atomic resume and optional snapshot")
    out = args.output_directory.absolute()
    require(out.parent == args.plan.absolute().parent and not out.exists(), "Use a fresh stage directory beside the plan")
    watchdog_output = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not watchdog_output.exists(), "Preserve previous monitor output")
    out.mkdir()
    stage = {"schema": "latency58-training-stage-v1", "plan_sha256": args.plan_sha256,
             "start_step": latest["step"], "stop_step": args.stop_step,
             "resume_sha256": latest["resume_sha256"],
             "previous_checkpoint_audit": audit_binding,
             "previous_monitor": {"path": str(monitor_path), "sha256": sha(monitor_path)},
             "previous_event_record_id": monitor["last_event_record_id"],
             "save_inference": args.save_inference}
    stage_path = out / "stage.json"
    write_new(stage_path, stage)
    python = "/home/axel/miniforge3/bin/python"
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT),
            "environment": plan["environment"], "progress_path": str(run / "metrics.jsonl"),
            "argv": [python, "-u", str(ROOT / "research/direct/train_latency58_asymmetric.py"),
                     "--plan", str(args.plan.absolute()), "--plan-sha256", args.plan_sha256,
                     "--stage", str(stage_path), "--stage-sha256", sha(stage_path)]}
    spec_path = out / "watchdog-spec.json"
    write_new(spec_path, spec)
    argv = [python, plan["watchdog_source"], "--launch-spec", str(spec_path),
            "--launch-spec-sha256", sha(spec_path), "--output-dir", str(watchdog_output),
            "--max-runtime-seconds", "600", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60",
            "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10",
            "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    root_command = {"argv": argv, "source_bindings": plan["source_bindings"],
                    "plan_path": str(args.plan.absolute()), "plan_sha256": args.plan_sha256,
                    "stage_sha256": sha(stage_path), "watchdog_spec_sha256": sha(spec_path),
                    "launcher_sha256": sha(__file__)}
    write_new(out / "root-command.json", root_command)
    print(json.dumps({"event": "launch", "start_step": latest["step"], "stop_step": args.stop_step,
                      "monitor_directory": str(watchdog_output)}), flush=True)
    started = time.monotonic()
    with (out / "root-console.log").open("x") as log:
        # The watchdog, rather than a competing outer timeout, terminates its
        # owned process group and completes the post-exit event observation.
        completed = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(path) == digest for path, digest in plan["source_bindings"].items())
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - started,
                 "source_bindings_unchanged": unchanged, "root_command_sha256": sha(out / "root-command.json"),
                 "stage_sha256": sha(stage_path)}
    write_new(out / "root-execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and unchanged, "GPU stage failed; inspect retained logs")
    monitor = read(watchdog_output / "result.json")
    require(monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Monitor did not finish successfully")
    latest = read(run / "latest.json")
    require(latest["step"] == args.stop_step and sha(run / "resume.pt") == latest["resume_sha256"],
            "Requested endpoint was not saved")
    argv = [python, "-u", "-m", "research.direct.audit_latency58_asymmetric",
            "--plan", str(args.plan.absolute()), "--plan-sha256", args.plan_sha256,
            "--resume-sha256", latest["resume_sha256"], "--output", str(out / "checkpoint-audit.json")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    timed_out = False
    with (out / "audit-console.log").open("x") as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=15)
    audit_execution = {"actual_exit_code": code, "timed_out": timed_out,
                       "elapsed_seconds": time.monotonic() - started, "argv": argv,
                       "resume_sha256_before": latest["resume_sha256"],
                       "resume_sha256_after": sha(run / "resume.pt")}
    write_new(out / "audit-execution.json", audit_execution)
    require(code == 0 and not timed_out and audit_execution["resume_sha256_after"] == latest["resume_sha256"],
            "Separate saved-state audit failed")
    print(json.dumps({"event": "stage_and_audit_pass", "step": args.stop_step,
                      "resume_sha256": latest["resume_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
