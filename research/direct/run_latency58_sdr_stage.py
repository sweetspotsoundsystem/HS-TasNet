"""Run one frozen SDR training stage, audit it, and retire replaced Adam state."""
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
    parser.add_argument("--stop-step", type=int, choices=(2, 25, 250, 500, 1000), required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-sdr-training-v1", "Unknown training plan")
    from research.direct.latency58_sdr_checkpoint import read_generation, require_space
    require_space(plan, 350_000_000)
    out, run = args.output_directory.absolute(), Path(plan["run_dir"])
    require(out.is_relative_to(ROOT / "research/direct/runs/latency58") and not out.exists(), "Use a fresh stage directory")
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"], "Previous GPU stage did not finish successfully")
    resource = read(plan["full_resource"]["path"])
    resource_execution = read(plan["full_resource_execution"]["path"])
    resource_monitor = read(resource_execution["monitor_result"])
    require(resource["status"] == "pass" and resource["teacher_kind"] == plan["teacher_kind"]
            and resource["samples"] == 88064 and resource["teacher_weight"] == plan["teacher_weight"]
            and resource["initial_model_state_sha256"] == plan["initial_model_state_sha256"]
            and resource_execution["actual_exit_code"] == 0 and resource_execution["source_bindings_unchanged"]
            and resource_monitor["status"] == "pass" and resource_monitor["post_exit_quiet_completed"],
            "Full GPU resource proof differs from the training arm")
    start = 0
    bind = [args.plan, args.previous_monitor, args.previous_execution]
    resume_fields = {}
    previous_generation = None
    if run.exists():
        status, pointer, audit_pointer = read(run / "status.json"), read(run / "latest.json"), read(run / "audit-latest.json")
        require(status["status"] == "paused" and not Path("/proc", str(status["pid"])).exists(),
                "Existing training arm is not terminal and paused")
        previous_generation = Path(pointer["generation"])
        receipt = read_generation(previous_generation, expected_plan_sha=args.plan_sha256)
        start = receipt["step"]
        require(pointer["step"] == status["step"] == audit_pointer["step"] == start
                and pointer["receipt_sha256"] == sha(previous_generation / "receipt.json")
                and audit_pointer["generation_receipt_sha256"] == pointer["receipt_sha256"],
                "Resume pointers disagree")
        resume_fields = {"resume_generation": str(previous_generation),
                         "resume_audit": audit_pointer["audit"], "resume_audit_execution": audit_pointer["execution"]}
        bind += [previous_generation / name for name in ("model.pt", "optimizer.pt", "rng.pt", "metrics.jsonl", "receipt.json")]
        bind += [Path(audit_pointer[k]["path"]) for k in ("audit", "execution")]
    require(start < args.stop_step <= plan["config"]["steps"], "Stage must advance the existing training arm")
    stage = {"schema": "latency58-sdr-stage-v1", "plan_sha256": args.plan_sha256,
             "start_step": start, "stop_step": args.stop_step, "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"],
             "previous_monitor": {"path": str(args.previous_monitor.resolve()), "sha256": sha(args.previous_monitor)},
             "source_bindings": {str(p.resolve()): sha(p) for p in bind}, **resume_fields}
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve previous GPU monitor output")
    out.mkdir()
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_sdr", "--plan", str(args.plan.resolve()),
                     "--plan-sha256", args.plan_sha256, "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    timeout = max(240, (args.stop_step - start) * 6 + 180)
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(timeout), "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "cwd": str(ROOT), "plan_sha256": args.plan_sha256,
                                 "stage_sha256": sha(out / "stage.json")})
    print(json.dumps({"event": "training_launch", "teacher": plan["teacher_kind"],
                      "start_step": start, "stop_step": args.stop_step}), flush=True)
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
    argv = [PYTHON, "-u", "-m", "research.direct.audit_latency58_sdr", "--plan", str(args.plan.resolve()),
            "--plan-sha256", args.plan_sha256, "--generation", str(generation), "--output", str(out / "audit.json")]
    execute(argv, out, "audit", 180, bindings, {"plan_sha256": args.plan_sha256})
    audit = read(out / "audit.json")
    require(audit["status"] == "pass" and audit["step"] == args.stop_step, "Saved-state audit failed")
    helpers = load_source("latency58_sdr_stage_helpers", plan["helper_source"])
    helpers.atomic_json(run / "audit-latest.json", {
        "step": args.stop_step, "generation_receipt_sha256": sha(generation / "receipt.json"),
        "audit": {"path": str(out / "audit.json"), "sha256": sha(out / "audit.json")},
        "execution": {"path": str(out / "audit-execution.json"), "sha256": sha(out / "audit-execution.json")}})
    if previous_generation is not None:
        # The new complete generation and independent audit are durable before
        # removing only the superseded optimizer. Every model and RNG remains.
        old_receipt = read_generation(previous_generation, expected_plan_sha=args.plan_sha256)
        old_optimizer = previous_generation / "optimizer.pt"
        retirement = {"schema": "latency58-sdr-replaced-optimizer-v1", "path": str(old_optimizer),
                      "sha256": sha(old_optimizer), "bytes": old_optimizer.stat().st_size,
                      "retained_model": str(previous_generation / "model.pt"),
                      "replacement_generation": str(generation), "replacement_audit_sha256": sha(out / "audit.json")}
        write(out / "retire-previous-optimizer-intent.json", retirement)
        require(retirement["sha256"] == old_receipt["files"]["optimizer.pt"]["sha256"], "Old optimizer changed")
        old_optimizer.unlink()
        require(sha(previous_generation / "model.pt") == old_receipt["files"]["model.pt"]["sha256"], "Old model changed")
        write(out / "retire-previous-optimizer-receipt.json", {"status": "complete", "freed_bytes": retirement["bytes"],
                                                              "intent_sha256": sha(out / "retire-previous-optimizer-intent.json"),
                                                              "retained_model_unchanged": True})
    print(json.dumps({"event": "training_and_audit_pass", "step": args.stop_step, "teacher": plan["teacher_kind"]}), flush=True)


if __name__ == "__main__":
    main()
