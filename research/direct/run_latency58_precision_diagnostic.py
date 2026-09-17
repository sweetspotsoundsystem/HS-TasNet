"""Freeze and supervise the fixed warm500 training-precision diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.train_latency58 import ROOT, read, require, sha, verify_inputs
from research.direct.run_latency58_quality import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--parent-plan-sha256", required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    phase = ROOT / "research/direct/runs/latency58"
    out = args.output_directory.resolve()
    require(Path.cwd() == ROOT and out.parent == phase and not out.exists(), "Use fresh diagnostic evidence")
    require(sha(args.parent_plan) == args.parent_plan_sha256, "Parent plan changed")
    parent = read(args.parent_plan)
    verify_inputs(parent)
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"] and execution["monitor_result"] == str(args.previous_monitor.resolve()),
            "Previous GPU stage is not closed successfully")
    plan = {"schema": "latency58-training-precision-diagnostic-v1", "batches": 8, "training_updates": 0,
            "parent_plan": {"path": str(args.parent_plan.resolve()), "sha256": args.parent_plan_sha256},
            "previous_monitor": {"path": str(args.previous_monitor.resolve()), "sha256": sha(args.previous_monitor)},
            "previous_execution": {"path": str(args.previous_execution.resolve()), "sha256": sha(args.previous_execution)},
            "previous_event_record_id": previous["last_event_record_id"], "output_directory": str(out),
            "hypothesis": "The same frozen trained model may perform differently under BF16 training arithmetic and FP32 inference.",
            "scope": "32 recorded training crops, fixed weights, forward-only; no validation score or training memory qualification.",
            "source_bindings": dict(parent["source_bindings"])}
    for key in ("environment", "torch_version", "watchdog_source", "counted_roots", "stop_counted_bytes"):
        plan[key] = parent[key]
    journal = Path(parent["run_dir"]) / "checkpoints/step-000250/metrics.jsonl"
    plan["recorded_journal"] = {"path": str(journal), "sha256": sha(journal)}
    for path in (args.parent_plan, args.previous_monitor, args.previous_execution, Path(__file__), journal,
                 ROOT / "research/direct/diagnose_latency58_training_precision.py", ROOT / "research/metrics.py",
                 ROOT / "research/eval_config.json", phase / "sdr-softcap-and-gate-terminal-decision-001/decision.json"):
        plan["source_bindings"][str(path.resolve())] = sha(path)
    verify_inputs(plan)
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 2_000_000)
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve earlier monitor evidence")
    out.mkdir()
    write(out / "plan.json", plan)
    python = "/home/axel/miniforge3/bin/python"
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(out / "progress.jsonl"),
            "argv": [python, "-u", "-m", "research.direct.diagnose_latency58_training_precision",
                     "--plan", str(out / "plan.json"), "--plan-sha256", sha(out / "plan.json")]}
    write(out / "watchdog-spec.json", spec)
    argv = [python, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", "360", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "cwd": str(ROOT), "plan_sha256": sha(out / "plan.json")})
    print(json.dumps({"event": "precision_diagnostic_launch", "training_updates": 0, "examples": 32}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": all(sha(p) == s for p, s in plan["source_bindings"].items()),
                 "plan_sha256": sha(out / "plan.json"), "monitor_result": str(monitor_out / "result.json")}
    write(out / "execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and execution["source_bindings_unchanged"], "GPU diagnostic execution failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass"
            and terminal["child_exit_code"] == 0 and terminal["post_exit_quiet_completed"]
            and read(out / "result.json")["status"] == "pass", "Diagnostic or monitor result failed")


if __name__ == "__main__":
    main()
