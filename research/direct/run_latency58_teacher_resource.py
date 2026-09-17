"""Freeze and execute one requested, monitored teacher resource fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.train_latency58 import ROOT, read, require, sha, verify_inputs


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, choices=(4096, 88064), required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    phase = ROOT / "research/direct/runs/latency58"
    functional = phase / "teacher-functional-001"
    authority = functional / "plan.json"
    require(sha(authority) == "92113a1d823546d5a4eba0847107233a794bd6935139e43780f552dc98a5a14f",
            "Frozen functional authority changed")
    plan = read(authority)
    verify_inputs(plan)
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"],
            "Previous GPU stage did not finish successfully")
    out = args.output_directory.absolute()
    require(Path.cwd() == ROOT and out.parent == phase and not out.exists(), "Use a fresh phase directory")
    base = read(phase / "asymmetric-gpu-full-001/plan.json")
    plan.update(schema="latency58-teacher-gpu-resource-v1", samples=args.samples,
                output_directory=str(out), teacher_weight=0.5,
                environment=base["environment"], watchdog_source=base["watchdog_source"],
                cached_batch=base["cached_batch"], torch_version=base["torch_version"],
                prerequisite_execution=str(functional / "check-execution.json"),
                functional_result=str(functional / "result.json"),
                previous_event_record_id=previous["last_event_record_id"],
                reason="Discard one teacher-plus-student update to verify gradients, unchanged teacher and bounded GPU cost.")
    bind = [authority, Path(__file__), ROOT / "research/direct/check_latency58_teacher_gpu.py",
            functional / "check-execution.json", functional / "result.json",
            args.previous_monitor, args.previous_execution, Path(base["watchdog_source"]), Path(base["cached_batch"])]
    if args.samples == 88064:
        short = phase / "teacher-gpu-short-001"
        require(read(short / "result.json")["status"] == "pass"
                and read(short / "plan.json")["samples"] == 4096
                and args.previous_execution.resolve() == short / "root-execution.json",
                "Full resource check requires the completed short fixture")
        plan["short_resource_terminal"] = str(args.previous_monitor.resolve())
        bind += [short / name for name in ("plan.json", "result.json")]
    for path in bind:
        plan["source_bindings"][str(path.resolve())] = sha(path)
    verify_inputs(plan)
    monitor_out = Path(base["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve previous monitor outputs")
    out.mkdir()
    write(out / "plan.json", plan)
    plan_sha = sha(out / "plan.json")
    python = "/home/axel/miniforge3/bin/python"
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(out / "progress.jsonl"),
            "argv": [python, "-u", "-m", "research.direct.check_latency58_teacher_gpu",
                     "--plan", str(out / "plan.json"), "--plan-sha256", plan_sha]}
    write(out / "watchdog-spec.json", spec)
    argv = [python, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", "240", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "root-command.json", {"argv": argv, "plan_sha256": plan_sha,
                                      "source_bindings": plan["source_bindings"]})
    print(json.dumps({"event": "launch", "samples": args.samples, "plan_sha256": plan_sha}), flush=True)
    began = time.monotonic()
    with (out / "root-console.log").open("x") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(path) == digest for path, digest in plan["source_bindings"].items())
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": unchanged, "plan_sha256": plan_sha}
    write(out / "root-execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and unchanged, "Resource execution failed; retain all evidence")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass"
            and terminal["child_exit_code"] == 0 and terminal["post_exit_quiet_completed"]
            and read(out / "result.json")["status"] == "pass", "Resource or monitor did not pass")


if __name__ == "__main__":
    main()
