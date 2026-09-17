"""Freeze and execute one monitored resource fixture for both context arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.train_latency58 import ROOT, read, require, sha, verify_inputs


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--samples", type=int, choices=(4096, 88064), required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--short-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    phase = ROOT / "research/direct/runs/latency58"
    out = args.output_directory.absolute()
    require(Path.cwd() == ROOT and out.parent == phase and not out.exists(), "Use a fresh resource directory")
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"], "Previous GPU stage did not finish successfully")
    require(sha(args.recipe) == args.recipe_sha256, "Context recipe changed")
    recipe = read(args.recipe)
    require(recipe["schema"] == "latency58-context-recipe-v1", "Unknown context recipe")
    verify_inputs(recipe)
    plan = dict(recipe)
    plan.update(schema="latency58-context-resource-v1", scored_samples=args.samples,
                warmup_samples=1024 if args.samples == 4096 else 88064,
                output_directory=str(out), previous_event_record_id=previous["last_event_record_id"])
    plan["source_bindings"] = dict(recipe["source_bindings"])
    bind = [args.recipe, args.previous_monitor, args.previous_execution, Path(__file__),
            ROOT / "research/direct/check_latency58_context_resource.py"]
    if args.samples == 88064:
        require(args.short_directory is not None, "Full resource fixture requires its short result")
        short = args.short_directory.resolve()
        plan.update(short_result=str(short / "result.json"), short_execution=str(short / "execution.json"))
        bind += [short / "plan.json", short / "result.json", short / "execution.json"]
        bind.append(Path(read(short / "execution.json")["monitor_result"]))
    for path in bind:
        plan["source_bindings"][str(path.resolve())] = sha(path)
    verify_inputs(plan)
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve previous GPU monitor evidence")
    out.mkdir()
    write(out / "plan.json", plan)
    python = "/home/axel/miniforge3/bin/python"
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(out / "progress.jsonl"),
            "argv": [python, "-u", "-m", "research.direct.check_latency58_context_resource",
                     "--plan", str(out / "plan.json"), "--plan-sha256", sha(out / "plan.json")]}
    write(out / "watchdog-spec.json", spec)
    argv = [python, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", "240", "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "cwd": str(ROOT), "plan_sha256": sha(out / "plan.json")})
    print(json.dumps({"event": "resource_launch", "teacher": plan["teacher_kind"], "samples": args.samples}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": all(sha(p) == s for p, s in plan["source_bindings"].items()),
                 "plan_sha256": sha(out / "plan.json"), "monitor_result": str(monitor_out / "result.json")}
    write(out / "execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and execution["source_bindings_unchanged"], "GPU resource execution failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass"
            and terminal["child_exit_code"] == 0 and terminal["post_exit_quiet_completed"]
            and read(out / "result.json")["status"] == "pass", "Resource result or monitor failed")


if __name__ == "__main__":
    main()
