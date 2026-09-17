"""Wait on a live pilot supervisor, then execute its complete CPU quality bundle."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import select
import subprocess
import time

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_vocal_focus import REFERENCES, load_views
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and Path.cwd() == ROOT, "Queue plan or cwd changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-vocal-focus-queued-quality-plan-v1"
            and plan["maximum_wait_seconds"] == 9000 and not plan["launch_next_training_arm"], "Unexpected queued work")
    out, stage = Path(plan["output_directory"]), Path(plan["stage_directory"])
    require(out.is_dir() and out.is_relative_to(PHASE) and not (out / "result.json").exists(), "Preserve queued work")
    training_binding = plan["training_plan"]
    require(sha(training_binding["path"]) == training_binding["sha256"], "Training plan changed")
    training = read(training_binding["path"])
    require(training["schema"] == "latency58-vocal-focus-training-v1" and not training["resource_only"],
            "Only a production pilot may queue quality")
    prefix = plan["quality_prefix"]
    require(prefix == "vocal-focus-" + training["arm"].replace("_", "-") + "-250", "Different quality prefix")
    waited = 0.0
    root_identity = plan["supervisor"]
    if not ((stage / "execution.json").is_file() and (stage / "audit-execution.json").is_file()):
        pid = root_identity["pid"]
        # This Conda Python does not expose os.pidfd_open; the host libc does.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        libc.pidfd_open.restype = ctypes.c_int
        fd = libc.pidfd_open(pid, 0)
        if fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        try:
            # pidfd observes this exact process even if the numeric PID is reused.
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            command = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            command = [part.decode() for part in command]
            require(int(stat[19]) == root_identity["start_ticks"] and command == root_identity["argv"]
                    and command[command.index("-m") + 1] == "research.direct.run_latency58_vocal_focus_stage"
                    and command[command.index("--plan") + 1] == training_binding["path"]
                    and command[command.index("--output-directory") + 1] == str(stage), "Different live supervisor")
            write(out / "wait-start.json", {"supervisor": root_identity, "pidfd_opened": True,
                                            "plan_sha256": args.plan_sha256, "wall_time": time.time()})
            began = time.monotonic()
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            while not poller.poll(30000):
                require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded supervisor wait expired")
            waited = time.monotonic() - began
        finally:
            os.close(fd)
    require((stage / "execution.json").is_file() and (stage / "audit-execution.json").is_file(),
            "Supervisor exited without completed training and audit receipts")
    execution, audit_execution, audit = (read(stage / name) for name in ("execution.json", "audit-execution.json", "audit.json"))
    monitor = read(execution["monitor_result"])
    require(execution["actual_exit_code"] == audit_execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"] and audit_execution["source_bindings_unchanged"]
            and not audit_execution["timed_out"] and audit["status"] == "pass"
            and audit["step"] == 250 and audit["arm"] == training["arm"]
            and execution["plan_sha256"] == audit_execution["plan_sha256"] == training_binding["sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Pilot or original saved-state audit failed; no quality work is launched")
    verify_inputs(plan)
    write(out / "wait-complete.json", {"waited_seconds": waited, "training_execution_sha256": sha(stage / "execution.json"),
                                       "audit_execution_sha256": sha(stage / "audit-execution.json"), "actual_pilot_exit_code": 0})
    commands = [
        [PYTHON, "-u", "-m", "research.direct.run_latency58_vocal_focus_quality",
         "--training-plan", training_binding["path"], "--training-plan-sha256", training_binding["sha256"],
         "--stage-directory", str(stage), "--output-prefix", prefix, "--modes", "full14", "probes", "actions60"],
        [PYTHON, "-u", "-m", "research.direct.run_latency58_vocal_focus_views", "--prefix", prefix,
         "--reservation", plan["views_reservation"]["path"], "--reservation-sha256", plan["views_reservation"]["sha256"]],
    ]
    for label, argv in zip(("music-and-probes", "vocal-views"), commands, strict=True):
        write(out / (label + "-command.json"), {"argv": argv})
        began = time.monotonic()
        print({"event": "quality_start", "arm": training["arm"], "phase": label}, flush=True)
        with (out / (label + "-console.log")).open("x") as stream:
            completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        write(out / (label + "-execution.json"), {"actual_exit_code": completed.returncode,
                                                "elapsed_seconds": time.monotonic() - began})
        require(completed.returncode == 0, "Quality phase failed; preserve its logs and partial outputs")
    evidence = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    for label, reference_prefix in {"candidate": prefix, **REFERENCES}.items():
        for mode in ("full14", "actions60", "probes"):
            load_completed(PHASE / (reference_prefix + "-" + mode + "-001"), evidence,
                           canonical_baseline=label == "working")
        load_views(PHASE / (prefix + "-views-001" if label == "candidate" else "vocal-views-" + label + "-001"), evidence)
    summary_dir = PHASE / (prefix + "-summary-001")
    require(not summary_dir.exists(), "Preserve existing summary")
    summary_plan = {"schema": "latency58-vocal-focus-summary-plan-v1", "candidate_prefix": prefix,
                    "output_directory": str(summary_dir), "source_bindings": evidence}
    summary_dir.mkdir()
    write(summary_dir / "plan.json", summary_plan)
    execute([PYTHON, "-u", "-m", "research.direct.report_latency58_vocal_focus",
             "--plan", str(summary_dir / "plan.json"), "--plan-sha256", sha(summary_dir / "plan.json")],
            summary_dir, "summary", 240, evidence, {"plan_sha256": sha(summary_dir / "plan.json")})
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-vocal-focus-queued-quality-result-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "summary": {"path": str(summary_dir / "result.json"), "sha256": sha(summary_dir / "result.json")},
        "quality_selected": False, "next_training_arm_launched": False, "optimizer_retired": False,
        "human_listening_completed": False})
    print({"event": "complete_quality_ready_for_review", "arm": training["arm"], "quality_selected": False}, flush=True)


if __name__ == "__main__":
    main()
