"""Queue complete followup scoring once its model and parent scores are audited."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, disk_bytes
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_cleanup_followup import REFERENCES, load_views, reference_views


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def wait_for(identity, terminal, maximum_seconds):
    if terminal.is_file():
        return 0.0
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(identity["pid"], 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    try:
        proc = Path("/proc", str(identity["pid"]))
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [part.decode() for part in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
        require(int(stat[19]) == identity["start_ticks"] and argv == identity["argv"], "Supervisor identity changed")
        began = time.monotonic()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        while not poller.poll(30000):
            require(time.monotonic() - began < maximum_seconds, "Bounded prerequisite wait expired")
        require(terminal.is_file(), "Supervisor exited without its terminal receipt")
        return time.monotonic() - began
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require frozen CPU preparation")
    prospective = read(args.plan)
    verify_inputs(prospective)
    require(prospective["schema"] == "latency58-cleanup-followup-quality-preparation-v1"
            and prospective["maximum_wait_seconds"] == 12000, "Unexpected scoring scope")
    out = Path(prospective["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve queued scoring")
    write(out / "wait-start.json", {"plan_sha256": args.plan_sha256,
                                    "training_supervisor": prospective["training_supervisor"],
                                    "reference_supervisor": prospective["reference_supervisor"]})
    training_terminal = PHASE / "cleanup-followup-queue-001/result.json"
    reference_terminal = PHASE / "cleanup-successor-250-queue-001/result.json"
    waited = {
        "training": wait_for(prospective["training_supervisor"], training_terminal, prospective["maximum_wait_seconds"]),
        "reference": wait_for(prospective["reference_supervisor"], reference_terminal, prospective["maximum_wait_seconds"]),
    }
    completed, reference_completed = read(training_terminal), read(reference_terminal)
    require(completed["actual_exit_code"] == 0 and completed["source_bindings_unchanged"]
            and reference_completed["status"] == "pass" and reference_completed["source_bindings_unchanged"],
            "Training or reference quality did not finish successfully")
    verify_inputs(prospective)
    training_path = PHASE / "cleanup-followup-prep-001/training-plan.json"
    resource_path = training_path.parent / "resource-plan.json"
    training = read(training_path)
    verify_inputs(training)
    require(training["schema"] == "latency58-cleanup-followup-training-v1" and not training["resource_only"],
            "Different followup endpoint")
    stage = PHASE / "cleanup-followup-to-000250-001"
    audit = read(stage / "audit.json")
    require(audit["status"] == "pass" and audit["step"] == 250
            and audit["plan_sha256"] == sha(training_path), "Followup generation was not audited")
    bindings = {**training["source_bindings"], **prospective["source_bindings"], str(args.plan): args.plan_sha256}
    for path in (training_path, resource_path, training_terminal, reference_terminal):
        bindings[str(path)] = sha(path)
    for label, prefix in REFERENCES.items():
        for mode in ("full14", "actions60", "probes"):
            load_completed(PHASE / (prefix + "-" + mode + "-001"), bindings, canonical_baseline=label == "working")
        load_views(reference_views(label), bindings)
    counted = sum(disk_bytes(Path(path)) for path in training["counted_roots"])
    extra = 146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects")
    require(extra < 500_000_000 and counted + 400_000_000 < 79_500_000_000,
            "Complete scoring and the retained training reserve exceed the artifact cap")
    queue = PHASE / "cleanup-followup-250-queue-001"
    require(not queue.exists(), "Preserve previous followup scoring")
    queue.mkdir()
    reservation = {
        "schema": "latency58-cleanup-followup-views-reservation-v1",
        "evaluation_directories": [str(PHASE / "cleanup-followup-250-views-001")],
        "new_artifact_allowance_bytes": 10_000_000, "training_reserve_bytes": 350_000_000,
        "counted_bytes_at_preparation": counted, "counted_roots": training["counted_roots"],
        "stop_counted_bytes": training["stop_counted_bytes"], "source_bindings": bindings,
        "extra_artifact_bytes_outside_counted_roots": extra, "combined_cap_bytes": 80_000_000_000,
    }
    write(queue / "views-reservation.json", reservation)
    plan = {
        "schema": "latency58-cleanup-followup-queued-quality-plan-v1", "maximum_wait_seconds": 9000,
        "launch_next_training_arm": False, "output_directory": str(queue), "stage_directory": str(stage),
        "training_plan": binding(training_path), "resource_plan": binding(resource_path),
        "quality_prefix": "cleanup-followup-250", "supervisor": prospective["training_supervisor"],
        "views_reservation": binding(queue / "views-reservation.json"),
        "source_bindings": {**bindings, str(queue / "views-reservation.json"): sha(queue / "views-reservation.json")},
        "matched_training_effect_claimed": False,
        "confirmation_policy": "Development measurements only; old confirmation intervals are already seen.",
    }
    verify_inputs(plan)
    write(queue / "plan.json", plan)
    write(out / "wait-complete.json", {"waited_seconds": waited, "quality_plan": binding(queue / "plan.json")})
    command = [PYTHON, "-u", "-m", "research.direct.queue_latency58_cleanup_followup_quality",
               "--plan", str(queue / "plan.json"), "--plan-sha256", sha(queue / "plan.json")]
    write(out / "command.json", {"argv": command})
    print(json.dumps({"event": "followup_quality_start", "checkpoint": audit["checkpoint"]}), flush=True)
    began = time.monotonic()
    with (out / "quality-console.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=11000)
    verify_inputs(prospective)
    write(out / "result.json", {"actual_exit_code": result.returncode, "elapsed_seconds": time.monotonic() - began,
                               "source_bindings_unchanged": True, "quality_selected": False,
                               "quality_queue_plan": binding(queue / "plan.json")})
    require(result.returncode == 0, "Followup quality failed; retain execution evidence")


if __name__ == "__main__":
    main()
