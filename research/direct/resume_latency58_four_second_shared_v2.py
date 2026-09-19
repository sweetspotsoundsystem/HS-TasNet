"""Resume the unchanged four-second trajectory after an audited event-worker interruption.

The original run is retained by rename. The qualified trainer's existing
packed-resume path restores raw/Adam/EMA/RNG and repeats unsaved updates.
No training arithmetic or health-monitor implementation is replaced here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_four_second_shared import validate_recipe, runtime_policy, budget_snapshot
from research.direct.run_latency58_four_second_shared import require_resource_result
from research.direct.latency58_four_second_shared_qualification import require_cpu_evidence, require_gpu_evidence
from research.direct.latency58_four_second_monitor import require_monitor_qualification, require_monitor_closed
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def prepare(original_path, audit_path, previous_monitor):
    original_path, audit_path, previous_monitor = (Path(p).resolve() for p in (original_path, audit_path, previous_monitor))
    original, audit, previous = read(original_path), read(audit_path), read(previous_monitor)
    validate_recipe(original)
    verify_inputs(original)
    verify_inputs(audit)
    require_cpu_evidence(original)
    require_gpu_evidence(original)
    require_monitor_qualification()
    root = Path(original["output_directory"])
    require(root == ARTIFACT_ROOT / "branch-four-second-015"
            and original_path == root / "plan-recovery001.json"
            and original["resume_checkpoint"]["snapshot"]["step"] == 750,
            "Require the first recovery of the selected trajectory")
    require(audit["status"] == "pass" and audit["step"] == 1900 and audit["journal_prefix_exact"]
            and audit["discarded_unsaved_updates"] == 39
            and audit["all_raw_adam_ema_rng_audits_pass"] and audit["cuda_initialized"] is False,
            "Require the complete saved-state audit")
    for pid in (81641, 81642, 81818, 81826):
        require(not Path(f"/proc/{pid}").exists(), "Historical owned process is present")
    idle = root / "host-idle-002"
    idle_result = read(idle / "result.json")
    verify_inputs(idle_result)
    require(idle_result["status"] == "pass" and idle_result["monitor_result"] == binding(previous_monitor)
            and idle_result["execution"] == binding(idle / "execution.json"), "Fresh idle evidence changed")
    require_monitor_closed(read(idle / "execution.json"), previous, final_step=15)
    host_path = root / "post-interruption-host-audit-002.json"
    host = read(host_path)
    verify_inputs(host)
    require(host["event_gap_fully_covered"] and not host["fault_records"]
            and host["boot_id"] == audit["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "Host continuity or boot changed after inspection")
    run, archive = root / "production-run", root / "production-run-interrupted-step1939"
    out, plan_path = root / "recovery-stage-002", root / "plan-recovery002.json"
    require(run.is_dir() and not run.is_symlink() and not archive.exists() and not out.exists()
            and not plan_path.exists(), "Preserve all existing recovery attempts")
    snapshot = dict(audit["checkpoint"])
    for field, hash_field in (("path", "sha256"), ("receipt", "receipt_sha256")):
        path = Path(snapshot[field])
        require(path.is_relative_to(run) and not path.is_symlink() and sha(path) == snapshot[hash_field],
                "Saved recovery binding changed")
    result_path = root / "resource-run-005/result.json"
    parity_path = root / "resource-run-005/grouped-vocal-gpu-parity.json"
    initial_path = root / "plan-shared001.json"
    require_resource_result(read(result_path), read(parity_path), read(initial_path), sha(initial_path))
    budget = budget_snapshot(original)
    require(budget["new_root_actual_bytes"] + 1_000_000_000 + 50_000_000 < budget["new_root_reserved_peak_bytes"],
            "Reserve both archived saves, current/pending generations and monitor growth")
    paths = (original_path, audit_path, previous_monitor, host_path, idle / "result.json",
             idle / "execution.json", Path(__file__).resolve(),
             ROOT / "research/direct/supervise_latency58_four_second_recovery_v2.py")
    source_bindings = {**original["source_bindings"], **audit["source_bindings"], **host["source_bindings"],
                       **idle_result["source_bindings"], **{str(p): sha(p) for p in paths}}
    require(not any(Path(p).is_relative_to(run) for p in source_bindings),
            "Renaming this run would move a frozen prerequisite")
    run.rename(archive)
    for field in ("path", "receipt"):
        snapshot[field] = str(archive / Path(snapshot[field]).relative_to(run))
    plan = {**original, "source_bindings": source_bindings,
        "resume_checkpoint": {"training_plan": binding(original_path), "snapshot": snapshot},
        "recovery_controller_module": "research.direct.resume_latency58_four_second_shared_v2",
        "operational_recovery": {"reason": "event-monitor worker exited; owners did not record terminal receipts before host restart",
            "last_saved_step": 1900, "interrupted_completed_step": 1939, "unsaved_updates_repeated": 39,
            "interrupted_run": str(archive), "raw_adam_ema_rng_restored": True,
            "training_arithmetic_changed": False, "historical_trainer_actual_exit_code": None,
            "historical_monitor_successful": False, "event_gap_fault_records": []}}
    validate_recipe(plan)
    verify_inputs(plan)
    write(plan_path, plan)
    out.mkdir()
    stage = {"schema": "latency58-direct-sdr-stage-v1", "plan_sha256": sha(plan_path),
        "resource_only": False, "stop_step": plan["config"]["steps"], "run_directory": str(run),
        "output_directory": str(out), "previous_event_record_id": previous["last_event_record_id"],
        "previous_monitor": binding(previous_monitor), "resource_result": binding(result_path),
        "source_bindings": {str(p): sha(p) for p in (plan_path, previous_monitor, audit_path, result_path, parity_path)},
        "storage_before": budget, "recovery_from_failed_predecessor": True}
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "expected_final_step": stage["stop_step"],
        "cwd": str(ROOT), "environment": plan["environment"], "progress_path": str(run / "metrics.jsonl"),
        "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_four_second_shared",
                 "--plan", str(plan_path), "--plan-sha256", sha(plan_path),
                 "--stage", str(out / "stage.json"), "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    print(json.dumps({"status": "prepared", "plan": binding(plan_path), "stage": binding(out / "stage.json"),
                      "snapshot": snapshot, "archive": str(archive)}), flush=True)
    return plan_path


def launch(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    validate_recipe(plan)
    verify_inputs(plan)
    require(plan["resume_checkpoint"]["snapshot"]["step"] == 1900, "Unexpected recovery endpoint")
    root = Path(plan["output_directory"])
    out = root / "recovery-stage-002"
    stage_path, spec_path = out / "stage.json", out / "watchdog-spec.json"
    stage, spec = read(stage_path), read(spec_path)
    verify_inputs(stage)
    require(stage["plan_sha256"] == sha(plan_path) and stage["stop_step"] == 2000
            and spec["argv"] == [PYTHON, "-u", "-m", "research.direct.train_latency58_four_second_shared",
                "--plan", str(plan_path), "--plan-sha256", sha(plan_path),
                "--stage", str(stage_path), "--stage-sha256", sha(stage_path)]
            and not Path(stage["run_directory"]).exists(), "Recovery launch changed or already ran")
    budget_snapshot(plan)
    monitor_out = ARTIFACT_ROOT / "monitors/branch-four-second-015-recovery-002"
    require(not monitor_out.exists(), "Preserve prior monitor output")
    runtime = runtime_policy()
    remaining = stage["stop_step"] - plan["resume_checkpoint"]["snapshot"]["step"]
    maximum = remaining * runtime["production_seconds_per_update"] + 30 * runtime["save_seconds_per_generation"] + 600
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(spec_path),
        "--launch-spec-sha256", sha(spec_path), "--output-dir", str(monitor_out),
        "--max-runtime-seconds", str(maximum), "--poll-seconds", "2", "--query-timeout-seconds", "10",
        "--startup-grace-seconds", str(runtime["production_startup_grace_seconds"]),
        "--progress-timeout-seconds", str(runtime["progress_timeout_seconds"]),
        "--finalization-timeout-seconds", "180", "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10",
        "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "plan_sha256": sha(plan_path), "runtime_policy": runtime})
    began = time.monotonic()
    with (out / "console.log").open("x") as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(out / "launch.json", {"controller_pid": os.getpid(), "monitor_pid": child.pid,
            "monitor_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
            "argv": argv, "plan_sha256": sha(plan_path)})
        code = child.wait()
        log.flush(); os.fsync(log.fileno())
    unchanged = all(sha(p) == digest for p, digest in {**plan["source_bindings"], **stage["source_bindings"]}.items())
    execution = {"actual_exit_code": code, "elapsed_seconds": time.monotonic() - began,
        "source_bindings_unchanged": unchanged, "plan_sha256": sha(plan_path),
        "monitor_result": str(monitor_out / "result.json"), "command_sha256": sha(out / "command.json")}
    write(out / "execution.json", execution)
    require(code == 0 and unchanged, "Monitored recovery failed")
    require_monitor_closed(execution, read(monitor_out / "result.json"), final_step=2000)
    print(json.dumps({"status": "pass", "execution": binding(out / "execution.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-plan", type=Path)
    parser.add_argument("--recovery-audit", type=Path)
    parser.add_argument("--previous-monitor", type=Path)
    parser.add_argument("--launch-plan", type=Path)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, "Use the research checkout")
    if args.launch_plan:
        require(not any((args.original_plan, args.recovery_audit, args.previous_monitor)), "Choose preparation or launch")
        launch(args.launch_plan)
    else:
        require(all((args.original_plan, args.recovery_audit, args.previous_monitor)), "Supply all preparation inputs")
        prepare(args.original_plan, args.recovery_audit, args.previous_monitor)


if __name__ == "__main__":
    main()
