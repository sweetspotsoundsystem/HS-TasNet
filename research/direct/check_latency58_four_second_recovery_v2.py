"""Audit the step-1900 save and host event gap before a second recovery.

Historical process exits remain unknown when their owners did not record them.
Checkpoint inspection is CPU-only; host queries use the qualified transports.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_four_second_shared import validate_recipe, budget_snapshot
from research.direct.latency58_four_second_monitor import require_monitor_qualification

OUT = ROOT / "research/four_second_20260916/branch-four-second-015"
PLAN = OUT / "plan-recovery001.json"
FAILED = OUT.parent / "monitors/branch-four-second-015-recovery-001"


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def checkpoint():
    import torch
    from research.direct.latency58_four_second_recovery_files import read_snapshot
    from research.direct.latency58_four_second_shared_qualification import require_cpu_evidence, require_gpu_evidence
    plan = read(PLAN)
    began = time.monotonic()
    validate_recipe(plan)
    verify_inputs(plan)
    require_cpu_evidence(plan)
    require_gpu_evidence(plan)
    require_monitor_qualification()
    for pid in (81641, 81642, 81818, 81826):
        require(not Path(f"/proc/{pid}").exists(), "Historical owned process is present")
    run = OUT / "production-run"
    receipt = run / "packed-recovery-receipts/step-001900.json"
    saved = {**binding(run / "recovery.packed.pt"), "receipt": str(receipt),
             "receipt_sha256": sha(receipt), "step": 1900}
    snapshot, audited = read_snapshot(saved, plan, sha(PLAN))
    journal = (run / "metrics.jsonl").read_bytes().splitlines(keepends=True)
    require(len(journal) == 1939 and snapshot["journal"] == b"".join(journal[:1900]),
            "Saved journal prefix differs from interrupted trajectory")
    require(len(audited[1]["optimizer"]["state"]) == 40 and audited[3].updates == 1900
            and not torch.cuda.is_initialized(), "Saved raw/Adam/EMA audit differs")
    budget = budget_snapshot(plan)
    require(budget["new_root_actual_bytes"] + 1_000_000_000 < budget["new_root_reserved_peak_bytes"],
            "Retained archives plus current and pending saves exceed reservation")
    result = {"status": "pass", "checkpoint": saved, "step": 1900,
        "journal_prefix_exact": True, "discarded_unsaved_updates": 39,
        "next_sample_index": plan["config"]["data_start"] + 1900 * 16,
        "all_raw_adam_ema_rng_audits_pass": True, "cuda_initialized": False,
        "historical_owned_pids_absent": True, "historical_actual_exit_codes": None,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "source_bindings": {**plan["source_bindings"], str(PLAN): sha(PLAN), str(Path(__file__).resolve()): sha(__file__)},
        "storage": budget, "elapsed_seconds": time.monotonic() - began}
    path = OUT / "post-interruption-recovery-audit-002.json"
    require(not path.exists(), "Preserve audit evidence")
    write(path, result)
    print(json.dumps({"status": "pass", "step": 1900, "audit": binding(path),
                      "new_root_actual_bytes": budget["new_root_actual_bytes"]}), flush=True)


def host():
    from research.direct import watch_latency58_four_second as monitor
    require_monitor_qualification()
    target = OUT / "post-interruption-host-audit-002.json"
    require(not target.exists(), "Preserve host evidence")
    last_events, last_gpu, last_baseline = None, None, None
    tail = deque(maxlen=4)
    with (FAILED / "watchdog.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            tail.append(row)
            if row["event"] == "event_coverage":
                last_events = row
            if row["event"] == "query" and row.get("kind") == "nvml":
                last_gpu = row
            if row["event"] == "baseline":
                last_baseline = row
    require(any(row["event"] == "stop_reason" and row["latest_completed_step_seen"] == 1939
                and row["reason"] == "RuntimeError('Event worker exited with code 1')" for row in tail),
            "Unexpected interrupted monitor endpoint")
    require(not (FAILED / "result.json").exists(), "Reconcile completed monitor instead")
    # The last validated coverage row is recorded separately from the query.
    require(last_events is not None, "No validated historical event coverage")
    previous = last_events["newest"]
    with (monitor.HERE / "gpu-watchdog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        event_worker, gpu_worker = monitor.load_event_transport(), None
        result = {"status": "incomplete", "previous_event_record_id": previous,
                  "last_validated_events": last_events, "last_gpu_query": last_gpu,
                  "last_baseline": last_baseline, "historical_tail": list(tail),
                  "historical_actual_exit_codes": None, "historical_monitor_successful": False,
                  "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        try:
            query = event_worker.query(monitor.event_query(previous, verify_sentinel=True), 10)
            result["event_query"] = query
            require(query["worker_still_running"] and query["actual_exit_code"] is None,
                    "Host event worker ended during query")
            payload = json.loads(query["stdout"])
            newest, rows = monitor.validate_events(payload, previous)
            require(payload["SentinelVerified"], "Historical TDR sentinel missing")
            result.update(last_event_record_id=newest, event_gap_fully_covered=True,
                          fault_records=[row for row in rows if monitor.reset_event(row)])
            gpu_worker = monitor.load_gpu_transport()
            result["gpu_query"] = gpu_worker.query(10)
            require(result["gpu_query"]["worker_still_running"], "GPU telemetry worker ended during query")
            gpu = monitor.parse_gpu(result["gpu_query"]["stdout"])
            require(monitor.gpu_alert(gpu, gpu, max_temperature=80, memory_headroom=4096) is None,
                    "Current GPU health is outside qualified bounds")
            require(result["gpu_query"]["nvml"]["memory_bytes"]["free"] >= 4096 * 2**20,
                    "Insufficient physical GPU memory")
            result.update(status="observed", current_gpu=gpu, gpu_workload_started=False)
        finally:
            result["event_worker_close"] = event_worker.close()
            if gpu_worker is not None:
                result["gpu_worker_close"] = gpu_worker.close()
            result["source_bindings"] = {str(FAILED / "watchdog.jsonl"): sha(FAILED / "watchdog.jsonl"),
                                         str(Path(__file__).resolve()): sha(__file__)}
            write(target, result)
    for name in ("event_worker_close", "gpu_worker_close"):
        value = result[name]
        require(value["closed"] and not value["forced"] and value["actual_exit_code"] == 0,
                "Host audit worker did not close normally")
    print(json.dumps({"status": result["status"], "audit": binding(target), "gpu": result["current_gpu"],
                      "fault_records": result["fault_records"], "last_event_record_id": newest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("checkpoint", "host"))
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Use controlled checkout")
    (checkpoint if args.mode == "checkpoint" else host)()
