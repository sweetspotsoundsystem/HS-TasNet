"""Audit the saved 006 generation and recover CPU scoring after its failed teardown.

The original training/monitor exits stay failed. This operation neither reruns
training nor overwrites any original artifact, and makes no host-health claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

SOURCE = PHASE / "branch-long-context-006"
OUT = PHASE / "branch-long-context-recovery-006"
PLAN_SHA = "c6228b7a6d021728fc7b42397d53f4da4c1958a5130abb2dd5b30145c1ac8b22"
SCHEMA = "latency58-completed-generation-recovery-v1"


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def environment():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU processes")


def source_records():
    require(sha(SOURCE / "plan.json") == PLAN_SHA, "This recovery belongs only to the frozen 006 plan")
    source = read(SOURCE / "plan.json")
    training = read(SOURCE / "production-run/result.json")
    receipt = read(SOURCE / "production-run/checkpoint/receipt.json")
    execution = read(SOURCE / "production-stage/execution.json")
    root_execution = read(SOURCE / "root-execution.json")
    command = read(SOURCE / "root-command.json")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    require(root_execution["actual_exit_code"] == execution["actual_exit_code"] == 1
            and root_execution["actual_root_session"] == command["actual_root_session"] == 27228
            and root_execution["actual_tool_output_chunk"] == "1b6049"
            and root_execution["root_command_sha256"] == sha(SOURCE / "root-command.json")
            and root_execution["plan_sha256"] == execution["plan_sha256"] == PLAN_SHA
            and execution["source_bindings_unchanged"], "Original failed execution evidence differs")
    require(monitor["status"] == "stopped_by_watchdog" and monitor["supervisor_health"] == "failed"
            and monitor["reason"] == "RuntimeError('Owned child stopped reporting completed updates')"
            and monitor["child_exit_code"] == -15 and monitor["latest_completed_step_seen"] == 4000
            and monitor["identities_unchanged"] and not monitor["post_exit_quiet_completed"]
            and monitor["event_worker_close"]["closed"]
            and monitor["event_worker_close"]["actual_exit_code"] == 0, "Unexpected monitor failure")
    require(training["status"] == "pass" and training["checkpoint_written"]
            and training["source_bindings_unchanged"] and training["fixed_buffers_unchanged"]
            and training["updates"] == training["ema_updates"] == receipt["step"] == source["config"]["steps"] == 4000
            and training["plan_sha256"] == receipt["plan_sha256"] == PLAN_SHA
            and training["final_raw_model_state_sha256"] == receipt["raw_model_state_sha256"]
            and training["final_model_state_sha256"] == receipt["model_state_sha256"], "Incomplete saved training endpoint")
    require(training["checkpoint"] == binding(SOURCE / "production-run/checkpoint/model.pt"),
            "Saved inference binding differs")
    for pid in (command["actual_root_pid"], monitor["child_pid"], monitor["event_worker_close"]["linux_pid"]):
        require(not Path(f"/proc/{pid}").exists(), "Original owned process has not been reaped")
    return source, training, receipt, monitor_path, monitor


def prepare():
    environment()
    require(not OUT.exists(), "Preserve the existing recovery")
    source, training, receipt, monitor_path, monitor = source_records()
    paths = [SOURCE / n for n in ("plan.json", "root-command.json", "root-execution.json",
             "production-run/result.json", "production-run/metrics.jsonl", "production-stage/execution.json",
             "production-stage/stage.json", "production-stage/command.json", "production-stage/watchdog-spec.json")]
    paths.extend((SOURCE / "production-run/checkpoint").iterdir())
    paths.extend([monitor_path, Path(__file__).resolve()])
    paths.extend(Path(v["path"]) for v in monitor["artifacts"].values())
    budget_path = PHASE / "branch-gru-int8-post-ci-storage-001.json"
    paths.append(budget_path)
    bindings = dict(source["source_bindings"])
    for p in paths:
        digest = sha(p)
        require(str(p) not in bindings or bindings[str(p)] == digest, "Conflicting recovery input")
        bindings[str(p)] = digest
    storage = read(budget_path)
    plan = {"schema": SCHEMA, "source_training_root": str(SOURCE), "source_plan_sha256": PLAN_SHA,
            "output_directory": str(OUT), "source_bindings": bindings, "storage_budget": storage,
            "budget_before": budget_snapshot(storage), "roles": ["raw", "ema"],
            "original_root_exit_code": 1, "original_child_exit_code": -15,
            "recovery_scope": "Independent saved-generation integrity audit and unchanged CPU full14 scoring only",
            "training_replayed": False, "original_monitor_successful": False,
            "host_stability_proven": False, "quality_selected": False, "plugin_replaced": False}
    OUT.mkdir()
    write(OUT / "plan.json", plan)
    print(json.dumps({"event": "recovery_prepared", "plan": str(OUT / "plan.json"),
                      "plan_sha256": sha(OUT / "plan.json")}), flush=True)


def audit(plan):
    import torch
    from research.direct.latency58_branch_ema_checkpoint import audit_saved
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "Recovery must remain on CPU")
    source, training, receipt, monitor_path, monitor = source_records()
    metrics_path = SOURCE / "production-run/metrics.jsonl"
    require(sha(metrics_path) == receipt["metrics_sha256"], "Training journal differs from its save receipt")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    require(len(rows) == 4000, "Incomplete training journal")
    for index, row in enumerate(rows):
        require(row["step"] == row["ema_updates"] == index + 1
                and row["first_sample_index"] == source["config"]["data_start"] + index * 16
                and row["next_sample_index"] == source["config"]["data_start"] + (index + 1) * 16
                and row["adam_steps_this_update"] == row["gradient_clips_this_update"] == 1,
                "Training journal skipped or duplicated an update")
    require(rows[-1]["raw_model_state_sha256"] == training["final_raw_model_state_sha256"]
            and rows[-1]["ema_parameters_sha256"] == training["final_ema_parameters_sha256"],
            "Saved tensors do not match the final recorded update")
    log_path = Path(monitor["artifacts"]["watchdog_log"]["path"])
    require(sha(log_path) == monitor["artifacts"]["watchdog_log"]["sha256"], "Monitor log changed")
    faults, stops = [], []
    with log_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "stop_reason":
                stops.append(row)
            elif row.get("event") in ("query_timeout", "fresh_host_fault", "event_worker_error"):
                faults.append(row)
    require(len(stops) == 1 and not faults and stops[0]["latest_completed_step_seen"] == 4000
            and stops[0]["reason"] == monitor["reason"], "Additional monitor faults require separate investigation")
    alert = datetime.fromisoformat(stops[0]["utc"]).timestamp()
    save_times = {p.name: p.stat().st_mtime for p in (SOURCE / "production-run/checkpoint").iterdir()}
    result_mtime = (SOURCE / "production-run/result.json").stat().st_mtime
    require(max(save_times.values()) <= result_mtime < alert,
            "The completed generation and training result must precede the watchdog alert")
    result = audit_saved(training["checkpoint"], source, PLAN_SHA)
    require(result["step"] == 4000 and result["raw_model_state_sha256"] == training["final_raw_model_state_sha256"]
            and result["model_state_sha256"] == training["final_model_state_sha256"], "Saved audit endpoint differs")
    verify_inputs(plan)
    write(OUT / "checkpoint-audit.json", {**result, "source_bindings_unchanged": True,
          "recovery_plan_sha256": sha(OUT / "plan.json"), "training_journal_updates_verified": 4000,
          "save_file_mtimes": save_times, "training_result_mtime": result_mtime,
          "watchdog_alert_utc": stops[0]["utc"], "saved_generation_preceded_alert": True,
          "original_monitor_successful": False, "original_root_exit_code": 1,
          "host_stability_proven": False, "cpu_only": True})
    print(json.dumps({"event": "recovered_generation_audit_pass", "updates": 4000,
                      "original_monitor_successful": False}), flush=True)


def run(expected_sha):
    environment()
    require(sha(OUT / "plan.json") == expected_sha, "Recovery plan changed")
    plan = read(OUT / "plan.json")
    require(plan["schema"] == SCHEMA and plan["source_plan_sha256"] == PLAN_SHA, "Wrong recovery plan")
    verify_inputs(plan)
    budget_snapshot(plan["storage_budget"])
    audit(plan)
    source, training, receipt, _, _ = source_records()
    bindings = {**plan["source_bindings"], str(OUT / "plan.json"): expected_sha,
                str(OUT / "checkpoint-audit.json"): sha(OUT / "checkpoint-audit.json")}
    checkpoints, reports = {}, {}
    for role, filename in (("raw", "raw-model.pt"), ("ema", "model.pt")):
        checkpoint = binding(SOURCE / "production-run/checkpoint" / filename)
        quality = OUT / ("full14-" + role)
        quality.mkdir()
        qp = {"schema": "latency58-branch-memory-full14-plan-v1", "label": source["name"] + "-" + role,
              "checkpoint": checkpoint, "reference_result": source["reference_result"], "workers": 2,
              "track_indices": list(range(14)), "source_bindings": bindings, "output_directory": str(quality),
              "generation_recovery_audit": binding(OUT / "checkpoint-audit.json"),
              "original_monitor_successful": False}
        write(quality / "plan.json", qp)
        argv = [PYTHON, "-u", "-m", "research.direct.evaluate_latency58_branch_memory", "--plan",
                str(quality / "plan.json"), "--plan-sha256", sha(quality / "plan.json")]
        write(quality / "command.json", {"argv": argv})
        print(json.dumps({"event": "recovered_full14_started", "role": role}), flush=True)
        execute(argv, quality, "evaluation", 1800, bindings, {"plan_sha256": sha(quality / "plan.json")})
        report = read(quality / "result.json")
        require(report["status"] == "pass" and report["track_count"] == 14 and report["excerpt_count"] == 28
                and report["results"][0]["checkpoint"] == checkpoint, "Incomplete recovered full14 scoring")
        checkpoints[role], reports[role] = checkpoint, report
        budget_snapshot(plan["storage_budget"])
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    raw, averaged = (reports[role]["results"][0] for role in ("raw", "ema"))
    write(OUT / "paired-full14-comparison.json", {"comparison": compare(raw, averaged),
          "all_track_stem_cells": music_cells(raw, averaged), "quality_selected": False})
    verify_inputs(plan)
    result = {"schema": SCHEMA, "status": "recovered_saved_generation_audited_and_paired_full14_complete",
              "plan_sha256": expected_sha, "source_bindings_unchanged": True, "checkpoints": checkpoints,
              "full_sdr_db": {role: reports[role]["results"][0]["aggregate"]["full_sdr_db"] for role in reports},
              "quality_results": {role: binding(OUT / ("full14-" + role) / "result.json") for role in reports},
              "checkpoint_audit": binding(OUT / "checkpoint-audit.json"),
              "paired_comparison": binding(OUT / "paired-full14-comparison.json"),
              "original_training_root": str(SOURCE), "original_root_exit_code": 1,
              "original_child_exit_code": -15, "original_monitor_successful": False,
              "host_stability_proven": False, "training_replayed": False, "gpu_used": False,
              "quality_selected": False, "plugin_replaced": False, "budget_after": budget_snapshot(plan["storage_budget"])}
    write(OUT / "result.json", result)
    print(json.dumps({"event": "recovery_complete", "full_sdr_db": result["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    else:
        run(args.plan_sha256)
