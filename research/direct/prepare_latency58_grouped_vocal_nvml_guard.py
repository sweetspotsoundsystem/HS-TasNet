"""Bind the audited step-150 recovery to the qualified persistent NVML monitor."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_grouped_vocal_nvml_guard import validate_recipe
from research.direct.recover_latency58_nvml_guard_idle import (
    OUT as IDLE, WATCHDOG, WATCHDOG_SHA, require_monitor_qualification)
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.run_latency58_paired_vocal_views import binding, merge_bindings

SOURCE = PHASE / "branch-grouped-vocal-011"
OUT = PHASE / "branch-grouped-vocal-012"


def prepare():
    require(Path.cwd() == ROOT and not OUT.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Prepare on CPU1 with CUDA hidden and preserve earlier attempts")
    source_path = SOURCE / "plan.json"
    source = read(source_path)
    require(sha(source_path) == "20962649939a648735b36b3e69978c5c74eddea5f6cb6399408a425190db024e",
            "Original full-500 plan changed")
    verify_inputs(source)
    failed_root = read(SOURCE / "production-root-execution.json")
    failed_execution = read(SOURCE / "production-stage/execution.json")
    failed_path = Path(failed_execution["monitor_result"])
    failed = read(failed_path)
    require(failed_root["actual_session_id"] == 54158 and failed_root["actual_tool_chunk_id"] == "a79ba7"
            and failed_root["actual_exit_code"] == failed_execution["actual_exit_code"] == failed["child_exit_code"] == 1
            and failed_root["source_bindings_unchanged"] and failed_execution["source_bindings_unchanged"]
            and failed_root["execution_sha256"] == sha(SOURCE / "production-stage/execution.json")
            and failed_root["monitor_result_sha256"] == sha(failed_path)
            and failed_root["plan_sha256"] == sha(source_path)
            and failed_root["retained_journal_sha256"] == sha(SOURCE / "production-run/metrics.jsonl")
            and failed["latest_completed_step_seen"] == failed_root["completed_training_updates"] == 171
            and failed["reason"] == "RuntimeError('nvml telemetry timed out')"
            and not Path("/proc", str(failed["child_pid"])).exists()
            and not Path("/proc", str(failed["event_worker_close"]["linux_pid"])).exists()
            and not (SOURCE / "production-run/checkpoint").exists()
            and not (SOURCE / "production-run/checkpoint.pending").exists()
            and not (SOURCE / "production-run/recovery.pending.pt").exists(), "Failed attempt differs")
    audit = read(SOURCE / "post-alert-recovery-audit.json")
    audit_execution = read(SOURCE / "post-alert-recovery-audit-execution.json")
    snapshot = audit["binding"]
    require(audit_execution["actual_exit_code"] == 0 and not audit_execution["timed_out"]
            and audit_execution["actual_session_id"] == 17753 and audit_execution["actual_tool_chunk_id"] == "87c93e"
            and audit_execution["result_sha256"] == sha(SOURCE / "post-alert-recovery-audit.json")
            and audit_execution["plan_sha256"] == audit["plan_sha256"] == sha(source_path)
            and audit_execution["failed_root_execution_sha256"] == audit["failed_root_execution_sha256"]
            == sha(SOURCE / "production-root-execution.json"), "Recovery audit did not close successfully")
    require(audit["status"] == "pass" and audit["checkpoint_step"] == snapshot["step"] == 150
            and audit["unchanged_from_pre_alert_observation"] and audit["cpu_loaded_and_audited"]
            and audit["saved_optimizer_states"] == 40 and audit["all_saved_adam_steps"] == 150
            and audit["optimizer_owner"] == "raw" and audit["raw_ema_and_journal_identities_match"]
            and audit["full_500_update_schedule_preserved"] and not audit["gpu_initialized"]
            and audit["absolute_next_sample_index"] == 4094400 and audit["algorithmic_latency_samples"] == 256
            and snapshot == failed_root["recovery_binding"]
            and sha(snapshot["path"]) == snapshot["sha256"]
            == "7795967819d80c0a27f68bae49c76cc9674df1a12649658d7f3807665e713413"
            and sha(snapshot["receipt"]) == snapshot["receipt_sha256"], "Audited recovery changed")
    idle_inputs, idle, idle_execution, idle_root = (read(IDLE / name) for name in
        ("inputs.json", "result.json", "execution.json", "root-execution.json"))
    verify_inputs(idle_inputs)
    idle_path = Path(idle_execution["monitor_result"])
    terminal = read(idle_path)
    require(idle_root["actual_session_id"] == 87346 and idle_root["actual_tool_chunk_id"] == "9b26f1"
            and idle_root["actual_exit_code"] == idle_execution["actual_exit_code"] == terminal["child_exit_code"] == 0
            and idle_root["execution_sha256"] == sha(IDLE / "execution.json")
            and idle_root["result_sha256"] == sha(IDLE / "result.json")
            and idle_root["monitor_result_sha256"] == sha(idle_path)
            and idle_execution["source_bindings_unchanged"] and idle["source_bindings_unchanged"]
            and idle["status"] == terminal["status"] == terminal["supervisor_health"] == "pass"
            and terminal["source_sha256"] == WATCHDOG_SHA and terminal["identities_unchanged"]
            and terminal["post_exit_quiet_completed"] and terminal["finalization_started"]
            and 60 < terminal["finalization_elapsed_seconds"] < 180
            and idle_inputs["previous_event_record_id"] == failed["last_event_record_id"] == 65004,
            "Fresh host idle is incomplete")
    for name in ("event_worker_close", "gpu_worker_close"):
        closed = terminal[name]
        require(closed["closed"] and closed["actual_exit_code"] == 0 and not closed["forced"]
                and not Path("/proc", str(closed["linux_pid"])).exists(), "Idle worker did not close normally")
    require(terminal["gpu_worker_close"]["identities_unchanged"]
            and not Path("/proc", str(terminal["child_pid"])).exists(), "Idle identity or child differs")
    old_trainer = ROOT / "research/direct/train_latency58_grouped_vocal_nvml.py"
    new_trainer = ROOT / "research/direct/train_latency58_grouped_vocal_nvml_guard.py"
    trees = [ast.parse(p.read_text()) for p in (old_trainer, new_trainer)]
    for name in ("applied_policy", "validate_recipe"):
        nodes = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
        require(ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False),
                "Scientific policy function changed: " + name)
    paths = [Path(__file__).resolve(), new_trainer, ROOT / "research/direct/run_latency58_grouped_vocal_nvml_guard.py",
             source_path, failed_path, idle_path, Path(snapshot["path"]), Path(snapshot["receipt"])]
    paths.extend(SOURCE / name for name in ("production-root-execution.json", "production-stage/execution.json",
        "production-run/metrics.jsonl", "post-alert-recovery-audit.json", "post-alert-recovery-audit-execution.json"))
    paths.extend(IDLE / name for name in ("inputs.json", "result.json", "execution.json", "root-execution.json",
        "child-result.json", "event-continuity.json"))
    bindings = dict(source["source_bindings"])
    merge_bindings(bindings, require_monitor_qualification())
    merge_bindings(bindings, idle_inputs["source_bindings"])
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    storage_budget = copy.deepcopy(source["storage_budget"])
    # Completed diagnostics are counted at actual size. Reserve 250 MB for new
    # diagnostics while this serialized continuation runs; downloads and builds
    # need separate budget accounting. Preserve the 600 MB save and 800 MB
    # outside-artifact reserves and the exact 90 GB cap.
    require(storage_budget["diagnostic_artifact_allowance_bytes"] == 500_000_000,
            "Previous diagnostic reservation changed")
    storage_budget["diagnostic_artifact_allowance_bytes"] = 250_000_000
    before = budget_snapshot(storage_budget)
    require(before["headroom_bytes"] > 1_350_000_000,
            "Reserve two 600 MB generations and 150 MB of logs beyond all standing reserves")
    outside = (before["external_git_common_bytes"] + storage_budget["other_outside_allowance_bytes"]
               + storage_budget["diagnostic_artifact_allowance_bytes"])
    plan = copy.deepcopy(source)
    plan.update(name=OUT.name, output_directory=str(OUT), source_bindings=bindings,
        retry_of=binding(source_path),
        retry_reason="Resume the audited step-150 snapshot using the qualified required-health NVML reader after the supplemental utilization query timed out; first reproduce updates 151 through 153 under the original full-500 schedule.",
        prior_failed_training=binding(SOURCE / "production-root-execution.json"),
        recovered_host_idle=binding(IDLE / "root-execution.json"),
        post_alert_recovery_audit=binding(SOURCE / "post-alert-recovery-audit.json"),
        previous_execution_for_resource=str(IDLE / "execution.json"),
        watchdog_source=str(WATCHDOG), supervision={"version": "persistent-nvml-planned-final-step-v1",
            "finalization_timeout_seconds": 180, "watchdog_sha256": WATCHDOG_SHA},
        resume_checkpoint={"step": 150, "snapshot": snapshot, "training_plan": binding(source_path)},
        resume_prefix_qualification={"start_step": 150, "stop_step": 153,
                                    "full_schedule_steps": 500, "save_quality_checkpoint": False},
        replay_reference=binding(SOURCE / "production-run/metrics.jsonl"),
        event_continuity_scope="Fresh resource continues the successful NVML idle, whose child checked all host records from the failed attempt's last successful poll.",
        storage_budget=storage_budget,
        continuation_storage_forecast={"new_rolling_generation_bytes": 600_000_000,
            "new_final_generation_bytes": 600_000_000, "logs_and_metadata_bytes": 150_000_000,
            "projected_total_including_standing_reserves": before["conservative_total_with_reservations"] + 1_350_000_000,
            "previous_diagnostic_reservation_bytes": 500_000_000, "new_diagnostic_reservation_bytes": 250_000_000,
            "no_baseline_or_failed_attempt_removed": True,
            "new_downloads_and_builds_require_separate_budget_accounting": True},
        budget_before=before, outside_roots_reservation_bytes=outside, stop_counted_bytes=90_000_000_000 - outside,
        prior_unsaved_training_updates=21, replay_from_saved_selected_parent=False)
    validate_recipe(plan)
    require(all(plan[key] == source[key] for key in ("config", "parent_checkpoint", "parent_model_state_sha256",
        "initialized_model_state_sha256", "ema", "objective_version", "grouped_vocal_loss", "accumulation_policy",
        "qualified_data_prefix", "inference_architecture", "runtime_allowance", "recovery_checkpoint")),
        "NVML recovery changed the scientific trajectory")
    verify_inputs(plan)
    OUT.mkdir(); write(OUT / "plan.json", plan)
    print(json.dumps({"status": "prepared", "plan_sha256": sha(OUT / "plan.json"),
        "resumed_from_step": 150, "full_schedule_steps": 500,
        "gpu_workload_started": False, "budget_before": before}), flush=True)


if __name__ == "__main__":
    prepare()
