"""Repeat the qualified pilot with enough total time for measured update cost."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_grouped_vocal_canonical import validate_recipe
from research.direct.run_latency58_grouped_vocal_runtime import require_resource_result
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.run_latency58_paired_vocal_views import binding, merge_bindings

SOURCE = PHASE / "branch-grouped-vocal-008"
OUT = PHASE / "branch-grouped-vocal-009"
ALLOWANCE = {"version": "canonical-observed-runtime-per-update60-plus600-v1", "resource_max_seconds": 1200,
             "production_seconds_per_update": 60, "production_fixed_allowance_seconds": 600}


def prepare():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and not OUT.exists(), "Prepare on CPU1 and preserve earlier attempts")
    required = [SOURCE / n for n in ("plan.json", "controlled-stop-request.json", "production-root-execution.json",
        "production-stage/execution.json", "production-run/metrics.jsonl", "resource-root-execution.json",
        "resource-stage/execution.json", "resource-run/result.json", "resource-run/grouped-vocal-gpu-parity.json")]
    require(all(p.is_file() for p in required), "Settle and retain the controlled stop before preparing another GPU stage")
    source, request, root, execution = (read(p) for p in required[:4])
    require(sha(SOURCE / "plan.json") == "1ccde8ac407102f06b355880e3b73fdc6dc2317d1337510ec1801e1edfbe7501",
            "Qualified source plan differs")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    rows = [json.loads(line) for line in required[4].read_text().splitlines()]
    require(root["actual_exit_code"] == execution["actual_exit_code"] == monitor["child_exit_code"] == 1
            and root["actual_session_id"] == 94178 and root["actual_tool_chunk_id"] == "40ba86"
            and root["status"] == "controlled_stop_before_runtime_allowance_retry"
            and root["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and root["controlled_stop_request_sha256"] == sha(required[1])
            and root["execution_sha256"] == sha(required[3]) and root["monitor_result_sha256"] == sha(monitor_path)
            and root["retained_journal_sha256"] == sha(required[4])
            and request["child_pid"] == monitor["child_pid"] and request["signal"] == "SIGTERM"
            and request["plan_sha256"] == sha(required[0]) and not request["scientific_recipe_change_requested"]
            and monitor["supervisor_health"] == "pass" and monitor["identities_unchanged"]
            and monitor["post_exit_quiet_completed"] and monitor["event_worker_close"]["actual_exit_code"] == 0
            and not monitor["event_worker_close"]["forced"] and not Path("/proc", str(monitor["child_pid"])).exists()
            and len(rows) == root["completed_training_updates"] == monitor["latest_completed_step_seen"] == 9
            and [r["step"] for r in rows] == list(range(1, 10))
            and not (SOURCE / "production-run/checkpoint").exists() and not (SOURCE / "production-run/checkpoint.pending").exists(),
            "Controlled stop, journal or host settlement differs")
    resource_root, resource_execution, resource, parity = (read(p) for p in required[5:])
    resource_monitor_path = Path(resource_execution["monitor_result"])
    resource_monitor = read(resource_monitor_path)
    require(resource_root["actual_exit_code"] == resource_execution["actual_exit_code"] == resource_monitor["child_exit_code"] == 0
            and resource_root["source_bindings_unchanged"] and resource_execution["source_bindings_unchanged"]
            and resource_root["resource_result_sha256"] == sha(required[7])
            and resource_root["gpu_parity_sha256"] == sha(required[8])
            and resource_monitor["status"] == resource_monitor["supervisor_health"] == "pass"
            and resource_monitor["post_exit_quiet_completed"] and resource_monitor["identities_unchanged"]
            and resource_monitor["finalization_started"]
            and resource_monitor["last_event_record_id"] <= monitor["last_event_record_id"], "Original resource proof differs")
    require_resource_result(resource, parity, source, sha(required[0]))
    for row, expected in zip(rows[:2], resource["matching_production_updates"], strict=True):
        normalized = {k: v for k, v in row.items() if k not in
                      ("data_wait_seconds", "compute_and_audit_seconds", "elapsed_seconds", "peak_vram_gib")}
        require(normalized == expected, "Stopped production did not reproduce its qualified two-update prefix")
    bindings = dict(source["source_bindings"])
    paths = [*required, monitor_path, resource_monitor_path, Path(__file__).resolve(),
             ROOT / "research/direct/run_latency58_grouped_vocal_runtime.py"]
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    before = budget_snapshot(source["storage_budget"])
    outside = before["external_git_common_bytes"] + source["storage_budget"]["other_outside_allowance_bytes"] + source["storage_budget"]["diagnostic_artifact_allowance_bytes"]
    plan = copy.deepcopy(source)
    plan.update(name=OUT.name, output_directory=str(OUT), source_bindings=bindings,
        runtime_allowance=ALLOWANCE, retry_of=binding(SOURCE / "plan.json"),
        retry_reason="Controlled stop after nine updates to add total-runtime margin for the measured canonical accumulation cost; mathematical objective, training data and all per-update/host/finalization guards unchanged.",
        prior_controlled_stop=binding(SOURCE / "production-root-execution.json"),
        prior_successful_resource=binding(SOURCE / "resource-root-execution.json"),
        previous_execution_for_resource=str(SOURCE / "resource-stage/execution.json"),
        event_continuity_scope="Fresh host-event scan from the last successful resource monitor includes the intervening documented controlled stop; latest stopped supervisor closed normally with unchanged identity and no host fault.",
        budget_before=before, outside_roots_reservation_bytes=outside, stop_counted_bytes=90_000_000_000 - outside,
        prior_unsaved_training_updates=9, replay_from_saved_selected_parent=True)
    validate_recipe(plan)
    require(all(plan[k] == source[k] for k in ("config", "parent_checkpoint", "parent_model_state_sha256",
        "initialized_model_state_sha256", "ema", "objective_version", "grouped_vocal_loss", "accumulation_policy",
        "qualified_data_prefix", "inference_architecture", "supervision")), "Runtime retry changed its scientific recipe")
    OUT.mkdir(); write(OUT / "plan.json", plan)
    print(json.dumps({"status": "prepared", "plan_sha256": sha(OUT / "plan.json"),
        "production_max_runtime_seconds": source["config"]["steps"] * 60 + 600,
        "gpu_workload_started": False}), flush=True)


if __name__ == "__main__":
    prepare()
