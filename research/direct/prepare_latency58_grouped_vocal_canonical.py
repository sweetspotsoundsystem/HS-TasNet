"""Prepare the unchanged 500-update pilot with qualified canonical accumulation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.train_latency58_grouped_vocal_canonical import validate_recipe
from research.direct.latency58_grouped_vocal_canonical import policy
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.run_latency58_paired_vocal_views import binding, merge_bindings

SOURCE = PHASE / "branch-grouped-vocal-007"
DIAGNOSTIC = PHASE / "grouped-vocal-gradient-diagnostic-001"
CPU = PHASE / "grouped-vocal-canonical-cpu-001"
OUT = PHASE / "branch-grouped-vocal-008"


def prepare():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Prepare using CUDA-hidden CPU1")
    require(not OUT.exists(), "Preserve earlier canonical pilot plans")
    required = [CPU / (n + ".json") for n in ("plan", "result", "execution", "gradients", "restart")]
    require(all(p.is_file() for p in required), "Complete canonical CPU qualification before preparing GPU work")
    cpu_plan, cpu_result, cpu_execution, gradients, restart = (read(p) for p in required)
    require(cpu_result["status"] == gradients["status"] == restart["status"] == "pass"
            and cpu_result["source_bindings_unchanged"]
            and cpu_execution["actual_exit_code"] == cpu_execution["actual_enclosing_exit_code"] == 0
            and cpu_execution["source_bindings_unchanged"] and not cpu_execution["timed_out"]
            and cpu_execution["plan_sha256"] == cpu_result["plan_sha256"] == sha(CPU / "plan.json")
            and cpu_execution["result_sha256"] == sha(CPU / "result.json")
            and cpu_result["gradients_sha256"] == sha(CPU / "gradients.json")
            and cpu_result["restart_sha256"] == sha(CPU / "restart.json")
            and cpu_result["accumulation_policy"] == gradients["accumulation_policy"] == restart["accumulation_policy"] == policy()
            and len(gradients["all_40_gradients"]) == 40 and gradients["absolute_gradient_tolerance"] == 1e-7
            and gradients["relative_gradient_tolerance"] == 1e-4 and gradients["relative_l2_tolerance"] == 5e-5
            and all(row["relative_l2_error"] < 5e-5 and row["reference_norm"] > 0 for row in gradients["all_40_gradients"].values())
            and restart["all_40_adam_states_checked"] and restart["third_update_raw_adam_ema_and_accounting_bit_exact"]
            and restart["noncontiguous_step_rejected_before_gradients"] and len(restart["interrupted_accumulations"]) == 2
            and all(row["weights_adam_ema_unchanged"] for row in restart["interrupted_accumulations"])
            and cpu_result["parent_and_rng_unchanged"] and cpu_result["gpu_entry_rejected_before_cuda_initialization"]
            and not cpu_result["original_tolerances_changed"] and not cpu_result["gpu_used"],
            "Canonical CPU qualification differs")
    source = read(SOURCE / "plan.json")
    failed_root, failed_execution = (read(SOURCE / n) for n in ("resource-root-execution.json", "resource-stage/execution.json"))
    failed_monitor_path = Path(failed_execution["monitor_result"])
    failed_monitor = read(failed_monitor_path)
    require(sha(SOURCE / "plan.json") == "595aef9fbdb413a5a3aae79a76f9df5556f1219da7e6d6d04f16c94aad002d82"
            and failed_root["actual_exit_code"] == failed_execution["actual_exit_code"] == failed_monitor["child_exit_code"] == 1
            and failed_root["actual_tool_chunk_id"] == "9c3860" and failed_root["recorded_training_updates"] == 0
            and failed_execution["source_bindings_unchanged"] and failed_monitor["supervisor_health"] == "pass"
            and failed_monitor["latest_completed_step_seen"] is None
            and not (SOURCE / "resource-run/metrics.jsonl").exists() and not (SOURCE / "production-run").exists()
            and cpu_plan["fixture_model_state_sha256"] == source["initialized_model_state_sha256"],
            "Original failure or selected parent differs")
    diagnostic_inputs, diagnostic, diagnostic_execution, diagnostic_root = (read(DIAGNOSTIC / n) for n in
        ("inputs.json", "child-result.json", "execution.json", "root-execution.json"))
    monitor_path = Path(diagnostic_execution["monitor_result"])
    monitor = read(monitor_path)
    require(diagnostic["status"] == "diagnostic_complete" and diagnostic["source_bindings_unchanged"]
            and diagnostic_execution["actual_exit_code"] == diagnostic_root["actual_exit_code"] == 0
            and diagnostic_execution["source_bindings_unchanged"] and diagnostic_root["source_bindings_unchanged"]
            and diagnostic_root["child_result_sha256"] == sha(DIAGNOSTIC / "child-result.json")
            and diagnostic_root["execution_sha256"] == sha(DIAGNOSTIC / "execution.json")
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["last_event_record_id"] >= failed_monitor["last_event_record_id"]
            and monitor["identities_unchanged"] and monitor["post_exit_quiet_completed"] and monitor["finalization_started"]
            and monitor["source_sha256"] == source["supervision"]["watchdog_sha256"] == sha(source["watchdog_source"])
            and monitor["event_worker_close"]["actual_exit_code"] == 0 and not monitor["event_worker_close"]["forced"]
            and not Path("/proc", str(monitor["child_pid"])).exists(), "Diagnostic or real host monitor is incomplete")
    bindings = dict(source["source_bindings"])
    merge_bindings(bindings, diagnostic_inputs["source_bindings"])
    merge_bindings(bindings, cpu_plan["source_bindings"])
    paths = [*required, SOURCE / "plan.json", SOURCE / "resource-root-execution.json",
             SOURCE / "resource-stage/execution.json", failed_monitor_path, monitor_path, Path(__file__).resolve()]
    paths.extend(DIAGNOSTIC / n for n in ("inputs.json", "child-result.json", "result.json", "execution.json", "root-execution.json"))
    paths.extend(ROOT / "research/direct" / n for n in
        ("latency58_grouped_vocal_canonical.py", "check_latency58_grouped_vocal_canonical_cpu.py",
         "check_latency58_grouped_vocal_canonical_gpu.py", "train_latency58_grouped_vocal_canonical.py",
         "run_latency58_grouped_vocal_canonical.py"))
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    before = budget_snapshot(source["storage_budget"])
    outside = before["external_git_common_bytes"] + source["storage_budget"]["other_outside_allowance_bytes"] + source["storage_budget"]["diagnostic_artifact_allowance_bytes"]
    plan = copy.deepcopy(source)
    plan.update(name=OUT.name, output_directory=str(OUT), source_bindings=bindings,
        accumulation_policy=policy(), retry_of=binding(SOURCE / "plan.json"),
        retry_reason="The original grouped reduction differed from the original whole-group output derivatives by FP32 rounding, amplified in BF16 backward. Compute the unchanged whole-group loss once and replay its derivatives; retain original tolerances.",
        canonical_cpu_qualification=binding(CPU / "result.json"), original_resource_failure=binding(SOURCE / "resource-root-execution.json"),
        numerical_diagnostic=binding(DIAGNOSTIC / "child-result.json"),
        original_failed_resource_preserved=True, original_resource_updates=0,
        previous_execution_for_resource=str(DIAGNOSTIC / "execution.json"), budget_before=before,
        outside_roots_reservation_bytes=outside, stop_counted_bytes=90_000_000_000 - outside,
        training_context_implementation="Independent full warmup and score per ordinary/auxiliary render; whole-group output derivatives and exact microbatch replay",
        resource_context_comparison="Unchanged context checks and parameter tolerances, canonical replay equality, interruption and raw/Adam/EMA serialization/restart")
    validate_recipe(plan)
    require(plan["config"] == source["config"] and plan["parent_checkpoint"] == source["parent_checkpoint"]
            and plan["qualified_data_prefix"] == source["qualified_data_prefix"]
            and plan["ema"] == source["ema"] and plan["objective_version"] == source["objective_version"]
            and plan["grouped_vocal_loss"] == source["grouped_vocal_loss"]
            and not plan["inference_architecture_changed"] and not plan["automatic_plugin_replacement"],
            "Canonical retry changed its selected scientific recipe or inference")
    OUT.mkdir(); write(OUT / "plan.json", plan)
    print(json.dumps({"status": "prepared", "plan_sha256": sha(OUT / "plan.json"),
        "steps": plan["config"]["steps"], "parent_role": plan["parent_weight_role"], "gpu_workload_started": False}), flush=True)


if __name__ == "__main__":
    prepare()
