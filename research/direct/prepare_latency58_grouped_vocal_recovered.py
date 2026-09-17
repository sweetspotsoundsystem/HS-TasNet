"""Prepare one grouped-vocal pilot only after complete recovered-pair review."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import os

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.prepare_latency58_recovered_vocal_views import load_endpoint, SOURCE, RECOVERY
from research.direct.run_latency58_paired_vocal_views import binding, merge_bindings
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.train_latency58_grouped_vocal import validate_recipe, applied_policy
from research.direct.latency58_grouped_vocal_auxiliary import VERSION

VIEWS = PHASE / "paired-vocal-long-context-006"
IDLE = PHASE / "branch-long-context-finalization-idle-007"
OUT = PHASE / "branch-grouped-vocal-007"
MONITOR = PHASE.parent / "latency11/smoke/gpu-crash-followup/watch_gpu_process_finalization.py"
MONITOR_SHA = "58832a30ddd9c38e603273638834546eec4978623cf3d88c25990a6d08be840e"


def qualified(directory, bindings):
    plan, result, execution = (read(directory / (n + ".json")) for n in ("plan", "result", "execution"))
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == execution["actual_enclosing_exit_code"] == 0
            and execution["source_bindings_unchanged"] and not execution["timed_out"]
            and result["plan_sha256"] == execution["plan_sha256"] == sha(directory / "plan.json")
            and execution["result_sha256"] == sha(directory / "result.json"), "CPU qualification is incomplete")
    merge_bindings(bindings, plan["source_bindings"])
    merge_bindings(bindings, {str(directory / (n + ".json")): sha(directory / (n + ".json"))
                              for n in ("plan", "result", "execution")})
    return plan, result


def completed_review(selection_path):
    required = [selection_path, VIEWS / "plan.json", VIEWS / "result.json", VIEWS / "root-execution.json"]
    require(all(p.is_file() for p in required), "Complete and review both vocal endpoints before preparing a GPU pilot")
    selection, pair_plan, pair, pair_execution = (read(p) for p in required)
    require(pair["status"] == "pass" and pair["source_bindings_unchanged"]
            and pair["track_count_per_endpoint"] == 14 and pair["excerpt_count_per_view_per_endpoint"] == 28
            and pair_execution["actual_exit_code"] == 0 and pair_execution["source_bindings_unchanged"]
            and not pair_execution["timed_out"] and pair_execution["result_sha256"] == sha(VIEWS / "result.json")
            and pair["plan_sha256"] == pair_execution["plan_sha256"] == sha(VIEWS / "plan.json")
            and set(pair["models"]) == set(pair["reports"]) == {"raw", "ema"}, "Paired vocal evidence is incomplete")
    require(selection["schema"] == "latency58-grouped-vocal-parent-selection-v1"
            and selection["status"] == "selected_for_guarded_pilot"
            and selection["paired_vocal_result"] == binding(VIEWS / "result.json")
            and selection["paired_vocal_execution"] == binding(VIEWS / "root-execution.json")
            and selection["full14_parent_review"] == binding(RECOVERY / "full14-parent-review.json")
            and selection["all_track_stem_and_worst_windows_reviewed"]
            and selection["objective_version"] == VERSION and selection["grouped_vocal_loss"] == applied_policy()
            and not selection["overall_goal_complete"] and not selection["plugin_replaced"]
            and not selection["original_training_monitor_successful"], "Selection record is incomplete or changes the objective")
    role = selection["selected_role"]
    require(role in ("raw", "ema") and selection["selected_checkpoint"] == pair["models"][role]["checkpoint"]
            and selection["selected_model_state_sha256"] == pair["models"][role]["model_state_sha256"],
            "Selected parent differs from the evaluated weights")
    schedule = selection["pilot_schedule"]
    require(set(schedule) == {"steps", "warmup", "lr", "min_lr"}
            and type(schedule["steps"]) is int and schedule["steps"] in (250, 500, 1000)
            and type(schedule["warmup"]) is int and 0 < schedule["warmup"] <= 100
            and schedule["warmup"] < schedule["steps"]
            and 0 < schedule["min_lr"] <= schedule["lr"] <= 6e-5,
            "Pilot schedule is outside the reviewed bounded continuation range")
    return selection, pair_plan, pair, required


def prepare(selection_path):
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and not OUT.exists(), "Use CUDA-hidden CPU1 and preserve earlier pilots")
    selection, pair_plan, pair, paths = completed_review(selection_path)
    records, models, recovered_bindings = load_endpoint()
    source = records["source"]
    require(pair["models"] == models, "Vocal pair and recovered generation differ")
    bindings = dict(source["source_bindings"])
    merge_bindings(bindings, recovered_bindings)
    merge_bindings(bindings, pair_plan["source_bindings"])
    prefix_plan, prefix = qualified(PHASE / "grouped-vocal-next-prefix-001", bindings)
    _, integration = qualified(PHASE / "grouped-vocal-trainer-integration-cpu-001", bindings)
    require(integration["gpu_entry_rejected_cpu_before_cuda_initialization"]
            and integration["retained_parent_and_rng_unchanged"] and not integration["gpu_execution_qualified"]
            and prefix["independent_ordinary_references_match"] and prefix["zero_and_two_worker_replay_exact"]
            and prefix["ordinary_samples_and_rng_unchanged_by_source_views"]
            and prefix["source_views_exact_through_warmup_and_scored_suffix"]
            and prefix["first_sample_index"] == prefix_plan["first_sample_index"]
                == source["config"]["data_start"] + source["config"]["steps"] * 16
            and prefix["stop_sample_index"] == prefix["first_sample_index"] + 64,
            "Prospective grouped data or trainer qualification differs")
    inputs, idle, idle_execution, idle_root = (read(IDLE / n) for n in
        ("inputs.json", "result.json", "execution.json", "root-execution.json"))
    monitor_path = Path(idle_execution["monitor_result"])
    monitor = read(monitor_path)
    require(idle["status"] == "pass" and idle["source_bindings_unchanged"]
            and idle_execution["actual_exit_code"] == idle_root["actual_exit_code"] == 0
            and idle_execution["source_bindings_unchanged"] and idle_root["source_bindings_unchanged"]
            and idle_root["result_sha256"] == sha(IDLE / "result.json")
            and idle_root["execution_sha256"] == sha(IDLE / "execution.json")
            and inputs["previous_event_record_id"] == records["monitor"]["last_event_record_id"]
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["post_exit_quiet_completed"] and monitor["source_sha256"] == sha(MONITOR) == MONITOR_SHA
            and monitor["finalization_started"] and 60 < monitor["finalization_elapsed_seconds"] < 180,
            "Post-failure real host continuity and finalization have not completed")
    merge_bindings(bindings, inputs["source_bindings"])
    paths.extend(IDLE / n for n in ("inputs.json", "result.json", "execution.json", "root-execution.json",
                                   "command.json", "watchdog-spec.json", "child-result.json", "event-continuity.json"))
    paths.extend([monitor_path, Path(__file__).resolve(), RECOVERY / "full14-parent-review.json"])
    for directory in (VIEWS / "raw", VIEWS / "ema"):
        paths.extend(directory / (n + ".json") for n in ("plan", "result", "execution"))
    paths.extend(ROOT / "research/direct" / n for n in
        ("run_latency58_grouped_vocal_finalization.py", "train_latency58_grouped_vocal.py",
         "latency58_grouped_vocal_step.py", "check_latency58_grouped_vocal_gpu.py",
         "prepare_latency58_recovered_vocal_views.py", "run_latency58_deployed_vocal_views.py"))
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    role = selection["selected_role"]
    checkpoint = models[role]["checkpoint"]
    model, _ = load_model(checkpoint)
    require(not torch.cuda.is_initialized() and state_sha256(model.state_dict()) == models[role]["model_state_sha256"]
            and state_sha256(dict(model.named_buffers())) == source["fixed_buffers_sha256"]
            and model.architecture_metadata == source["inference_architecture"]
            and model.provenance["training_updates"] == source["parent_training_updates"] + 4000,
            "Recovered initialization, fixed buffers or inference architecture changed")
    budget = read(PHASE / "branch-gru-int8-post-ci-storage-001.json")
    budget = {**budget, "counted_roots": list(budget["counted_roots"])}
    before = budget_snapshot(budget)
    outside = (before["external_git_common_bytes"] + budget["other_outside_allowance_bytes"]
               + budget["diagnostic_artifact_allowance_bytes"])
    plan = copy.deepcopy(source)
    for key in ("retry_of", "retry_reason", "independent_observation_start", "parent_selection_review"):
        plan.pop(key, None)
    schedule = selection["pilot_schedule"]
    plan.update(schema="latency58-grouped-vocal-training-plan-v1", name=OUT.name, output_directory=str(OUT),
        config={**source["config"], **schedule, "checkpoint_every": schedule["steps"],
                "auxiliary_microbatch_size": 2, "data_start": prefix["first_sample_index"]},
        parent_checkpoint=checkpoint, parent_weight_role=role, parent_training_updates=model.provenance["training_updates"],
        parent_model_state_sha256=models[role]["model_state_sha256"],
        initialized_model_state_sha256=models[role]["model_state_sha256"],
        reference_result=models[role]["original_full_mixture_report"]["path"],
        parent_full_sdr_db=records["terminal"]["full_sdr_db"][role], parent_selection_review=binding(selection_path),
        continuation_kind="recovered_saved_parent_with_grouped_vocal_source_views",
        objective_version=VERSION, grouped_vocal_loss=applied_policy(),
        quality_endpoints=[schedule["steps"]], quality_weight_roles=["raw", "ema"],
        qualified_data_prefix=prefix["batches"], storage_budget=budget, budget_before=before,
        counted_roots=list(budget["counted_roots"]), outside_roots_reservation_bytes=outside,
        stop_counted_bytes=90_000_000_000 - outside, concurrent_training_reservation_bytes=0,
        watchdog_source=str(MONITOR), previous_execution_for_resource=str(IDLE / "execution.json"),
        supervision={"version": "planned-final-step-bounded-finalization-v1",
                     "finalization_timeout_seconds": 180, "watchdog_sha256": MONITOR_SHA},
        training_context_implementation="Independent complete two-second warmup and score for ordinary and auxiliary groups",
        resource_context_comparison="Independent ordinary and auxiliary context, whole-group gradient and raw/Adam/EMA restart parity",
        source_bindings=bindings, original_parent_training_monitor_successful=False,
        parent_recovery_result=binding(RECOVERY / "result.json"),
        all_three_acceptance_gates_required=True, automatic_plugin_replacement=False,
        gpu_resource_qualification_required=True, resource_rehearsal_updates=2)
    validate_recipe(plan)
    require(plan["config"]["seed"] == prefix_plan["augmentation_seed"]
            and plan["config"]["augmentation"] == source["config"]["augmentation"]
            and not plan["inference_architecture_changed"] and plan["optimizer_initialization"] == "fresh_adam",
            "Qualified augmentation or initialization policy differs")
    verify_inputs(plan)
    OUT.mkdir()
    write(OUT / "plan.json", plan)
    print({"status": "prepared", "plan": str(OUT / "plan.json"), "plan_sha256": sha(OUT / "plan.json"),
           "parent_role": role, "steps": schedule["steps"], "gpu_workload_started": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-review", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.selection_review.resolve())
