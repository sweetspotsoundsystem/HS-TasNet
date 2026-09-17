"""Freeze the controlled weighted-vocal trial after its CPU qualifications."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.run_latency58_paired_vocal_serial import binding, load_endpoint, merge_bindings
from research.direct.train_latency58_weighted_vocal import validate_recipe, applied_policy, runtime_policy, VERSION
from research.direct.latency58_weighted_vocal_canonical import policy as accumulation_policy
from research.direct.latency58_lossless_recovery_codec_v2 import policy as packed_policy
from research.direct.latency58_weighted_storage import policy as storage_policy, snapshot
from research.direct.run_latency58_weighted_vocal import require_cpu_evidence, require_monitor_closed
from research.direct.recover_latency58_nvml_guard_idle import require_monitor_qualification

SOURCE = PHASE / "branch-grouped-vocal-013"
PARENT = PHASE / "branch-grouped-vocal-012"
OUT = PHASE / "branch-weighted-vocal-014"
DECISION = PHASE / "grouped-gradient-probe-training-review-001/next-experiment-decision.json"
STORAGE = PHASE / "weighted-vocal-quarter-storage-001"


def same_training_prefix():
    paths = [ROOT / "research/direct" / name for name in
             ("train_latency58_grouped_continuation.py", "train_latency58_weighted_vocal.py")]
    prefixes = []
    for path in paths:
        tree = ast.parse(path.read_text())
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and ast.unparse(node.target) == "(mixture_cpu, truth_cpu)"]
        require(len(loops) == 1, "Ambiguous training loop")
        body = loops[0].body
        boundary = [index for index, node in enumerate(body) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "step % 25 == 0"]
        require(len(boundary) == 1, "Missing scientific loop boundary")
        prefixes.append([ast.dump(node, include_attributes=False) for node in body[:boundary[0]]])
    require(prefixes[0] == prefixes[1], "Data, learning rate or complete update loop differs")
    return {"status": "pass", "reference": binding(paths[0]), "weighted": binding(paths[1]),
            "compared_top_level_statements": len(prefixes[0]),
            "scope": "Complete data, learning-rate, update, prefix comparison and journal loop before storage audits",
            "objective_change": "Qualified weighted grouped_update and applied_policy imports"}


def prepare():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 preparation")
    require(OUT.is_dir() and {path.name for path in OUT.iterdir()} <=
            {"cpu-integration", "cpu-integration-stage", "preparation-stage"}, "Preserve existing plans and runs")
    decision = read(DECISION)
    require(sha(DECISION) == "f258eb12e90c5212dae95ab7cb96e3eeee750415a01491bdf291c8b5d631d33e"
            and decision["status"] == "selected_for_cpu_qualification" and decision["source_bindings_unchanged"]
            and decision["all_40_neural_parameter_tensors_to_remain_trainable"]
            and not decision["quality_selected"] and not decision["overall_goal_complete"], "Wrong scientific decision")
    records, reference_models, bindings = load_endpoint(SOURCE)
    parent_records, parent_models, parent_bindings = load_endpoint(PARENT)
    source = records["source"]
    merge_bindings(bindings, parent_bindings)
    merge_bindings(bindings, source["source_bindings"])
    merge_bindings(bindings, decision["source_bindings"])
    selected = parent_models["ema"]
    require(selected == decision["preserved_models"]["starting_parent"]
            and reference_models == {role: decision["preserved_models"][role] for role in ("raw", "ema")}
            and source["config"] == decision["proposed_training_config"]
            and source["parent_checkpoint"] == selected["checkpoint"] == decision["parent_checkpoint"]
            and source["parent_model_state_sha256"] == selected["model_state_sha256"] == decision["parent_model_state_sha256"]
            and source["ema"] == decision["ema"]
            and source["parameter_names"] == decision["trainable_parameter_names"], "Controlled comparison identities differ")
    qualifications = ((PHASE / "weighted-vocal-quarter-cpu-002", "weighted_cpu_qualification"),
                      (PHASE / "lossless-recovery-cpu-003", "packed_cpu_qualification"),
                      (OUT / "cpu-integration", "packed_integration_cpu_qualification"))
    paths = [Path(__file__).resolve(), DECISION]
    for directory, _ in qualifications:
        merge_bindings(bindings, read(directory / "plan.json")["source_bindings"])
        paths.extend(directory / name for name in ("plan.json", "result.json", "execution.json"))
    paths.extend(OUT / "cpu-integration" / name for name in ("fixture-plan.json", "synthetic-disk-receipt.json"))
    paths.extend(OUT / "cpu-integration-stage" / name for name in ("command.json", "execution.json"))
    allocation, allocation_execution = (read(STORAGE / name) for name in
                                        ("allocation-qualification.json", "allocation-qualification-execution.json"))
    retirement, retirement_execution = (read(STORAGE / name) for name in
                                        ("retirement-receipt.json", "retirement-execution.json"))
    require(allocation["status"] == "pass" and allocation["source_bindings_unchanged"]
            and allocation["policy"] == storage_policy() and allocation["all_standing_reservation_capacity_retained"]
            and allocation_execution["actual_exit_code"] == 0 and not allocation_execution["timed_out"]
            and allocation_execution["source_bindings_unchanged"]
            and allocation_execution["result_sha256"] == sha(STORAGE / "allocation-qualification.json"),
            "Storage allocation qualification is incomplete")
    require(retirement["status"] == "complete" and retirement["all_selected_files_absent"]
            and retirement["all_preserved_files_unchanged"] and retirement["source_bindings_unchanged"]
            and retirement["all_models_optimizers_plugin_binaries_sources_tests_and_logs_preserved"]
            and retirement_execution["actual_exit_code"] == 0 and not retirement_execution["timed_out"]
            and retirement_execution["source_bindings_unchanged"]
            and retirement_execution["receipt_sha256"] == sha(STORAGE / "retirement-receipt.json"),
            "Generated-cache retirement is incomplete")
    paths.extend(STORAGE / name for name in ("inventory.json", "historical-references.json", "retirement-intent.json",
        "retirement-receipt.json", "retirement-execution.json", "allocation-qualification.json", "allocation-qualification-execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in ("train_latency58_weighted_vocal.py",
        "run_latency58_weighted_vocal.py", "latency58_weighted_vocal_auxiliary.py", "latency58_weighted_vocal_canonical.py",
        "check_latency58_weighted_vocal_device.py", "latency58_weighted_storage.py", "check_latency58_weighted_storage.py",
        "latency58_lossless_recovery_codec_v2.py", "latency58_lossless_recovery_files_v3.py",
        "latency58_weighted_recovery_check.py", "check_latency58_weighted_recovery_integration.py",
        "run_latency58_grouped_vocal_runtime.py", "recover_latency58_nvml_guard_idle.py"))
    merge_bindings(bindings, {str(path): sha(path) for path in paths})
    require_monitor_qualification()
    require_monitor_closed(records["production_execution"], records["monitor"], final_step=1000)
    prefix_proof = same_training_prefix()
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model, _ = load_model(selected["checkpoint"])
    require(type(model) is Latency58BranchMemoryModel and not torch.cuda.is_initialized()
            and state_sha256(model.state_dict()) == selected["model_state_sha256"]
            and state_sha256(dict(model.named_buffers())) == source["fixed_buffers_sha256"]
            and model.architecture_metadata == source["inference_architecture"]
            and list(dict(model.named_parameters())) == source["parameter_names"]
            and model.provenance["training_updates"] == source["parent_training_updates"] == 43750
            and parent_records["terminal"]["full_sdr_db"]["ema"] == decision["parent_full_sdr_db"],
            "Selected saved parent, buffers, parameters or architecture changed")
    plan = copy.deepcopy(source)
    for key in ("continuation_review", "continuation_storage_forecast", "schedule_recovery_cpu_qualification",
                "canonical_cpu_control_scope", "canonical_cpu_control_model_state_sha256",
                "stop_counted_bytes", "outside_roots_reservation_bytes"):
        plan.pop(key, None)
    plan.update(name=OUT.name, output_directory=str(OUT), source_bindings=bindings,
        objective_version=VERSION, grouped_vocal_loss=applied_policy(), accumulation_policy=accumulation_policy(),
        packed_recovery=packed_policy(), weighted_storage=storage_policy(), runtime_allowance=runtime_policy(),
        scientific_decision=binding(DECISION), controlled_reference_training_plan=binding(SOURCE / "plan.json"),
        continuation_kind="fresh_adam_same_parent_and_addresses_with_quarter_vocal_auxiliary_weight",
        previous_execution_for_resource=str(SOURCE / "production-stage/execution.json"),
        event_continuity_scope="Continue the completed 013 production monitor into the weighted 014 resource stage.",
        qualified_data_prefix_scope="Unchanged same-address 013 input, augmentation and whole-group reference counts.",
        historical_cpu_qualification_scope="Inherited checks cover shared components; the three new CPU results qualify the changed objective and packed recovery.",
        weighted_training_prefix_ast_proof=prefix_proof,
        storage_allocation_qualification=binding(STORAGE / "allocation-qualification.json"),
        generated_cache_retirement=binding(STORAGE / "retirement-receipt.json"),
        quality_selected=False, overall_goal_complete=False,
        post_training_requirements=["Audit both packed raw and EMA roles from disk and authenticate all optimizer state.",
            "Run unchanged full14 and paired source-view protocols for both roles.",
            "Review each stem and worst windows against 012 EMA, both 013 endpoints and retained 006 EMA.",
            "Qualify export and physical M4 playback before any release; preserve all rollback baselines."])
    for directory, key in qualifications:
        plan[key] = binding(directory / "result.json")
    validate_recipe(plan)
    require_cpu_evidence(plan)
    verify_inputs(plan)
    before = snapshot(plan)
    plan["budget_before"] = before
    plan["weighted_storage_forecast"] = {"projected_peak_bytes": before["projected_peak_bytes"],
        "all_existing_checkpoints_and_optimizers_preserved": True,
        "persistent_packed_generation_bytes": 380_000_000, "pending_save_reserve_bytes": 600_000_000,
        "monitor_training_and_evaluation_diagnostic_quota_bytes": 280_000_000,
        "other_outside_allowance_bytes": 800_000_000,
        "packed_final_reuses_rolling_file": True, "all_current_external_git_bytes_included": True}
    write(OUT / "plan.json", plan)
    write(OUT / "preparation-result.json", {"status": "prepared", "plan_sha256": sha(OUT / "plan.json"),
        "source_bindings_unchanged": True, "source_binding_count": len(bindings),
        "parent_model_state_sha256": selected["model_state_sha256"], "training_prefix_ast_proof": prefix_proof,
        "all_required_cpu_evidence_pass": True, "budget_after": snapshot(plan),
        "gpu_workload_started": False, "quality_measured": False})
    print(json.dumps({"status": "prepared", "plan_sha256": sha(OUT / "plan.json"), "updates": plan["config"]["steps"],
        "source_binding_count": len(bindings), "projected_peak_bytes": before["projected_peak_bytes"],
        "headroom_after_complete_peak_bytes": before["headroom_after_complete_peak_bytes"], "gpu_workload_started": False}), flush=True)


if __name__ == "__main__":
    prepare()
