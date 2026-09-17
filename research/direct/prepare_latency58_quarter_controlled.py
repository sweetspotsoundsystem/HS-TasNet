"""Freeze a matched view-frequency trial after its synthetic CPU proof passes."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, state_sha256, verify_inputs


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation required")
    from research.direct.latency58_quarter_controlled_augmentation import VERSION, VIEW_CYCLE
    from research.direct.latency58_quarter_controlled_checkpoint import validate_recipe, load_parent, validate_matched_inputs
    from research.direct.prove_latency58_quarter_controlled import normalized

    prep = PHASE / "quarter-controlled-prep-001"
    require(not prep.exists(), "Preserve an existing preparation")
    control_path = PHASE / "cleanup-successor-prep-002/training-plan.json"
    control = read(control_path)
    verify_inputs(control)
    functional_path = PHASE / "quarter-controlled-functional-001/result.json"
    functional_plan_path = functional_path.parent / "plan.json"
    functional = read(functional_path)
    verify_inputs(read(functional_plan_path))
    require(functional["status"] == "pass" and functional["source_bindings_unchanged"]
            and functional["plan_sha256"] == sha(functional_plan_path), "Functional proof failed or changed")

    observations, evidence = {}, {}
    for prefix, stage_name in (("cleanup-successor", "cleanup-successor-to-000250-002"),
                               ("cleanup-followup", "cleanup-followup-to-000250-001")):
        stage = PHASE / stage_name
        execution, audit_execution, audit = [read(stage / name) for name in ("execution.json", "audit-execution.json", "audit.json")]
        monitor = read(execution["monitor_result"])
        summary = PHASE / (prefix + "-250-summary-001/result.json")
        result = read(summary)
        require(execution["actual_exit_code"] == audit_execution["actual_exit_code"] == monitor["child_exit_code"] == 0
                and execution["source_bindings_unchanged"] and audit_execution["source_bindings_unchanged"]
                and not audit_execution["timed_out"] and audit["status"] == result["status"] == "pass"
                and audit["model_state_sha256"] == result["model_state_sha256"]
                and result["source_bindings_unchanged"] and audit["step"] == 250
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["post_exit_quiet_completed"], "Prior experiment did not close healthily")
        for path in (summary, stage / "execution.json", stage / "audit-execution.json", stage / "audit.json", Path(execution["monitor_result"])):
            evidence[str(path)] = sha(path)
        observations[prefix] = {"model_state_sha256": result["model_state_sha256"],
                               **{key: result["full_mixture_aggregate"][key]
                                  for key in ("full_sdr_db", "low_sdr_db", "bleed_sir_db")}}

    source_dir = ROOT / "research/direct"
    replacements = (("cleanup_rebound", "quarter_controlled"), ("cleanup-rebound", "quarter-controlled"),
                    ("matched_learning_rate_schedule", "matched_controlled_view_frequency"),
                    ("latency58_vocal_focus_augmentation", "latency58_quarter_controlled_augmentation"),
                    ("latency58_controlled_deployed_journal", "latency58_quarter_controlled_journal"),
                    ("latency58_counterfactual_journal", "latency58_variable_counterfactual_journal"))
    require(normalized(source_dir / "train_latency58_cleanup_rebound.py", replacements)
            == normalized(source_dir / "train_latency58_quarter_controlled.py"),
            "Trainer changed beyond family, augmentation and journal routing")
    def functions(path, transform=False):
        text = path.read_text()
        if transform:
            for old, new in replacements:
                text = text.replace(old, new)
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef)}
    old = functions(source_dir / "latency58_cleanup_rebound_checkpoint.py", True)
    new = functions(source_dir / "latency58_quarter_controlled_checkpoint.py")
    unchanged = ("audit_live", "load_parent", "read_generation", "load_model", "save_generation")
    require(all(old[key] == new[key] for key in unchanged), "Model or optimizer checkpoint arithmetic changed")
    old_runner = functions(source_dir / "run_latency58_cleanup_rebound.py", True)
    new_runner = functions(source_dir / "run_latency58_quarter_controlled.py")
    require(old_runner["stage"] == new_runner["stage"], "Watchdog or stage resource limits changed")

    # Synthetic journal fixtures use the real control's address/RNG metadata.
    # Changed target hashes below are fixtures, never claims about executed audio.
    control_journal = Path(control["run_dir"]) / "checkpoints/step-000250/metrics.jsonl"
    reference = [json.loads(line) for line in control_journal.read_bytes().splitlines()][:2]
    fixture = copy.deepcopy(reference)
    for row in fixture:
        for micro in row["microbatches"]:
            micro["view_codes"] = [VIEW_CYCLE[(micro["first_sample_index"] + i) % 8] for i in range(4)]
            if all(code >= 2 for code in micro["view_codes"]):
                micro["augmented_batch_sha256"] = micro["original_augmentation_sha256"]
                micro["teacher_targets_sha256"] = hashlib.sha256(b"synthetic fixture, not an inference result").hexdigest()
    matched_fixture = validate_matched_inputs(fixture, reference)
    rejected = []
    for name, micro_index, key, value in (
        ("pristine crop", 1, "pristine_batch_sha256", "0" * 64),
        ("original augmentation", 1, "original_augmentation_sha256", "0" * 64),
        ("RNG before", 1, "augmentation_rng_before_sha256", "0" * 64),
        ("RNG after", 1, "augmentation_rng_after_sha256", "0" * 64),
        ("absolute address", 1, "first_sample_index", 0),
        ("view cycle", 1, "view_codes", [0, 1, 2, 3]),
        ("shared teacher targets", 0, "teacher_targets_sha256", "0" * 64),
        ("shared augmented inputs", 0, "augmented_batch_sha256", "0" * 64),
        ("restored ordinary inputs", 1, "augmented_batch_sha256", "0" * 64)):
        bad = copy.deepcopy(fixture)
        bad[0]["microbatches"][micro_index][key] = value
        try:
            validate_matched_inputs(bad, reference)
        except (RuntimeError, ValueError, AssertionError):
            rejected.append(name)
        else:
            raise AssertionError("Matched input audit accepted corruption: " + name)

    outside = 146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects")
    counted = sum(disk_bytes(Path(path)) for path in control["counted_roots"])
    require(outside < 500_000_000 and counted + 1_250_000_000 < 79_500_000_000,
            "Insufficient storage for both pending LR trials and the prospective view-frequency trial")
    prep.mkdir()
    decision = {"schema": "latency58-quarter-controlled-decision-v1",
                "status": "train_quarter_controlled_independent_of_deployment", "config": control["config"],
                "training_parent": control["parent"], "maximum_production_updates": 250,
                "resource_updates": 2, "quality_endpoints": [250], "qualification_blocks_training": False,
                "release_work_blocks_training": False, "optimizer_initialization": "fresh_adam",
                "matched_control_training_plan": binding(control_path), "observations": observations,
                "hypothesis": "Completed continuation endpoints improved isolated-vocal cleanup while full-mixture SDR and interference did not improve. Reduce forced instrumental/vocal-only views from one half to one quarter of examples, restoring more original-distribution mixtures. Hold the accepted parent, 4000 absolute crop addresses, original augmentation RNG, context, learning-rate schedule and all loss coefficients fixed against the completed successor.",
                "comparison_limits": ["Teacher participation and deployed-truth supervision frequency change with the view policy; this is not a teacher-coefficient-only or auxiliary-coefficient-only comparison.",
                                      "Original distribution includes subset and vocal-derangement augmentation.",
                                      "One seed and the repeatedly used selection panel do not establish generalization."],
                "evaluation": "Unchanged full14, per-stem/bands/interference/absence, Actions60, probes, vocal-only and instrumental views. Compare accepted parent, working baseline and matched half-controlled successor. Reserved additional confirmation remains untouched until one candidate is frozen after primary quality review.",
                "gpu_queue_policy": "After the lower-rate trial and its independent checkpoint audit exit successfully; CPU quality and plugin qualification do not block training.",
                "storage": {"counted_bytes": counted, "outside_counted_roots_bytes": outside,
                            "reserve_for_pending_lr_and_new_trial_bytes": 1_250_000_000,
                            "new_trial_checkpoint_and_quality_reserve_bytes": 400_000_000,
                            "counted_stop_bytes": 79_500_000_000, "combined_cap_bytes": 80_000_000_000}}
    write(prep / "decision.json", decision)
    write(prep / "arithmetic-proof.json", {"status": "pass", "numeric_loss_functions_unchanged": True,
          "trainer_changes": "family, augmentation import, journal import and comparison metadata only",
          "checkpoint_functions_ast_identical_after_family_rename": list(unchanged),
          "watchdog_stage_ast_identical_after_family_rename": True, "synthetic_matched_fixture": matched_fixture,
          "corrupt_input_fixtures_rejected": rejected, "functional_proof": binding(functional_path),
          "gpu_training_match_proven": False})
    plan = copy.deepcopy(control)
    for key in ("resource_plan", "full_resource", "full_resource_execution"):
        plan.pop(key, None)
    plan.update(schema="latency58-quarter-controlled-training-v1", resource_only=True,
                run_dir=str(PHASE / "quarter-controlled-resource-run-001"),
                comparison_variable="matched_controlled_view_frequency", augmentation_version=VERSION,
                view_cycle=list(VIEW_CYCLE), controlled_examples_per_update=4, ordinary_examples_per_update=12,
                stop_counted_bytes=79_500_000_000, preparation_decision=binding(prep / "decision.json"),
                functional_proof=binding(functional_path), matched_control_training_plan=binding(control_path))
    plan["source_bindings"].update(evidence)
    paths = [control_path, functional_path, functional_plan_path, prep / "decision.json", prep / "arithmetic-proof.json",
             source_dir / "latency58_variable_counterfactual_journal.py", source_dir / "prove_latency58_quarter_controlled.py",
             *(source_dir.glob('*quarter_controlled*.py')),
             *((Path(control["run_dir"]) / 'checkpoints/step-000250').glob('*'))]
    plan["source_bindings"].update({str(path): sha(path) for path in paths if path.is_file()})
    verify_inputs(plan)
    validate_recipe(plan)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(plan)
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"]
            and not torch.cuda.is_initialized(), "Accepted parent identity or CPU preparation differs")
    write(prep / "resource-plan.json", plan)
    print(json.dumps({"status": "prepared_not_launched", "plan": binding(prep / "resource-plan.json"),
                      "storage": decision["storage"], "synthetic_matched_fixture": matched_fixture}), flush=True)


if __name__ == "__main__":
    main()
