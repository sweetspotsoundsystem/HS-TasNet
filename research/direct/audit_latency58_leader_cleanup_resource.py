"""Compare the actual leader rehearsal with the retained working-parent recipe."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs
from research.direct.latency58_leader_cleanup_checkpoint import RECIPE_KEYS, validate_recipe, validate_journal, require_space
from research.direct.latency58_controlled_deployed_checkpoint import IDENTITY_KEYS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the frozen CPU comparison")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-resource-comparison-plan-v1", "Different comparison")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve comparison")
    require_space(plan, 400_000_000)
    for item in plan["prerequisites"].values():
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound resource prerequisite")
    spec = plan["prerequisites"]
    training = read(spec["training_plan"]["path"])
    reference = validate_recipe(training)
    verify_inputs(training)
    verify_inputs(reference)
    resource, execution, qualification = (read(spec[k]["path"]) for k in ("resource", "execution", "qualification"))
    monitor = read(execution["monitor_result"])
    run = Path(training["run_dir"])
    status = read(run / "status.json")
    for path in (Path(execution["monitor_result"]), run / "status.json", run / "metrics.jsonl"):
        require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound monitor or actual journal")
    require(training["resource_only"] and resource["schema"] == "latency58-leader-cleanup-resource-result-v1"
            and resource["status"] == qualification["status"] == "pass" and resource["source_bindings_unchanged"]
            and resource["training_updates_executed"] == 2 and resource["augmented_examples_executed"] == 32
            and resource["all_parameter_gradients_present"] and resource["fixed_buffers_unchanged"]
            and resource["teacher_unchanged"] and not resource["checkpoint_written"]
            and resource["initial_model_state_sha256"] == training["initialized_model_state_sha256"]
            and resource["plan_sha256"] == execution["plan_sha256"] == spec["training_plan"]["sha256"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"]
            and status["status"] == "resource_complete" and status["step"] == 2
            and status["model_state_sha256"] == resource["final_model_state_sha256"]
            and qualification["resource"] == spec["resource"] and qualification["execution"] == spec["execution"],
            "Actual leader rehearsal did not close successfully")
    for key in ("model_check", "model_check_execution"):
        item = qualification[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound new parent CPU proof")
    checked = read(qualification["model_check"]["path"])
    require(checked["status"] == "pass" and checked["parent_model_state_sha256"] == training["initialized_model_state_sha256"],
            "Different CPU parent proof")
    helpers = load_source("leader_cleanup_resource_helpers", training["helper_source"])
    rows = validate_journal((run / "metrics.jsonl").read_bytes(), 2, training, helpers)
    require(rows == resource["matching_production_updates"], "Resource result differs from actual journal")
    old_binding = reference["full_resource"]
    old = read(old_binding["path"])
    old_execution = read(reference["full_resource_execution"]["path"])
    old_monitor = read(old_execution["monitor_result"])
    old_plan = read(reference["resource_plan"]["path"])
    from research.direct.latency58_controlled_deployed_checkpoint import validate_journal as validate_old
    old_rows = validate_old((Path(old_plan["run_dir"]) / "metrics.jsonl").read_bytes(), 2, old_plan, helpers)
    require(old["schema"] == "latency58-controlled-deployed-resource-result-v1" and old["status"] == "pass"
            and old["source_bindings_unchanged"] and old["training_updates_executed"] == 2
            and old_execution["actual_exit_code"] == 0 and old_execution["source_bindings_unchanged"]
            and old_monitor["status"] == old_monitor["supervisor_health"] == "pass"
            and old_monitor["child_exit_code"] == 0 and old_monitor["post_exit_quiet_completed"]
            and old_execution["plan_sha256"] == old["plan_sha256"] == reference["resource_plan"]["sha256"]
            and old["matching_production_updates"] == old_rows
            and all(old_plan[k] == training[k] for k in RECIPE_KEYS)
            and old["initial_model_state_sha256"] != resource["initial_model_state_sha256"],
            "Different or incomplete reference rehearsal")
    identity_keys = ("first_sample_index", "next_sample_index", "micro_index", "batch_size", "view_codes",
                     "augmented_batch_sha256", "teacher_targets_sha256", *IDENTITY_KEYS)
    for old_row, new_row in zip(old_rows, rows, strict=True):
        require(old_row["lr"] == new_row["lr"], "Learning rates differ")
        for a, b in zip(old_row["microbatches"], new_row["microbatches"], strict=True):
            require(all(a[k] == b[k] for k in identity_keys), "Data, teacher targets or random draws differ")
    require(resource["final_rng_state_sha256"] == old["final_rng_state_sha256"]
            and set(resource["final_rng_state_sha256"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and resource["final_model_state_sha256"] != old["final_model_state_sha256"],
            "Different RNG streams or no parent effect on the resulting model")
    for document in (training, reference, old_plan):
        require(all(plan["source_bindings"].get(p) == s for p, s in document["source_bindings"].items()),
                "Comparison omitted source bindings")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-leader-cleanup-resource-comparison-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "training_plan": spec["training_plan"], "resource": spec["resource"], "execution": spec["execution"],
          "reference_training_plan": training["reference_training_plan"], "reference_resource": old_binding,
          "updates_per_parent": 2, "microbatches_per_parent": 8, "examples_per_parent": 32,
          "operational_recipe_exact": True, "all_inputs_and_teacher_targets_exact": True,
          "final_rng_states_exact": True, "comparison_variable": "training_parent",
          "initial_model_states": {"leader": resource["initial_model_state_sha256"], "working": old["initial_model_state_sha256"]},
          "final_model_states": {"leader": resource["final_model_state_sha256"], "working": old["final_model_state_sha256"]},
          "peak_vram_gib": resource["peak_vram_gib"], "training_updates_executed": 0, "quality_selected": False,
          "matched_loss_effect_from_leader_claimed": False,
          "limitations": ["Two rehearsal updates only. The 250-update run must reproduce this prefix exactly and complete the unchanged quality protocol.",
                          "This comparison tests parent transfer under one fixed recipe; it does not isolate the added loss against an ordinary-only leader control."]})
    print({"status": "pass", "parent_transfer_inputs_exact": True, "final_rng_states_exact": True}, flush=True)


if __name__ == "__main__":
    main()
