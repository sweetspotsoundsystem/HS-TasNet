"""Authenticate two actual rehearsals and the original focused-loss replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, load_source
from research.direct.latency58_counterfactual_checkpoint import (
    SHARED_CONTROL_KEYS, IDENTITY_KEYS, validate_recipe, validate_journal, require_space)
from research.direct.latency58_counterfactual_journal import compare_control_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 pairing audit")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-counterfactual-resource-pairing-plan-v1"
            and set(plan["modes"]) == {"all_views", "ordinary_only"}, "Different rehearsal scope")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve pairing evidence")
    require_space(plan, 3_000_000)
    plans, results, journals = {}, {}, {}
    for mode, spec in plan["modes"].items():
        for item in spec.values():
            require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                    "Unbound rehearsal prerequisite")
        training, resource, execution = (read(spec[k]["path"]) for k in ("training_plan", "resource", "execution"))
        validate_recipe(training)
        verify_inputs(training)
        monitor_path = Path(execution["monitor_result"])
        monitor = read(monitor_path)
        run = Path(training["run_dir"])
        status, qualification = read(run / "status.json"), read(spec["qualification"]["path"])
        for path in (monitor_path, run / "status.json", run / "metrics.jsonl"):
            require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound actual journal or clean exit")
        require(training["resource_only"] and training["teacher_mode"] == mode
                and resource["schema"] == "latency58-counterfactual-resource-result-v1"
                and resource["status"] == qualification["status"] == "pass"
                and resource["source_bindings_unchanged"] and resource["teacher_mode"] == mode
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
                "Rehearsal did not complete cleanly with these exact inputs")
        helpers = load_source("counterfactual_pairing_helpers", training["helper_source"])
        rows = validate_journal((run / "metrics.jsonl").read_bytes(), 2, training, helpers)
        require(rows == resource["matching_production_updates"], "Resource result differs from the actual journal")
        require(all(plan["source_bindings"].get(p) == s for p, s in training["source_bindings"].items()),
                "Pairing omitted training inputs")
        plans[mode], results[mode], journals[mode] = training, resource, rows
    reference, selected = plans["all_views"], plans["ordinary_only"]
    require(all(reference[k] == selected[k] for k in (*SHARED_CONTROL_KEYS, "matched_protocol",
                "matched_control_training_plan", "preparation_decision", "counterfactual_version", "functional_proofs")),
            "Rehearsals change more than teacher participation")
    control = read(reference["matched_control_training_plan"]["path"])
    old = read(control["full_resource"]["path"])
    require(plan["source_bindings"].get(control["full_resource"]["path"]) == control["full_resource"]["sha256"]
            and results["all_views"]["existing_control_prefix_exact"]
            and results["all_views"]["final_model_state_sha256"] == old["final_model_state_sha256"],
            "The original two-update model did not replay exactly")
    compare_control_prefix(journals["all_views"], old)
    identity_keys = ("first_sample_index", "next_sample_index", "micro_index", "batch_size", "view_codes",
                     "augmented_batch_sha256", "teacher_targets_sha256", *IDENTITY_KEYS)
    for a, b in zip(journals["all_views"], journals["ordinary_only"], strict=True):
        require(a["lr"] == b["lr"], "Learning rates differ")
        for x, y in zip(a["microbatches"], b["microbatches"], strict=True):
            require(all(x[k] == y[k] for k in identity_keys), "Actual examples, teacher targets or random draws differ")
            if a["step"] == 1:
                require(all(x[k] == y[k] for k in ("supervised_loss", "waveform_l1", "projection",
                        "projection_contribution", "unweighted_waveform_l1", "raw_drum_l1", "uniform_loss",
                        "uniform_teacher_l1", "uniform_unweighted_teacher_l1", "uniform_teacher_drum_l1")),
                        "The initial shared supervised loss or uniform teacher diagnostic differs")
    rng = results["all_views"]["final_rng_state_sha256"]
    require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and all(len(x) == 64 for x in rng.values()) and rng == results["ordinary_only"]["final_rng_state_sha256"],
            "Terminal RNG streams differ")
    require(results["all_views"]["final_model_state_sha256"] != results["ordinary_only"]["final_model_state_sha256"]
            and journals["all_views"][0]["first_update_inherited_gradient_sha256"]
                != journals["ordinary_only"][0]["first_update_inherited_gradient_sha256"]
            and all(m["teacher_l1"] < m["uniform_teacher_l1"] for m in journals["ordinary_only"][0]["microbatches"]),
            "Removing controlled-view distillation did not change the actual optimization")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-counterfactual-resource-pairing-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "resource_results": {mode: spec["resource"] for mode, spec in plan["modes"].items()},
          "training_plans": {mode: spec["training_plan"] for mode, spec in plan["modes"].items()},
          "updates_per_mode": 2, "microbatches_per_mode": 8, "examples_per_mode": 32,
          "existing_control_replay_exact": True, "all_inputs_and_teacher_targets_exact": True,
          "initial_supervised_and_uniform_teacher_terms_exact": True, "final_rng_states_exact": True,
          "selected_loss_changes_updates": True, "final_rng_state_sha256": rng,
          "final_model_states": {mode: r["final_model_state_sha256"] for mode, r in results.items()},
          "training_updates_executed": 0, "quality_selected": False,
          "limitations": ["Only the first two updates were rehearsed; full training matching and quality remain required."]})
    print({"status": "pass", "existing_control_replay_exact": True, "selected_loss_changes_updates": True}, flush=True)


if __name__ == "__main__":
    main()
