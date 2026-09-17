"""Verify matched vocal-focus inputs and the first inherited GPU gradients."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs

ARMS = ("original", "focused", "focused_mixer")


def compare_journals(plans, rows):
    from research.direct.latency58_vocal_focus_checkpoint import IDENTITY_KEYS, LOSS_KEYS
    shared = ("config", "parent", "warmup_samples", "scored_samples", "carry_state",
              "teacher_kind", "teacher_weight", "teacher_model_state_sha256", "teacher",
              "precision_policy", "torch_version", "environment", "helper_source", "watchdog_source",
              "manifest_sha256", "functional_proofs", "microbatch_size", "accumulation_steps",
              "accumulation_version", "drum_weight", "objective_version", "augmentation_version",
              "mixer_initialization_seed", "automatic_continuation", "preparation_decision")
    base = plans["original"]
    require(set(plans) == set(rows) == set(ARMS)
            and all(all(plans[arm][key] == base[key] for key in shared) for arm in ARMS)
            and plans["original"]["initialized_model_state_sha256"]
            == plans["focused"]["initialized_model_state_sha256"] == base["parent"]["model_state_sha256"]
            and plans["original"]["architecture"] == plans["focused"]["architecture"]
            and len({len(value) for value in rows.values()}) == 1, "Pilot recipes or endpoints are unmatched")
    micro_fields = (*IDENTITY_KEYS, "micro_index", "first_sample_index", "next_sample_index", "batch_size")
    focused_fields = ("augmented_batch_sha256", "teacher_targets_sha256", "view_codes", "deranged_examples")
    compared, changed = 0, 0
    for row_index, triples in enumerate(zip(*(rows[arm] for arm in ARMS), strict=True)):
        original, focused, mixer = triples
        for key in ("step", "lr", "first_sample_index", "next_sample_index", *IDENTITY_KEYS):
            require(original[key] == focused[key] == mixer[key], "Unmatched update address, LR or original RNG")
        for micros in zip(*(row["microbatches"] for row in triples), strict=True):
            original_micro, focused_micro, mixer_micro = micros
            require(all(original_micro[key] == focused_micro[key] == mixer_micro[key] for key in micro_fields),
                    "Pristine crop, original augmentation or RNG differs across arms")
            require(all(focused_micro[key] == mixer_micro[key] for key in focused_fields),
                    "The focused arms received different final inputs or teacher targets")
            if row_index == 0:
                require(all(focused_micro[key] == mixer_micro[key] for key in LOSS_KEYS),
                        "Zero-initialized mixer changed a first-update BF16 microbatch loss")
            changed += original_micro["augmented_batch_sha256"] != focused_micro["augmented_batch_sha256"]
            compared += 1
    require(changed > 0 and rows["focused"][0]["first_update_inherited_gradient_sha256"]
            == rows["focused_mixer"][0]["first_update_inherited_gradient_sha256"],
            "Focused data was unchanged or zero initialization changed inherited GPU gradients")
    return {"updates_per_arm": len(rows["original"]), "microbatches_per_arm": compared,
            "examples_per_arm": 4 * compared, "changed_microbatches_original_to_focused": changed,
            "all_three_pristine_and_original_draws_exact": True,
            "focused_inputs_and_teacher_targets_exact": True,
            "initial_inherited_gradients_exact": True,
            "first_update_four_microbatch_losses_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Resource pairing plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-resource-pairing-plan-v1"
            and set(plan["arms"]) == set(ARMS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Require CUDA-hidden CPU1")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve pairing result")
    from research.direct.latency58_vocal_focus_checkpoint import validate_journal, require_space
    began, before = time.monotonic(), require_space(plan, 2_000_000)
    plans, rows, fingerprints, resources = {}, {}, {}, {}
    for arm in ARMS:
        spec = plan["arms"][arm]
        for binding in spec.values():
            require(sha(binding["path"]) == binding["sha256"], "Resource pairing input changed")
        training, result, execution = (read(spec[key]["path"]) for key in ("plan", "result", "execution"))
        monitor = read(execution["monitor_result"])
        verify_inputs(training)
        require(training["arm"] == result["arm"] == arm and training["resource_only"]
                and result["schema"] == "latency58-vocal-focus-resource-result-v1"
                and result["status"] == "pass" and result["training_updates_executed"] == 2
                and result["augmented_examples_executed"] == 32 and not result["checkpoint_written"]
                and result["source_bindings_unchanged"] and result["teacher_unchanged"]
                and result["all_parameter_gradients_present"] and result["fixed_buffers_unchanged"]
                and result["initial_model_state_sha256"] == training["initialized_model_state_sha256"]
                and result["teacher_model_state_sha256"] == training["teacher_model_state_sha256"]
                and result["config"] == training["config"]
                and result["plan_sha256"] == execution["plan_sha256"] == spec["plan"]["sha256"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                "Resource stage is incomplete, changed or not qualified")
        helpers = load_source("vocal_focus_pairing_helpers", training["helper_source"])
        raw = (Path(training["run_dir"]) / "metrics.jsonl").read_bytes()
        rows[arm] = validate_journal(raw, 2, training, helpers)
        require(rows[arm] == result["matching_production_updates"], "Resource result and journal differ")
        plans[arm], fingerprints[arm], resources[arm] = training, result["final_model_state_sha256"], spec["result"]
    comparisons = compare_journals(plans, rows)
    import torch
    require(not torch.cuda.is_initialized(), "Pairing audit initialized CUDA")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-vocal-focus-resource-pairing-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "resource_results": resources,
        "final_resource_model_state_sha256": fingerprints, **comparisons,
        "elapsed_seconds": time.monotonic() - began, "counted_bytes_before": before,
        "counted_bytes_after": require_space(plan, 0), "training_updates_executed": 0,
        "optimizer_instances": 0, "cuda_initialized": False, "quality_selected": False,
        "limitations": ["Two actual GPU updates per arm qualify matching and resource use, not separation quality.",
                        "The mixer has additional gradients; later model weights and optimizer trajectories intentionally differ."]})
    print({"status": "pass", **comparisons}, flush=True)


if __name__ == "__main__":
    main()
