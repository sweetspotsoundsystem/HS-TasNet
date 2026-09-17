"""Prepare a fixed parent-transfer experiment using the closed cleanup review and retained SDR leader."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_leader_cleanup_checkpoint_v2 import validate_recipe, load_parent, require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use frozen CPU1 preparation")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-preparation-plan-v1", "Different preparation")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "decision.json").exists(), "Preserve prospective decision")
    for key in ("reference_training_plan", "closed_review", "closed_review_execution", "retirement", "retirement_execution"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound preparation input")
    reference = read(plan["reference_training_plan"]["path"])
    verify_inputs(reference)
    review, review_execution = read(plan["closed_review"]["path"]), read(plan["closed_review_execution"]["path"])
    require(review["schema"] == "latency58-controlled-deployed-review-v1" and review["status"] == "pass"
            and review["source_bindings_unchanged"] and review["training_closed"] and review["completed_step"] == 250
            and review["training_plan"] == plan["reference_training_plan"] and review["further_optimizer_updates"] == 0
            and review["all_quality_metrics_reviewed"] and not review["quality_selected"] and not review["goal_complete"]
            and review_execution["actual_exit_code"] == 0 and not review_execution["timed_out"]
            and review_execution["source_bindings_unchanged"] and review_execution["plan_sha256"] == review["plan_sha256"],
            "Working-parent cleanup review is incomplete")
    verify_inputs(review)
    retirement, retired_execution = read(plan["retirement"]["path"]), read(plan["retirement_execution"]["path"])
    require(retirement["schema"] == "latency58-sdr-diagnostic-export-retirement-v1" and retirement["status"] == "complete"
            and retirement["source_bindings_unchanged"] and retirement["native_failure_evidence_retained"]
            and retirement["reserved_bytes"] == 400_000_000 and not retirement["accepted_model_deleted"]
            and not retirement["trained_checkpoint_deleted"] and not retirement["source_audio_deleted"]
            and retired_execution["actual_exit_code"] == 0 and not retired_execution["timed_out"]
            and retired_execution["source_bindings_unchanged"] and retired_execution["plan_sha256"] == retirement["plan_sha256"],
            "Space retirement is incomplete")
    before = require_space(reference, 400_000_000)
    from research.direct.report_latency58_sdr import load_completed
    from research.direct.evaluate_latency58_sdr_drum_accum import load_evaluation_model
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bindings = {**reference["source_bindings"], **review["source_bindings"], **plan["source_bindings"],
                str(args.plan.resolve()): args.plan_sha256}
    qualities, results = {}, {}
    for mode in ("full14", "actions60", "probes"):
        qualities[mode], results[mode] = load_completed(PHASE / ("sdr-drum-accum-500-" + mode + "-001"), bindings)
        bindings.update(qualities[mode]["source_bindings"])
    quality = qualities["full14"]
    require(quality["step"] == 500 and all(q["generation"] == quality["generation"] for q in qualities.values()),
            "Mixed leader endpoints")
    rng = torch.get_rng_state().clone()
    model, receipt = load_evaluation_model(quality)
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == results["full14"]["results"][0]["model"]["model_state_sha256"]
            == results["actions60"]["results"][0]["model"]["model_state_sha256"] == results["probes"]["model_state_sha256"],
            "Leader quality does not belong to the retained weights")
    for name in ("model.pt", "receipt.json", "rng.pt", "metrics.jsonl"):
        path = Path(quality["generation"]) / name
        bindings[str(path)] = sha(path)
    parent = {"kind": "drum_accum", "quality_plan": binding(PHASE / "sdr-drum-accum-500-full14-001/plan.json"),
              "checkpoint": quality["checkpoint"], "model_state_sha256": fingerprint,
              "provenance": model.provenance, "architecture": model.architecture_metadata}
    full = results["full14"]["results"][0]["aggregate"]["full_sdr_db"]
    require(full == 4.025094965347405 and model.architecture_metadata == reference["architecture"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "Different parent or geometry")
    del model
    decision = {"schema": "latency58-leader-cleanup-decision-v1", "status": "prepare_fixed_parent_transfer",
        "source_bindings": bindings, "reference_training_plan": plan["reference_training_plan"],
        "closed_working_parent_review": plan["closed_review"], "training_parent": parent, "config": reference["config"],
        "additional_loss_weight": .5, "additional_loss_version": reference["additional_loss_version"],
        "resource_updates": 2, "maximum_production_updates": 250, "quality_endpoints": [250],
        "automatic_continuation": False, "optimizer_initialization": "fresh_adam",
        "comparison_variable": "training_parent", "matched_loss_effect_from_leader_claimed": False,
        "parent_full_sdr_db": full, "minimum_full_sdr_db": 4.057715948706591, "stretch_full_sdr_db": 4.258989,
        "checkpoint_and_quality_reserve_bytes": 400_000_000, "space_retirement": plan["retirement"],
        "hypothesis": "The fixed controlled deployed-truth cleanup recipe may reduce vocal spill from the retained SDR leader while retaining more of its instrumental separation than the same recipe initialized from the working model.",
        "comparison_scope": "Only the trained parent and its initialized model-state identity differ from the completed 250-update cleanup recipe. Keep data counter 972000, seed, augmentation, teacher participation and targets, coefficients, context, Adam initialization and schedule fixed. Compare the new endpoint with the working model, its SDR-leader parent and the completed working-parent cleanup endpoint. This tests parent transfer, not a matched loss effect from the leader.",
        "prior_costs": "The working-parent cleanup review retained quiet-wanted and local instrumental regressions and a severe vocal-isolation failure on Skelpolu. Lower spill alone does not satisfy the goal. The leader has weaker vocal-only Other suppression; its higher aggregate SDR does not establish adoption quality.",
        "evaluation": "Use unchanged full14 two-excerpt scoring with per-stem full/low-band SDR, interference, absence and probes. Compare both controlled vocal directions, wanted vocal gain and wanted instrumental quality, including raw quiet-window regressions. Retain all local and aggregate costs. Review the four Actions captures and gather human listening separately. No normalization or added validation exclusions.",
        "confirmation_and_deployment": "Only a selected improvement that clears the original quality criteria proceeds to reserved confirmation material, continuous streaming, ONNX/native parity, latency and M4 runtime checks. Preserve source audio, accepted models, the working M4 rollback and prior paced-timing failures. The unchanged graph architecture and 128-sample graph delay plus 128-sample queue are metadata expectations until the candidate is qualified.",
        "storage_forecast": {"total_reserved_bytes": 400_000_000, "checkpoint_and_training_records": 350_000_000,
            "four_actions_stem_wavs": 21_200_000, "quality_reports_and_monitors": 28_800_000,
            "previous_comparable_trial_growth_approx_bytes": 392_400_000,
            "additional_onnx_or_listening_captures_require_separate_reservation": True},
        "quality_selected": False, "goal_complete": False, "training_updates_executed": 0,
        **{k: reference[k] for k in ("counted_roots", "stop_counted_bytes")}}
    write(out / "decision.json", decision)
    candidate = copy.deepcopy(reference)
    for key in ("matched_control_training_plan", "matched_ordinary_training_plan", "reference_resource", "matched_protocol",
                "resource_plan", "full_resource", "full_resource_execution", "resource_pairing", "resource_pairing_execution"):
        candidate.pop(key, None)
    candidate.update(schema="latency58-leader-cleanup-training-v1", resource_only=True,
        parent=parent, initialized_model_state_sha256=fingerprint,
        comparison_variable="training_parent", checkpoint_and_quality_reserve_bytes=400_000_000,
        reference_training_plan=plan["reference_training_plan"], preparation_decision=binding(out / "decision.json"),
        run_dir=str(PHASE / "leader-cleanup-resource-run-001"),
        source_bindings={**bindings, str(out / "decision.json"): sha(out / "decision.json")})
    validate_recipe(candidate)
    verify_inputs(candidate)
    model = load_parent(candidate)
    require(state_sha256(model.state_dict()) == fingerprint and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized(), "Prepared parent replay differs")
    write(out / "resource-plan.json", candidate)
    model_out = PHASE / "leader-cleanup-model-check-001"
    require(not model_out.exists(), "Preserve CPU proof")
    model_out.mkdir()
    checked_plan = {"schema": "latency58-leader-cleanup-model-check-plan-v1", "output_directory": str(model_out),
        "recipe_plan": binding(out / "resource-plan.json"),
        "source_bindings": {**candidate["source_bindings"], str(out / "resource-plan.json"): sha(out / "resource-plan.json")},
        **{k: reference[k] for k in ("counted_roots", "stop_counted_bytes")}}
    write(model_out / "plan.json", checked_plan)
    verify_inputs(plan)
    after = require_space(candidate, 400_000_000)
    write(out / "result.json", {"schema": "latency58-leader-cleanup-preparation-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": bindings, "source_bindings_unchanged": True,
        "resource_plan": binding(out / "resource-plan.json"), "model_check_plan": binding(model_out / "plan.json"),
        "parent_model_state_sha256": fingerprint, "parent_full_sdr_db": full, "parent_replay_exact": True,
        "training_updates_executed": 0, "cuda_initialized": False, "cpu_rng_unchanged": True,
        "counted_bytes_before": before, "counted_bytes_after": after, "quality_selected": False})
    print({"status": "pass", "resource_plan": binding(out / "resource-plan.json"),
           "model_check_plan": binding(model_out / "plan.json"), "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
