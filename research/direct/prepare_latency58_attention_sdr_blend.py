"""Freeze a fixed SDR-auxiliary trial from the selected 4.330201 dB checkpoint."""
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 1000, "lr": 1e-5, "min_lr": 1e-6, "warmup": 50,
          "data_start": 3_500_000, "seed": 20261020, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_temporal_attention_checkpoint import load_model, audit_saved
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_attention_sdr_blend import VERSION, SDR_WEIGHT
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_root, preceding = PHASE / "attention-continuation-001", PHASE / "attention-grouped-001"
    source_path = parent_root / "plan.json"
    source, review = read(source_path), read(parent_root / "selection-review.json")
    prior_review = read(preceding / "selection-review.json")
    for record in (source, review, prior_review):
        verify_inputs(record)
    terminal, quality = read(parent_root / "result.json"), read(parent_root / "full14/result.json")
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and prior_review["status"] == "not_selected" and prior_review["actual_root_exit_code"] == 0
            and terminal["status"] == "training_audit_and_full14_complete"
            and terminal["full_sdr_db"] == prior_review["best_full_sdr_db"] == 4.330200584072513 < 5.0
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == review["best_research_checkpoint"]
            == prior_review["best_research_checkpoint"] == terminal["checkpoint"],
            "Require the saved selected attention endpoint and completed grouped review")
    paths = [Path(__file__).resolve(), source_path]
    for directory in (parent_root, preceding):
        for name in ("production-stage/execution.json", "full14/execution.json", "root-execution.json"):
            path = directory / name
            execution = read(path)
            require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                    and not execution.get("timed_out", False), "Prior execution is incomplete")
            paths.append(path)
        paths.extend(directory / name for name in (
            "selection-review.json", "result.json", "full14/result.json", "checkpoint-audit.json"))
    audit = audit_saved(terminal["checkpoint"], source, sha(source_path))
    model, payload = load_model(terminal["checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 30
            and payload["step"] == 4000 and model.provenance["training_updates"] == 25250,
            "Saved parent audit or lineage failed")
    functional = PHASE / "attention-sdr-blend-functional-001"
    proof, proof_plan, execution = (read(functional / name) for name in ("result.json", "plan.json", "execution.json"))
    require(proof["status"] == "pass" and proof["objective_version"] == VERSION
            and proof["direct_sdr_weight"] == SDR_WEIGHT == .1
            and proof["combined_gradient_contains_direct_sdr_contribution"]
            and proof["numpy_primary_metric_check"]["status"] == "pass"
            and not proof["validation_audio_used"] and not proof["coefficient_search_performed"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Blended objective functional proof failed")
    verify_inputs(proof_plan)
    paths.extend(functional / name for name in ("plan.json", "result.json", "execution.json"))
    retirement_root = PHASE / "attention-sdr-blend-storage-001"
    retirement = read(retirement_root / "receipt.json")
    require(retirement["status"] == "complete" and retirement["preserved_bindings_unchanged"]
            and retirement["all_inference_models_and_source_audio_preserved"]
            and retirement["all_selected_optimizers_preserved"]
            and retirement["reserved_bytes"] == 400_000_000, "Training storage reservation incomplete")
    paths.extend(retirement_root / name for name in ("intent.json", "receipt.json"))
    paths.extend(parent_root / "production-run/checkpoint" / name for name in ("model.pt", "optimizer.pt", "receipt.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_attention_sdr_blend.py", "train_latency58_attention_sdr_blend.py", "run_latency58_attention_sdr_blend.py"))
    history = []
    for name in ("direct-sdr-001", "direct-sdr-musdb-001"):
        result, training = read(PHASE / name / "result.json"), read(PHASE / name / "plan.json")
        comparison = read(PHASE / name / "full14/result.json")["comparison"]["metrics"]["full_sdr_db"]
        history.append({"name": name, "full_sdr_db": result["full_sdr_db"], "delta_db": comparison["delta"],
                        "objective_version": training["objective_version"], "steps": training["config"]["steps"]})
        paths.extend(PHASE / name / filename for filename in ("plan.json", "result.json", "full14/result.json"))
    out = PHASE / "attention-sdr-blend-001"
    require(not out.exists(), "Preserve prior plans")
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"]}
    plan = {**source, "name": out.name, "output_directory": str(out), "config": config,
            "source_bindings": {**source["source_bindings"], **proof_plan["source_bindings"],
                                **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": terminal["checkpoint"], "parent_kind": "saved_temporal_attention",
            "optimizer_initialization": "fresh_adam", "reference_result": str(parent_root / "full14/result.json"),
            "parent_model_state_sha256": fingerprint, "parent_training_updates": 25250,
            "initialized_model_state_sha256": fingerprint, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
            "quality_endpoints": [1000], "inference_architecture_changed": False,
            "continuation_uses_trained_nonzero_attention_head": True,
            "new_weights_native_cost_measured": False, "pr13_model_will_not_be_changed_by_training": True}
    counted = require_space(plan, 400_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized() and state_sha256(model.state_dict()) == fingerprint, "Preparation changed the parent")
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": terminal["full_sdr_db"],
          "parent_audit": audit, "objective_functional_proof_passed": True,
          "all_30_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "recipe": RECIPE,
          "direct_sdr_weight": SDR_WEIGHT, "counted_bytes_before": counted, "reserved_training_bytes": 400_000_000,
          "forecast_including_outside_and_run": counted + 800_000_000 + 400_000_000,
          "cuda_initialized": False, "pr13_model_preserved": True, "earlier_direct_loss_trials": history,
          "rationale": "Earlier direct-loss replacements regressed from the fixed-share parent. Keep the selected model's full reconstruction objective and add a fixed 0.1-weighted direct-SDR auxiliary from the stronger trained attention checkpoint. Select only after saved-checkpoint audit and complete unchanged full14 review.",
          "limitation": "New loss, fresh Adam, schedule and sample range together; not a matched causal ablation or exact optimizer restart. The fixed coefficient is a prospective hypothesis, not evidence of benefit. All parent optimizers remain retained."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json")}), flush=True)


if __name__ == "__main__":
    main()
