"""Freeze a 4,000-update continuation from the selected 21,250-update attention model."""
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 4000, "lr": 6e-5, "min_lr": 6e-6, "warmup": 100,
          "data_start": 3_300_000, "seed": 20261018, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_temporal_attention_checkpoint import load_model, audit_saved
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_root = PHASE / "temporal-attention-001"
    source_path = parent_root / "plan.json"
    source, review = read(source_path), read(parent_root / "selection-review.json")
    verify_inputs(source)
    verify_inputs(review)
    terminal, quality = read(parent_root / "result.json"), read(parent_root / "full14/result.json")
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and terminal["status"] == "training_audit_and_full14_complete"
            and terminal["full_sdr_db"] == 4.288099064999147 < 5.0
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == review["best_research_checkpoint"] == terminal["checkpoint"],
            "Require the saved selected attention endpoint")
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(parent_root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"], "Parent is incomplete")
    audit = audit_saved(terminal["checkpoint"], source, sha(source_path))
    model, payload = load_model(terminal["checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 30
            and payload["step"] == 1000 and model.provenance["training_updates"] == 21250,
            "Saved parent audit or lineage failed")
    memory_root = PHASE / "attention-continuation-memory-functional-001"
    memory, memory_plan = read(memory_root / "result.json"), read(memory_root / "plan.json")
    memory_execution_path = PHASE / "attention-continuation-memory-functional-stage-001/execution.json"
    execution = read(memory_execution_path)
    require(memory["status"] == "pass" and memory["source_bindings_unchanged"]
            and memory["trained_parent_context"]["status"] == "pass"
            and len(memory["trained_parent_context"]["all_30_gradients"]) == 30
            and memory["resumed_third_update_and_adam_moments_bit_exact"]
            and memory["inference_outputs_and_states_bit_exact"]
            and memory_plan["parent_checkpoint"] == terminal["checkpoint"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Trained-parent CPU proof must close")
    verify_inputs(memory_plan)
    pr_path = PHASE / "attention-model-pr-created.json"
    ci_path = PHASE / "attention-plugin-ci-001/result.json"
    pr, ci = read(pr_path), read(ci_path)
    require(pr["pr_url"] == "https://github.com/sweetspotsoundsystem/stemgen-rt/pull/13"
            and pr["plugin_commit"] == "c84805083c1127f6749f140ce8a49ae8b5acbf54"
            and ci["status"] == "pass" and ci["head_sha"] == pr["plugin_commit"]
            and ci["run_id"] == 34712530853 and ci["windows_passed"] and ci["macos_passed"],
            "Finish the requested exact-model plugin handoff before the next training run")
    retirement_root = PHASE / "attention-plugin-build-object-retirement-001"
    retirement = read(retirement_root / "receipt.json")
    require(retirement["status"] == "complete" and retirement["preserved_bindings_unchanged"]
            and retirement["all_inference_models_and_source_audio_preserved"]
            and retirement["reserved_bytes"] == 420_000_000, "Training storage reservation incomplete")
    out = PHASE / "attention-continuation-001"
    require(not out.exists(), "Preserve prior plans")
    paths = [Path(__file__).resolve(), source_path, memory_root / "plan.json", memory_root / "result.json",
             memory_execution_path, pr_path, ci_path, retirement_root / "intent.json", retirement_root / "receipt.json"]
    paths.extend(parent_root / name for name in (
        "selection-review.json", "result.json", "full14/result.json", "full14/execution.json",
        "production-stage/execution.json", "checkpoint-audit.json", "production-run/checkpoint/receipt.json",
        "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "train_latency58_temporal_attention_continuation.py", "run_latency58_temporal_attention_continuation.py",
        "check_latency58_attention_continuation_memory.py"))
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"]}
    plan = {**source, "name": out.name, "output_directory": str(out), "config": config,
            "source_bindings": {**source["source_bindings"], **memory_plan["source_bindings"],
                                **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": terminal["checkpoint"], "parent_kind": "saved_temporal_attention",
            "optimizer_initialization": "fresh_adam", "reference_result": str(parent_root / "full14/result.json"),
            "parent_model_state_sha256": fingerprint, "parent_training_updates": 21250,
            "initialized_model_state_sha256": fingerprint, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "quality_endpoints": [4000], "inference_architecture_changed": False,
            "continuation_uses_trained_nonzero_attention_head": True,
            "new_weights_native_cost_measured": False, "pr13_model_will_not_be_changed_by_training": True}
    counted = require_space(plan, 420_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized() and state_sha256(model.state_dict()) == fingerprint, "Preparation changed the parent")
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": terminal["full_sdr_db"],
          "parent_audit": audit, "trained_parent_context_and_checkpoint_memory_proof_passed": True,
          "all_30_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "recipe": RECIPE,
          "counted_bytes_before": counted, "reserved_training_bytes": 420_000_000,
          "forecast_including_outside_and_run": counted + 800_000_000 + 420_000_000,
          "cuda_initialized": False, "pr13_model_preserved": True,
          "rationale": "The 1,000-update attention pilot was selected on the unchanged development panel. Extend training with the same architecture, objective, corpus and streaming cost; select only after saved-checkpoint audit and complete full14 review.",
          "limitation": "Fresh Adam and a new schedule/data range, not an exact continuation of the pilot optimizer. The original selected optimizer remains retained. Single-seed development selection does not establish unseen-track generalization."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json")}), flush=True)


if __name__ == "__main__":
    main()
