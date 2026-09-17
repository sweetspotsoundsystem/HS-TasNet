"""Freeze one branch-memory trial from the saved 4.353585 dB SDR-blend endpoint."""
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 4000, "lr": 3e-5, "min_lr": 3e-6, "warmup": 100,
          "data_start": 3_600_000, "seed": 20261021, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_temporal_attention_checkpoint import load_model, audit_saved
    from research.direct.latency58_branch_memory import Latency58BranchMemoryModel, VERSION as MODEL_VERSION
    from research.direct.latency58_attention_sdr_blend import VERSION, SDR_WEIGHT
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.check_latency58_branch_memory import check as check_functional
    from research.direct.check_latency58_branch_memory_checkpoint import check as check_checkpoint
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_root = PHASE / "attention-sdr-blend-001"
    source_path = parent_root / "plan.json"
    source, review = read(source_path), read(parent_root / "selection-review.json")
    terminal, quality = read(parent_root / "result.json"), read(parent_root / "full14/result.json")
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and review["best_full_sdr_db"] == terminal["full_sdr_db"] == 4.3535847721055445 < 5.0
            and review["optimizer_may_be_retired_after_this_review"] is False
            and terminal["status"] == "training_audit_and_full14_complete"
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == review["best_research_checkpoint"] == terminal["checkpoint"],
            "Require the reviewed saved SDR-blend parent")
    verify_inputs(source)
    verify_inputs(review)
    paths = [Path(__file__).resolve(), source_path]
    for name in ("production-stage/execution.json", "full14/execution.json", "root-execution.json"):
        path = parent_root / name
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "Prior execution is incomplete")
        paths.append(path)
    paths.extend(parent_root / name for name in ("selection-review.json", "result.json", "full14/result.json",
                 "checkpoint-audit.json", "production-run/checkpoint/model.pt",
                 "production-run/checkpoint/optimizer.pt", "production-run/checkpoint/receipt.json"))
    audit = audit_saved(terminal["checkpoint"], source, sha(source_path))
    parent, payload = load_model(terminal["checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 30
            and payload["step"] == 1000 and parent.provenance["training_updates"] == 26250
            and parent_sha == "aab5bd6e2d613340d9c67fc9e1eee264d8d8c7c693a28dc8026a569c9eee810c",
            "Selected parent audit or lineage changed")
    storage = PHASE / "branch-memory-storage-001"
    receipt = read(storage / "receipt.json")
    require(receipt["status"] == "complete" and receipt["preserved_bindings_unchanged"]
            and receipt["all_inference_models_and_source_audio_preserved"]
            and receipt["all_selected_optimizers_preserved"] and receipt["reserved_bytes"] == 500_000_000
            and receipt["intent_sha256"] == sha(storage / "intent.json"), "Branch-memory storage reservation incomplete")
    paths.extend(storage / name for name in ("intent.json", "receipt.json"))
    for name in ("branch-memory-functional", "branch-memory-checkpoint-functional"):
        proof_root = PHASE / (name + "-001")
        proof, proof_plan = read(proof_root / "result.json"), read(proof_root / "plan.json")
        execution_path = PHASE / (name + "-stage-001/execution.json")
        execution = read(execution_path)
        require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Branch-memory CPU proof failed")
        verify_inputs(proof_plan)
        paths.extend((proof_root / "result.json", proof_root / "plan.json", execution_path))
    files = ("latency58_branch_memory.py", "latency58_branch_memory_context.py", "latency58_branch_memory_checkpoint.py",
             "check_latency58_branch_memory.py", "check_latency58_branch_memory_checkpoint.py",
             "check_latency58_branch_memory_gpu.py", "latency58_attention_sdr_blend.py",
             "train_latency58_branch_memory.py", "run_latency58_branch_memory.py", "evaluate_latency58_branch_memory.py")
    paths.extend(ROOT / "research/direct" / name for name in files)
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-memory-001"
    require(not out.exists(), "Preserve earlier branch-memory plans")
    counted = require_space(source, 500_000_000)
    out.mkdir()
    began = time.monotonic()
    # Repeat the functional checks against the newly selected weights. The older
    # prototype proofs use the preceding parent and remain intact.
    functional = check_functional(parent)
    write(out / "current-parent-functional.json", functional)
    fixture_source = {**source, "parent_checkpoint": terminal["checkpoint"],
                      "parent_model_state_sha256": parent_sha, "parent_training_updates": 26250}
    checkpoint = check_checkpoint(fixture_source, out)
    write(out / "current-parent-checkpoint-functional.json", checkpoint)
    require(functional["status"] == checkpoint["status"] == "pass"
            and checkpoint["all_40_optimizer_states_checked"]
            and checkpoint["resumed_third_update_and_adam_moments_bit_exact"], "Current-parent CPU qualification failed")
    model = Latency58BranchMemoryModel.from_parent(parent)
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized()
            and model.algorithmic_latency_samples == 256 and len(list(model.parameters())) == 40,
            "Preparation changed the parent, state inventory or algorithmic delay")
    fingerprint = state_sha256(model.state_dict())
    paths.extend(out / name for name in ("current-parent-functional.json", "fixture-plan.json",
                                        "current-parent-checkpoint-functional.json"))
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"]}
    plan = {**source, "schema": "latency58-branch-memory-training-plan-v1", "name": out.name,
            "output_directory": str(out), "config": config,
            "source_bindings": {**bindings, **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": terminal["checkpoint"], "parent_kind": "saved_temporal_attention",
            "parent_model_state_sha256": parent_sha, "parent_training_updates": 26250,
            "optimizer_initialization": "fresh_adam", "reference_result": str(parent_root / "full14/result.json"),
            "initialized_model_state_sha256": fingerprint, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "parameter_names": [name for name, _ in model.named_parameters()], "all_neural_parameter_tensors_trained": 40,
            "inference_architecture": model.architecture_metadata, "inference_architecture_changed": True,
            "architecture_version": MODEL_VERSION, "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
            "quality_endpoints": [4000], "native_cost_measured": False, "new_weights_native_cost_measured": False,
            "training_context_implementation": "branch-memory final-frame warmup and physically aligned scored render",
            "branch_memory_gpu_fixture_requires_exact_original_weight_and_rng_restoration": True,
            "pr13_model_will_not_be_changed_by_training": True}
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": terminal["full_sdr_db"],
          "parent_audit": audit, "recipe": RECIPE, "current_parent_functional_and_ram_checkpoint_passed": True,
          "all_40_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "total_public_stream_states": 8,
          "additional_branch_memory_parameters": model.architecture_metadata["branch_memory_parameters"],
          "counted_bytes_before": counted, "reserved_training_bytes": 500_000_000,
          "forecast_including_outside_and_run": counted + 800_000_000 + 500_000_000,
          "cuda_initialized": False, "pr13_model_preserved": True, "elapsed_seconds": time.monotonic() - began,
          "rationale": "Add separate causal GRU memory to the spectral and waveform branches, starting with zero output projections that preserve the selected saved model. Retain the selected blended objective and all 301 recorded training tracks. Train all 40 learned tensors for 4000 updates before unchanged saved full14 review.",
          "regressions_carried_forward": "The SDR-blend parent improved 9 tracks but regressed 19/56 SDR cells, 30/56 SIR cells and 21/23 eligible absent-source cells versus its parent. Bass SIR fell 0.253387 dB and absent bass output rose 1.388762 dB. Skelpolu remains substantially worse than C204. Retain complete parent/C204/fixed-share comparisons at the new endpoint.",
          "limitation": "Exploratory single-seed architecture, schedule and crop-range change with fresh Adam. No matched causal claim, deployment timing or quality gain is established by functional qualification."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    require_space(plan, 495_000_000)
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json"),
                      "initialized_model_state_sha256": fingerprint}), flush=True)


if __name__ == "__main__":
    main()
