"""Prepare fused-feature refinement after the completed complex-mask trial is reviewed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def prepare(args):
    import torch
    from research.direct.latency58_quadrature_checkpoint import load_model, audit_saved
    from research.direct.latency58_fusion_refinement import Latency58FusionRefinementModel, FusionRefinementState
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and args.name and all(c.isalnum() or c in "-_" for c in args.name)
            and 250 <= args.steps <= 4000 and 0 < args.lr <= 1e-4
            and 0 < args.warmup < args.steps and args.data_start >= 3_016_000
            and args.seed > 20261015 and bool(args.rationale.strip()), "Invalid CPU preparation or recipe")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    previous_root = PHASE / "complex-mask-001"
    source_path = previous_root / "plan.json"
    source = read(source_path)
    verify_inputs(source)
    terminal = read(previous_root / "result.json")
    review = read(previous_root / "selection-review.json")
    previous_quality = read(previous_root / "full14/result.json")
    require(terminal["status"] == "training_audit_and_full14_complete"
            and terminal["quality_result"] == binding(previous_root / "full14/result.json")
            and previous_quality["status"] == "pass" and previous_quality["track_count"] == 14
            and previous_quality["excerpt_count"] == 28 and review["actual_root_exit_code"] == 0
            and review["status"] == "not_selected",
            "The complex-mask trial must finish scoring and receive its rejection review")
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(previous_root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "The previous run has not closed successfully")
    verify_inputs(review)
    parent_binding = binding(args.parent)
    parent, payload = load_model(parent_binding)
    reference_path = args.reference.resolve(strict=True)
    reference = read(reference_path)
    prior_reference = read(source["reference_result"])
    best_score = max(terminal["full_sdr_db"], prior_reference["results"][0]["aggregate"]["full_sdr_db"])
    require(reference["status"] == "pass" and reference["track_count"] == 14
            and reference["excerpt_count"] == 28 and reference["results"][0]["checkpoint"] == parent_binding
            and parent_binding == review["best_research_checkpoint"]
            and parent_binding == source["parent_checkpoint"]
            and reference["results"][0]["aggregate"]["full_sdr_db"] == best_score < 5.0,
            "Use the best saved quadrature parent while the quality target remains unmet")
    parent_root = Path(parent_binding["path"]).parents[2]
    parent_plan_path = parent_root / "plan.json"
    parent_plan = read(parent_plan_path)
    verify_inputs(parent_plan)
    parent_audit = audit_saved(parent_binding, parent_plan, sha(parent_plan_path))
    require(parent_audit["status"] == "pass" and parent_audit["saved_optimizer_tensor_count"] == 24
            and payload["plan_sha256"] == sha(parent_plan_path), "Selected parent generation audit failed")
    functional_paths = (PHASE / "fusion-refinement-functional-001/result.json",
                        PHASE / "fusion-refinement-checkpoint-memory-functional-001/result.json")
    functional, checkpoint_check = (read(p) for p in functional_paths)
    require(functional["status"] == checkpoint_check["status"] == "pass"
            and functional["training_context"]["status"] == "pass"
            and len(functional["all_26_neural_tensors_updated"]) == 26
            and checkpoint_check["all_26_optimizer_states_checked"]
            and checkpoint_check["inference_outputs_and_states_bit_exact"]
            and checkpoint_check["resumed_third_update_and_adam_moments_bit_exact"]
            and checkpoint_check["cpu_rng_formats_replay_exact"]
            and not checkpoint_check["disk_generation_path_exercised"], "Fusion-refinement CPU prerequisites failed")
    prerequisite_paths = []
    prerequisite_bindings = {}
    for root, stage in (("fusion-refinement-functional-001", "fusion-refinement-functional-stage-001"),
                        ("fusion-refinement-checkpoint-memory-functional-001", "fusion-refinement-checkpoint-memory-functional-stage-001")):
        check_plan_path, execution_path = PHASE / root / "plan.json", PHASE / stage / "execution.json"
        check_plan = read(check_plan_path)
        verify_inputs(check_plan)
        prerequisite_bindings.update(check_plan["source_bindings"])
        execution = read(execution_path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "A CPU prerequisite lacks a completed successful execution")
        prerequisite_paths.extend((check_plan_path, execution_path))
    require(read(PHASE / "fusion-refinement-functional-001/plan.json")["parent_checkpoint"] == parent_binding
            and functional["parent_model_state_sha256"] == payload["model_state_sha256"],
            "Use the exact current parent exercised by the CPU proofs")
    model = Latency58FusionRefinementModel.from_parent(parent)
    with torch.inference_mode():
        audio = .03 * torch.randn(2, 2, 19 * 128, generator=torch.Generator().manual_seed(202609144))
        state = parent.initial_state(2)
        first, second = parent.render(audio, state), model.render(audio, FusionRefinementState(*state))
        require(all(torch.equal(getattr(first, key).view(torch.int32), getattr(second, key).view(torch.int32))
                    for key in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(first.state, second.state, strict=True)),
                "Zero refinement initialization changed the selected parent")
    out = PHASE / args.name
    require(not out.exists(), "Preserve existing trial plans")
    paths = [Path(__file__).resolve(), source_path, parent_plan_path, Path(parent_binding["path"]), reference_path,
             *functional_paths, *prerequisite_paths]
    paths.extend(previous_root / name for name in ("result.json", "selection-review.json", "full14/result.json",
                 "full14/execution.json", "production-stage/execution.json", "checkpoint-audit.json"))
    paths.extend(Path(parent_binding["path"]).parent / name for name in ("optimizer.pt", "receipt.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_fusion_refinement.py", "latency58_fusion_refinement_context.py", "latency58_fusion_refinement_checkpoint.py",
        "check_latency58_fusion_refinement.py", "check_latency58_fusion_refinement_gpu.py",
        "check_latency58_fusion_refinement_checkpoint_memory.py", "train_latency58_fusion_refinement.py",
        "run_latency58_fusion_refinement.py", "evaluate_latency58_fusion_refinement.py"))
    config = {**source["config"], "steps": args.steps, "checkpoint_every": args.steps, "lr": args.lr,
              "min_lr": args.lr / 10, "warmup": args.warmup, "data_start": args.data_start, "seed": args.seed}
    plan = {**source, "schema": "latency58-fusion-refinement-training-plan-v1", "name": args.name,
            "output_directory": str(out), "config": config,
            "source_bindings": {**source["source_bindings"], **prerequisite_bindings,
                                **{str(p): sha(p) for p in paths}},
            "parent_checkpoint": parent_binding, "reference_result": str(reference_path),
            "parent_kind": "saved_quadrature", "optimizer_initialization": "fresh_adam",
            "parent_model_state_sha256": state_sha256(parent.state_dict()),
            "parent_training_updates": parent.provenance["training_updates"],
            "initialized_model_state_sha256": state_sha256(model.state_dict()),
            "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "parameter_names": [name for name, _ in model.named_parameters()],
            "inference_architecture": model.architecture_metadata, "inference_architecture_changed": True,
            "quality_endpoints": [args.steps], "all_neural_parameter_tensors_trained": 26,
            "training_context_implementation": "fusion-refinement final-frame warmup and physically aligned scored render",
            "cpu_fast_context_bit_exact": True, "gpu_fast_context_bit_exact_required_before_updates": True,
            "fusion_refinement_gpu_fixture_requires_exact_original_weight_restoration": True,
            "disk_generation_audit_required_before_scoring": True,
            "native_cost_measured": False, "torch_version": torch.__version__}
    plan.pop("quad_gpu_fixture_requires_exact_original_weight_restoration", None)
    plan.pop("complex_mask_gpu_fixture_requires_exact_original_weight_restoration", None)
    counted = require_space(plan, 390_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "Preparation initialized CUDA")
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": best_score,
          "previous_complex_mask_full_sdr_db": terminal["full_sdr_db"], "target_full_sdr_db": 5.0,
          "selected_parent_zero_initialization_bit_exact": True, "parent_saved_generation_audit": parent_audit,
          "all_26_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "additional_parameters": model.architecture_metadata["fusion_refinement_parameters"],
          "additional_state_tensors": 0, "additional_audio_buffering_samples": 0,
          "rationale": args.rationale, "counted_bytes_before": counted,
          "forecast_including_outside_and_run": counted + 800_000_000 + 390_000_000,
          "cuda_initialized": False, "quality_measured_for_fusion_refinement": False,
          "disk_serialization_prerequisite": "Payload and exact restart verified in RAM; actual generation is audited after training before quality scoring.",
          "limitation": "Exploratory single-seed architecture trial; the result is not a matched causal ablation."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    return out / "plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-start", type=int, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    plan_path = prepare(parser.parse_args())
    print(json.dumps({"event": "prepared", "plan": str(plan_path), "sha256": sha(plan_path)}), flush=True)


if __name__ == "__main__":
    main()
