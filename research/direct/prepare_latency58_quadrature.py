"""Freeze a quadrature trial after reviewing the completed remix full14 result."""
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
    from research.direct.latency58_full_magnitude_checkpoint import load_model
    from research.direct.latency58_quadrature import Latency58QuadratureModel, QuadratureState
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and args.name and all(c.isalnum() or c in "-_" for c in args.name)
            and 250 <= args.steps <= 4000 and 0 < args.lr <= 1e-4 and args.data_start >= 2_800_000
            and args.seed > 20261012 and bool(args.rationale.strip()), "Invalid CPU preparation or recipe")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    previous_root = PHASE / "remix-magnitude-001"
    source_path = previous_root / "plan.json"
    source = read(source_path)
    verify_inputs(source)
    terminal, review = read(previous_root / "result.json"), read(previous_root / "selection-review.json")
    previous_quality = read(previous_root / "full14/result.json")
    previous_execution = read(previous_root / "full14/execution.json")
    require(terminal["status"] == "training_audit_and_full14_complete"
            and terminal["quality_result"] == binding(previous_root / "full14/result.json")
            and previous_quality["status"] == "pass" and previous_execution["actual_exit_code"] == 0
            and previous_execution["source_bindings_unchanged"] and review["actual_root_exit_code"] == 0,
            "The previous remix run must finish scoring and receive a recorded review")
    verify_inputs(review)
    parent_binding = binding(args.parent)
    parent, _ = load_model(parent_binding)
    reference_path = args.reference.resolve(strict=True)
    reference = read(reference_path)
    prior_reference = read(source["reference_result"])
    best_score = max(terminal["full_sdr_db"], prior_reference["results"][0]["aggregate"]["full_sdr_db"])
    require(reference["status"] == "pass" and reference["results"][0]["checkpoint"] == parent_binding
            and parent_binding in (terminal["checkpoint"], source["parent_checkpoint"])
            and reference["results"][0]["aggregate"]["full_sdr_db"] == best_score < 5.0,
            "Use the better verified magnitude parent while the quality target remains unmet")
    functional_paths = (PHASE / "quadrature-functional-002/result.json",
                        PHASE / "quadrature-checkpoint-functional-001/result.json")
    functional, checkpoint_check = (read(p) for p in functional_paths)
    require(functional["status"] == checkpoint_check["status"] == "pass"
            and functional["training_context"]["status"] == "pass"
            and len(functional["all_24_neural_tensors_updated"]) == 24
            and checkpoint_check["saved_audit"]["saved_optimizer_tensor_count"] == 24
            and checkpoint_check["saved_output_and_state_replay_bit_exact"]
            and checkpoint_check["temporary_checkpoint_removed"], "Quadrature CPU prerequisites failed")
    # Bind the exact sources used by both CPU checks, including their actual
    # completed wrapper executions; current hashes must still match those plans.
    prerequisite_paths = []
    for root, stage in (("quadrature-functional-002", "quadrature-functional-stage-002"),
                        ("quadrature-checkpoint-functional-001", "quadrature-checkpoint-functional-stage-001")):
        check_plan_path, execution_path = PHASE / root / "plan.json", PHASE / stage / "execution.json"
        verify_inputs(read(check_plan_path))
        execution = read(execution_path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "A CPU prerequisite lacks a completed successful execution")
        prerequisite_paths.extend((check_plan_path, execution_path))
    model = Latency58QuadratureModel.from_parent(parent)
    generator = torch.Generator().manual_seed(202609128)
    with torch.inference_mode():
        audio = .03 * torch.randn(2, 2, 19 * 128, generator=generator)
        state = parent.initial_state(2)
        first, second = parent.render(audio, state), model.render(audio, QuadratureState(*state))
        require(all(torch.equal(getattr(first, k).view(torch.int32), getattr(second, k).view(torch.int32))
                    for k in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(first.state, second.state, strict=True)),
                "Zero quadrature initialization changed the selected parent")
    out = PHASE / args.name
    require(not out.exists(), "Preserve existing trial plans")
    paths = [Path(__file__).resolve(), source_path, Path(parent_binding["path"]), reference_path,
             *functional_paths, *prerequisite_paths]
    paths.extend(previous_root / name for name in ("result.json", "selection-review.json", "full14/result.json",
                 "full14/execution.json", "production-stage/execution.json", "checkpoint-audit.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_quadrature.py", "latency58_quadrature_context.py", "latency58_quadrature_checkpoint.py",
        "check_latency58_quadrature.py", "check_latency58_quadrature_gpu.py", "check_latency58_quadrature_checkpoint.py",
        "train_latency58_quadrature.py", "run_latency58_quadrature.py", "evaluate_latency58_quadrature.py"))
    sources = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    config = {**source["config"], "steps": args.steps, "checkpoint_every": args.steps, "lr": args.lr,
              "min_lr": args.lr / 10, "data_start": args.data_start, "seed": args.seed}
    plan = {**source, "schema": "latency58-quadrature-training-plan-v1", "name": args.name,
            "output_directory": str(out), "config": config, "source_bindings": sources,
            "parent_checkpoint": parent_binding, "reference_result": str(reference_path),
            "parent_model_state_sha256": state_sha256(parent.state_dict()),
            "parent_training_updates": parent.provenance["training_updates"],
            "initialized_model_state_sha256": state_sha256(model.state_dict()),
            "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "parameter_names": [name for name, _ in model.named_parameters()],
            "inference_architecture": model.architecture_metadata, "inference_architecture_changed": True,
            "quality_endpoints": [args.steps], "all_neural_parameter_tensors_trained": 24,
            "training_context_implementation": "quadrature final-frame warmup and physically aligned scored render",
            "cpu_fast_context_bit_exact": True, "gpu_fast_context_bit_exact_required_before_updates": True,
            "quad_gpu_fixture_requires_exact_original_weight_restoration": True,
            "native_cost_measured": False, "torch_version": torch.__version__}
    counted = require_space(plan, 390_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "Preparation initialized CUDA")
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": best_score,
          "previous_remix_full_sdr_db": terminal["full_sdr_db"], "target_full_sdr_db": 5.0,
          "selected_parent_zero_initialization_bit_exact": True, "all_24_tensors_trainable_in_production": True,
          "additional_parameters": model.architecture_metadata["phase_head_parameters"],
          "additional_state_tensors": 0, "additional_audio_buffering_samples": 0,
          "rationale": args.rationale, "counted_bytes_before": counted,
          "forecast_including_outside_and_run": counted + 800_000_000 + 390_000_000,
          "cuda_initialized": False, "quality_measured_for_quadrature": False,
          "limitation": "Exploratory single-seed architecture trial; the result is not a matched causal ablation."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    return out / "plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-start", type=int, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    plan_path = prepare(parser.parse_args())
    print(json.dumps({"event": "prepared", "plan": str(plan_path), "sha256": sha(plan_path)}), flush=True)


if __name__ == "__main__":
    main()
