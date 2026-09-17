"""Qualify a prospective drum-weighted objective on an accumulated parent."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Drum functional plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-drum-emphasis-functional-plan-v2"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 functional fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous qualification")
    import torch
    from research.direct.latency58_accum_parent import load_parent
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.latency_ola512_training import raw4_native_objective
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective, VERSION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260913)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    require(sha(plan["parent_plan"]["path"]) == plan["parent_plan"]["sha256"], "Parent loader plan changed")
    parent_plan = read(plan["parent_plan"]["path"])
    verify_inputs(parent_plan)
    model = load_parent(parent_plan).train().requires_grad_(True)
    teacher, identity = load_teacher("c91", parent_plan["teacher"])
    model_state, teacher_state = state_sha256(model.state_dict()), state_sha256(teacher.state_dict())
    require(model_state == parent_plan["parent"]["model_state_sha256"]
            and teacher_state == identity["model_state_sha256"], "Different model or teacher")
    generator = torch.Generator().manual_seed(20260913)
    warmup, scored = 1024, 1024
    target = torch.randn(4, 4, 2, warmup + scored, generator=generator) * .02
    target[0, 0].zero_()
    target[1, 2].zero_()
    target[2, 3].zero_()
    mixture = target.sum(dim=1)
    flags = torch.tensor([False, True, False, True])
    teacher_target = physical_context_teacher(teacher, mixture, kind="c91", warmup_samples=warmup)
    inputs = {"mixture": mixture, "target": target, "flags": flags, "teacher_target": teacher_target}
    inputs_sha, rng = state_sha256(inputs), torch.get_rng_state().clone()

    def render():
        result = render_scored_context(model, mixture, warmup_samples=warmup, carry_state=True)
        require(result.initial_state_detached and torch.equal(result.physical_mixture, mixture[..., warmup:]),
                "Detached warmup or physical alignment differs")
        return result

    # Independent reference uses four explicit stem-specific loss calls,
    # rather than deriving weighted values from an existing full-array mean.
    output = render()
    original = raw4_native_objective(output.raw, target[..., warmup:], flags, projection=True)
    raw_losses = torch.stack([torch.nn.functional.l1_loss(output.raw[:, i], target[:, i, :, warmup:])
                              for i in range(4)])
    teacher_losses = torch.stack([torch.nn.functional.l1_loss(output.deployed[:, i], teacher_target[:, i])
                                  for i in range(4)])
    weights = raw_losses.new_tensor([2, 1, 1, 1])
    reference = (weights * raw_losses).sum() / 5 + original.projection_contribution \
                + .5 * (weights * teacher_losses).sum() / 5
    expected = torch.autograd.grad(reference, tuple(model.parameters()))
    reference_value = float(reference.detach())
    del output, reference, original, raw_losses, teacher_losses
    output = render()
    actual = drum_emphasized_objective(output.raw, output.deployed, target[..., warmup:], teacher_target, flags)
    actual_gradients = torch.autograd.grad(actual.total, tuple(model.parameters()))
    gradients = {}
    for (name, _), observed, independent in zip(model.named_parameters(), actual_gradients, expected, strict=True):
        require(bool(torch.isfinite(observed).all()) and bool((observed != 0).any())
                and torch.allclose(observed, independent, rtol=3e-5, atol=3e-7), "Weighted gradient differs: " + name)
        gradients[name] = {"max_absolute_error": float((observed - independent).abs().max())}
    require(len(gradients) == 21 and abs(float(actual.total.detach()) - reference_value) < 1e-7,
            "Incomplete weighted gradient or value agreement")
    ordinary = raw4_native_objective(output.raw, target[..., warmup:], flags, projection=True)
    unchanged = drum_emphasized_objective(output.raw, output.deployed, target[..., warmup:],
                                          teacher_target, flags, drum_weight=1)
    original_total = ordinary.total + .5 * deployed_teacher_l1(output.deployed, teacher_target)
    require(torch.equal(unchanged.total, original_total)
            and torch.equal(actual.projection, ordinary.projection)
            and torch.equal(actual.projection_contribution, ordinary.projection_contribution),
            "Identity branch or original capped projection changed")
    zero = output.raw.detach().new_zeros(output.raw.shape)
    silence = drum_emphasized_objective(zero, zero, zero, zero, flags)
    require(float(silence.total) == 0, "Silent references do not yield zero loss")
    require(model_state == state_sha256(model.state_dict()) and teacher_state == state_sha256(teacher.state_dict())
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            and all(p.grad is None for p in model.parameters()) and state_sha256(inputs) == inputs_sha
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Functional check changed model, teacher, input, gradient buffers, RNG or CPU scope")
    verify_inputs(plan)
    result = {"schema": "latency58-drum-emphasis-functional-result-v2", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "version": VERSION, "stem_weights": [2, 1, 1, 1],
              "weight_denominator": 5, "teacher_weight": .5, "reference_loss": reference_value,
              "actual_loss": float(actual.total.detach()), "gradients": gradients,
              "all_21_parameter_gradients_match": True, "weight_one_loss_bit_exact": True,
              "original_projection_and_cap_unchanged": True, "silent_loss_zero": True,
              "model_state_sha256": model_state, "teacher_model_state_sha256": teacher_state,
              "inputs_model_teacher_and_rng_unchanged": True, "gradient_buffers_unchanged": True,
              "warmup_samples": warmup, "scored_samples": scored, "batch_size": 4,
              "training_updates_executed": 0, "optimizer_instances": 0, "checkpoint_written": False,
              "cuda_initialized": False, "quality_selected": False, "elapsed_seconds": time.monotonic() - began,
              "limitations": ["Short synthetic FP32 gradients do not qualify full-crop GPU resource use.",
                              "This prepares a possible objective; no training recipe or quality benefit is established."]}
    write(out / "result.json", result)
    print({"event": "drum_weighting_functional_pass", "loss": result["actual_loss"],
           "max_gradient_error": max(value["max_absolute_error"] for value in gradients.values())}, flush=True)


if __name__ == "__main__":
    main()
