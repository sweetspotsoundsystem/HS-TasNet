"""Check accumulated drum-weighted gradients against an independent joint mean."""
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
    require(sha(args.plan) == args.plan_sha256, "Functional plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-drum-accum-functional-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a CUDA-hidden CPU1 functional fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve functional outputs")
    import torch
    from research.direct.latency58_accum_parent import load_parent
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency_ola512_training import raw4_native_objective
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective, VERSION as OBJECTIVE_VERSION
    from research.direct.latency58_sdr_accum import backward_mean_loss, VERSION, MICROBATCH_SIZE, ACCUMULATION_STEPS

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260914)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    binding = plan["parent_plan"]
    require(sha(binding["path"]) == binding["sha256"]
            and plan["source_bindings"].get(binding["path"]) == binding["sha256"], "Parent loader plan changed")
    parent_plan = read(binding["path"])
    verify_inputs(parent_plan)
    require(all(plan["source_bindings"].get(p) == s for p, s in parent_plan["source_bindings"].items()),
            "Functional plan does not bind every parent input")
    model = load_parent(parent_plan).train().requires_grad_(True)
    teacher, teacher_identity = load_teacher("c91", parent_plan["teacher"])
    model_state, teacher_state = state_sha256(model.state_dict()), state_sha256(teacher.state_dict())
    require(model_state == parent_plan["parent"]["model_state_sha256"]
            and teacher_state == teacher_identity["model_state_sha256"], "Different fixture parent or teacher")
    generator = torch.Generator().manual_seed(20260914)
    warmup, scored = 1024, 1024
    batches = []
    for index in range(ACCUMULATION_STEPS):
        target = torch.randn(MICROBATCH_SIZE, 4, 2, warmup + scored, generator=generator) * (.01 + index * .005)
        # Exercise absent drums and every other stem, as well as both
        # projection-mask states, in each independent B4 microbatch.
        for example in range(MICROBATCH_SIZE):
            target[example, (index + example) % 4].zero_()
        mixture = target.sum(dim=1)
        flags = torch.tensor([False, True, False, True])
        teacher_target = physical_context_teacher(teacher, mixture, kind="c91", warmup_samples=warmup)
        batches.append((mixture, target, flags, teacher_target))
    rng = torch.get_rng_state().clone()
    input_hashes = [state_sha256({str(i): value for i, value in enumerate(batch)}) for batch in batches]

    def objective(batch, *, independent):
        mixture, target, flags, teacher_target = batch
        output = render_scored_context(model, mixture, warmup_samples=warmup, carry_state=True)
        require(output.initial_state_detached and torch.equal(output.physical_mixture, mixture[..., warmup:]),
                "Functional fixture lost detached warmup or physical alignment")
        if independent:
            # Four explicit stem losses provide a reference independent of
            # the normalized drum helper, including the original B4 cap.
            original = raw4_native_objective(output.raw, target[..., warmup:], flags, projection=True)
            raw_losses = torch.stack([torch.nn.functional.l1_loss(output.raw[:, i], target[:, i, :, warmup:])
                                      for i in range(4)])
            teacher_losses = torch.stack([torch.nn.functional.l1_loss(output.deployed[:, i], teacher_target[:, i])
                                          for i in range(4)])
            weights = raw_losses.new_tensor([2, 1, 1, 1])
            return (weights * raw_losses).sum() / 5 + original.projection_contribution \
                   + .5 * (weights * teacher_losses).sum() / 5
        return drum_emphasized_objective(output.raw, output.deployed, target[..., warmup:],
                                        teacher_target, flags).total

    # Retain all four short graphs and differentiate the explicit mean once.
    losses = [objective(batch, independent=True) for batch in batches]
    reference_values = [float(loss.detach()) for loss in losses]
    reference_loss = torch.stack(losses).mean()
    reference_gradients = torch.autograd.grad(reference_loss, tuple(model.parameters()))
    reference_value = float(reference_loss.detach())
    del losses, reference_loss
    model.zero_grad(set_to_none=True)
    sequential_values = []
    for batch in batches:
        loss = objective(batch, independent=False)
        sequential_values.append(float(loss.detach()))
        backward_mean_loss(loss)
        del loss
    require(len(reference_gradients) == 21 and len(sequential_values) == ACCUMULATION_STEPS
            and all(abs(a - b) < 1e-7 for a, b in zip(sequential_values, reference_values, strict=True)),
            "Accumulation changed the independent forward objectives")
    gradients = {}
    for (name, parameter), expected in zip(model.named_parameters(), reference_gradients, strict=True):
        actual = parameter.grad
        require(actual is not None and bool(torch.isfinite(actual).all()) and bool((actual != 0).any())
                and torch.allclose(actual, expected, rtol=3e-5, atol=3e-7), "Accumulated gradient differs: " + name)
        gradients[name] = {"max_absolute_error": float((actual - expected).abs().max()),
                           "reference_max_absolute": float(expected.abs().max())}
    require(model_state == state_sha256(model.state_dict()) and teacher_state == state_sha256(teacher.state_dict())
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and input_hashes == [state_sha256({str(i): value for i, value in enumerate(batch)}) for batch in batches],
            "Functional check changed tensors, teacher targets, inputs, RNG or device scope")
    model.zero_grad(set_to_none=True)
    require(all(p.grad is None for p in model.parameters()), "Gradient buffers were not cleared")
    verify_inputs(plan)
    result = {
        "schema": "latency58-drum-accum-functional-result-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "accumulation_version": VERSION,
        "objective_version": OBJECTIVE_VERSION, "stem_weights": [2, 1, 1, 1],
        "weight_denominator": 5, "teacher_weight": .5,
        "microbatch_size": MICROBATCH_SIZE, "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": MICROBATCH_SIZE * ACCUMULATION_STEPS,
        "all_21_parameter_gradients_match": True, "gradients": gradients,
        "reference_mean_loss": reference_value, "reference_microbatch_losses": reference_values,
        "actual_microbatch_losses": sequential_values,
        "model_state_sha256": model_state, "teacher_model_state_sha256": teacher_state,
        "warmup_samples": warmup, "scored_samples": scored, "cuda_initialized": False,
        "optimizer_instances": 0, "training_updates_executed": 0, "checkpoint_written": False,
        "inputs_teacher_and_model_unchanged": True, "render_rng_unchanged": True,
        "gradient_buffers_cleared": True, "quality_selected": False,
        "limitations": [
            "Short synthetic CPU FP32 gradients do not qualify full-crop GPU resource use.",
            "The mean retains each B4 projection cap; no single-forward B16 penalty equivalence is claimed.",
            "This prepares a possible objective; no training recipe or quality benefit is established.",
        ],
        "elapsed_seconds": time.monotonic() - began,
    }
    write(out / "result.json", result)
    print({"status": "pass", "effective_batch_size": result["effective_batch_size"],
           "mean_loss": reference_value,
           "max_gradient_error": max(v["max_absolute_error"] for v in gradients.values())}, flush=True)


if __name__ == "__main__":
    main()
