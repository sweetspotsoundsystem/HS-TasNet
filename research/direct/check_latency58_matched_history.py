"""Qualify exact-parent streaming geometry and both matched-history gradients."""
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
    require(plan["schema"] == "latency58-matched-history-functional-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a CUDA-hidden CPU1 fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve functional outputs")
    import torch
    import torch.nn.functional as F
    from research.direct.latency58_drum_accum_parent import load_parent
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_sdr_context import physical_context_teacher
    from research.direct.latency58_matched_history import render_history_view, VERSION as HISTORY_VERSION
    from research.direct.latency_ola512_training import raw4_native_objective
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective, VERSION as OBJECTIVE_VERSION
    from research.direct.latency58_sdr_accum import backward_mean_loss, VERSION

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    binding = plan["parent_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Parent loader plan differs")
    parent_plan = read(binding["path"])
    verify_inputs(parent_plan)
    require(all(plan["source_bindings"].get(p) == s for p, s in parent_plan["source_bindings"].items()),
            "Missing parent input binding")
    model = load_parent(parent_plan).train().requires_grad_(True)
    teacher, identity = load_teacher("c91", parent_plan["teacher"])
    model_state, teacher_state = state_sha256(model.state_dict()), state_sha256(teacher.state_dict())
    require(model_state == parent_plan["parent"]["model_state_sha256"]
            and teacher_state == identity["model_state_sha256"], "Different parent or teacher")

    # Full production lengths, plus a non-hop-aligned suffix, exercise the
    # exact starting weights without an optimizer or training-mode change.
    geometry = []
    for scored in (1025, 88064):
        teacher_warm = 352256
        audio = torch.randn(1, 2, teacher_warm + scored) * .03
        audio[..., -1] = torch.tensor([[.3125, -.21875]])
        before, rng = audio.clone(), torch.get_rng_state().clone()
        with torch.no_grad():
            shared_teacher = physical_context_teacher(teacher, audio, kind="c91", warmup_samples=teacher_warm)
            teacher_sha = state_sha256({"targets": shared_teacher})
            require(float((shared_teacher.sum(dim=1) - audio[..., teacher_warm:]).abs().max()) <= 1e-6,
                    "Shared teacher physical alignment differs")
            for warm in (88064, 352256):
                view = audio[..., teacher_warm - warm:]
                padded = F.pad(view, (0, (-scored) % 128 + 128))
                state, raw_parts, deployed_parts = model.initial_state(1), [], []
                for chunk in padded.split(128, dim=-1):
                    part = model.render(chunk, state)
                    state = part.state
                    raw_parts.append(part.raw)
                    deployed_parts.append(part.deployed)
                literal_raw = torch.cat(raw_parts, dim=-1)[..., warm + 128:warm + 128 + scored]
                literal_deployed = torch.cat(deployed_parts, dim=-1)[..., warm + 128:warm + 128 + scored]
                output = render_history_view(model, audio, student_history_samples=warm,
                                             teacher_history_samples=teacher_warm, scored_samples=scored)
                replay = render_history_view(model, audio, student_history_samples=warm,
                                             teacher_history_samples=teacher_warm, scored_samples=scored)
                raw_error = float((literal_raw - output.raw).abs().max())
                deployed_error = float((literal_deployed - output.deployed).abs().max())
                closure = float((output.deployed.sum(dim=1) - audio[..., teacher_warm:]).abs().max())
                require(raw_error <= 1e-6 and deployed_error <= 1e-6 and closure <= 1e-6
                        and torch.equal(output.raw, replay.raw) and torch.equal(output.deployed, replay.deployed)
                        and torch.equal(output.physical_mixture, audio[..., teacher_warm:])
                        and output.initial_state_detached and output.flush_hops == 1
                        and output.data_hops == (scored + 127) // 128
                        and state_sha256({"targets": shared_teacher}) == teacher_sha,
                        "Literal stream, replay, alignment, shared target or detached warmup differs")
                geometry.append({"warmup_samples": warm, "teacher_history_samples": teacher_warm,
                                 "scored_samples": scored, "literal_raw_max_abs": raw_error,
                                 "literal_deployed_max_abs": deployed_error, "closure_max_abs": closure,
                                 "physical_alignment_exact": True, "reset_replay_exact": True,
                                 "teacher_targets_sha256": teacher_sha, "initial_state_detached": True})
                del raw_parts, deployed_parts, literal_raw, literal_deployed, output, replay, state, part
        require(torch.equal(audio, before) and torch.equal(rng, torch.get_rng_state()), "Geometry changed input or RNG")
        print({"event": "geometry_pass", "scored_samples": scored, "elapsed_seconds": time.monotonic() - began}, flush=True)

    generator = torch.Generator().manual_seed(20260916)
    teacher_warm, scored = 4096, 1024
    batches = []
    for index in range(4):
        target = torch.randn(4, 4, 2, teacher_warm + scored, generator=generator) * (.01 + index * .005)
        for example in range(4):
            target[example, (index + example) % 4].zero_()
        mixture = target.sum(dim=1).requires_grad_()
        flags = torch.tensor([False, True, False, True])
        teacher_target = physical_context_teacher(teacher, mixture, kind="c91", warmup_samples=teacher_warm)
        batches.append((mixture, target, flags, teacher_target))
    input_hashes = [state_sha256({str(i): v.detach() for i, v in enumerate(batch)}) for batch in batches]
    rng = torch.get_rng_state().clone()
    arms = {}
    for warm in (1024, 4096):
        def objective(batch, *, independent):
            mixture, target, flags, teacher_target = batch
            output = render_history_view(model, mixture, student_history_samples=warm,
                                         teacher_history_samples=teacher_warm, scored_samples=scored)
            reference = target[..., teacher_warm:]
            require(torch.equal(output.physical_mixture, mixture[..., teacher_warm:]), "Gradient suffix differs")
            if independent:
                original = raw4_native_objective(output.raw, reference, flags, projection=True)
                raw_losses = torch.stack([F.l1_loss(output.raw[:, i], reference[:, i]) for i in range(4)])
                teacher_losses = torch.stack([F.l1_loss(output.deployed[:, i], teacher_target[:, i]) for i in range(4)])
                weights = raw_losses.new_tensor([2, 1, 1, 1])
                return (weights * raw_losses).sum() / 5 + original.projection_contribution \
                       + .5 * (weights * teacher_losses).sum() / 5
            return drum_emphasized_objective(output.raw, output.deployed, reference, teacher_target, flags).total

        losses = [objective(batch, independent=True) for batch in batches]
        reference_values = [float(v.detach()) for v in losses]
        joint = torch.stack(losses).mean()
        parameters = tuple(model.parameters())
        reference_gradients = torch.autograd.grad(joint, parameters + (batches[0][0],))
        input_gradient = reference_gradients[-1]
        require(bool((input_gradient[..., :teacher_warm] == 0).all())
                and bool((input_gradient[..., teacher_warm:] != 0).any()), "Warmup graph was not detached")
        del losses, joint
        model.zero_grad(set_to_none=True)
        actual_values = []
        for batch in batches:
            loss = objective(batch, independent=False)
            actual_values.append(float(loss.detach()))
            backward_mean_loss(loss)
            del loss
        require(len(parameters) == 21 and all(abs(a - b) < 1e-7 for a, b in zip(actual_values, reference_values)),
                "Accumulation forward loss differs")
        gradients = {}
        for (name, parameter), expected in zip(model.named_parameters(), reference_gradients[:21], strict=True):
            actual = parameter.grad
            require(actual is not None and bool(torch.isfinite(actual).all()) and bool((actual != 0).any())
                    and torch.allclose(actual, expected, rtol=3e-5, atol=3e-7), "Gradient differs: " + name)
            gradients[name] = {"max_absolute_error": float((actual - expected).abs().max()),
                               "reference_max_absolute": float(expected.abs().max())}
        arms[str(warm)] = {"gradients": gradients, "reference_microbatch_losses": reference_values,
                           "actual_microbatch_losses": actual_values, "all_21_parameter_gradients_match": True,
                           "warmup_input_gradient_zero": True, "scored_input_gradient_nonzero": True,
                           "shared_batch_and_teacher_sha256": input_hashes}
        model.zero_grad(set_to_none=True)
        for batch in batches:
            batch[0].grad = None
        del reference_gradients, input_gradient, expected, actual, parameter
        print({"event": "gradient_pass", "history_samples": warm}, flush=True)
    require(model_state == state_sha256(model.state_dict()) and teacher_state == state_sha256(teacher.state_dict())
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            and all(p.grad is None for p in model.parameters()) and torch.equal(rng, torch.get_rng_state())
            and input_hashes == [state_sha256({str(i): v.detach() for i, v in enumerate(batch)}) for batch in batches]
            and not torch.cuda.is_initialized(), "Functional fixture changed model, teacher, input, RNG or device")
    verify_inputs(plan)
    result = {"schema": "latency58-matched-history-functional-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "history_version": HISTORY_VERSION,
              "accumulation_version": VERSION, "objective_version": OBJECTIVE_VERSION,
              "stem_weights": [2, 1, 1, 1], "teacher_weight": .5, "microbatch_size": 4,
              "accumulation_steps": 4, "all_21_parameter_gradients_match": True,
              "geometry": geometry, "gradient_arms": arms, "model_state_sha256": model_state,
              "teacher_model_state_sha256": teacher_state, "teacher_history_samples": 352256,
              "student_history_samples": [88064, 352256], "scored_samples": 88064,
              "inputs_teacher_and_model_unchanged": True, "render_rng_unchanged": True,
              "optimizer_instances": 0, "training_updates_executed": 0, "checkpoint_written": False,
              "cuda_initialized": False, "quality_selected": False,
              "elapsed_seconds": time.monotonic() - began,
              "limitations": ["Full-length CPU FP32 geometry; gradients use shorter synthetic fixtures.",
                              "Neither CPU proof qualifies full-crop BF16 GPU resources or quality."]}
    write(out / "result.json", result)
    print({"status": "pass", "elapsed_seconds": result["elapsed_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
