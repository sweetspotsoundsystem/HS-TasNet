"""Qualify two-second B8 context gradients and B16 loss accumulation on CUDA."""
import torch

import gc
import weakref
from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
from research.direct.latency58_branch_sdr_blend import objective as full_objective
from research.direct.latency58_logical_batch_loss import objective, prepare_reduction
from research.direct.latency58_long_context_data import WARMUP_SAMPLES, SCORED_SAMPLES, CROP_SAMPLES
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256


def compare_context(model, mixture, warmup):
    import torch
    from research.direct.latency58_sdr_context import render_scored_context as full
    from research.direct.latency58_branch_memory_context import render_scored_context as fast
    fingerprint, rows = state_sha256(model.state_dict()), []
    lifetimes, memory = [], []

    def capture(render):
        model.zero_grad(set_to_none=True)
        audio = mixture.detach().clone().requires_grad_(True)
        result = render(model, audio, warmup_samples=warmup, carry_state=True)
        loss = result.raw.square().mean() + result.deployed.square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "Missing or nonfinite model gradient")
        require(torch.count_nonzero(audio.grad[..., :warmup]) == 0
                and torch.count_nonzero(audio.grad[..., warmup:]) > 0, "Detached warmup boundary changed")
        row = {"raw": result.raw.detach().cpu(), "deployed": result.deployed.detach().cpu(),
                     "physical": result.physical_mixture.detach().cpu(), "input_gradient": audio.grad.detach().cpu(),
                     "gradients": {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
                     "loss": float(loss.detach())}
        refs = [weakref.ref(value) for value in (audio, result.raw, result.deployed, loss)]
        return row, refs
    for name, render in (("reference", full), ("optimized", fast)):
        model.zero_grad(set_to_none=True)
        gc.collect()
        before = torch.cuda.memory_allocated(mixture.device) if mixture.is_cuda else None
        row, refs = capture(render)
        rows.append(row)
        model.zero_grad(set_to_none=True)
        gc.collect()
        released = all(ref() is None for ref in refs)
        require(released, "Completed comparison retained its audio/output/loss tensors")
        after = torch.cuda.memory_allocated(mixture.device) if mixture.is_cuda else None
        lifetimes.append({"pass": name, "audio_output_and_loss_released": released})
        memory.append({"pass": name, "allocated_before_forward": before, "allocated_after_release": after})
    first, second = rows
    errors = {k: float((first[k] - second[k]).abs().max()) for k in ("raw", "deployed", "physical", "input_gradient")}
    gradients = {name: {"maximum_error": float((value - second["gradients"][name]).abs().max()),
                        "reference_norm": float(value.norm())} for name, value in first["gradients"].items()}
    require(all(v == 0 for v in errors.values()) and first["loss"] == second["loss"]
            and len(gradients) == 40 and all(v["maximum_error"] == 0 and v["reference_norm"] > 0 for v in gradients.values()),
            "Warmup optimization changed scored output/gradients or left a parameter unexercised")
    require(state_sha256(model.state_dict()) == fingerprint, "Context comparison changed the model")
    model.zero_grad(set_to_none=True)
    return {"status": "pass", "output_and_input_gradient_errors": errors, "all_40_gradients": gradients,
            "warmup_input_gradient_zero": True, "scored_samples": mixture.shape[-1] - warmup,
            "loss": first["loss"], "model_unchanged": True,
            "completed_pass_tensors_released": lifetimes, "pass_memory_bytes": memory}


def check_loss(device, generator):
    target = .03 * torch.randn(16, 4, 2, SCORED_SAMPLES, device=device, generator=generator)
    target[:8, 1] = 0
    target[8:, 2] = 0
    target[0, 3, :, :44100] = .0001
    mixture = target.sum(1)
    noise = .008 * torch.randn(target.shape, device=device, generator=generator)
    noise[8:] *= 3
    raw, deployed = (target + noise).requires_grad_(), (target + .7 * noise).requires_grad_()
    full = full_objective(raw, deployed, target, mixture)
    expected = torch.autograd.grad(full.total, (raw, deployed))
    reduction = prepare_reduction(target)
    pieces = [objective(raw[i:i + 8], deployed[i:i + 8], target[i:i + 8], mixture[i:i + 8], reduction)
              for i in (0, 8)]
    combined = sum(p.total for p in pieces)
    actual = torch.autograd.grad(combined, (raw, deployed))
    errors = [float((a - b).abs().max()) for a, b in zip(actual, expected, strict=True)]
    require(torch.allclose(combined, full.total, atol=3e-6, rtol=3e-6)
            and all(torch.allclose(a, b, atol=2e-9, rtol=3e-5) for a, b in zip(actual, expected, strict=True))
            and torch.equal(sum(p.active_window_counts for p in pieces), reduction.active)
            and torch.equal(sum(p.absent_window_counts for p in pieces), reduction.absent),
            "CUDA microbatch reduction differs from the original full-batch objective")
    return {"status": "pass", "logical_batch_size": 16, "microbatch_size": 8,
            "scored_samples": SCORED_SAMPLES, "loss_absolute_error": float((combined - full.total).detach().abs()),
            "raw_and_deployed_gradient_max_abs_errors": errors,
            "active_windows": reduction.active.cpu().tolist(), "absent_windows": reduction.absent.cpu().tolist(),
            "scope": "Loss and gradients in audio-output coordinates; no full-B16 neural activation allocation"}


def check_gpu(model):
    parameter = next(model.parameters())
    require(type(model) is Latency58BranchMemoryModel and parameter.device.type == "cuda"
            and model.training and model.training_precision == "bf16"
            and all(p.requires_grad and p.grad is None for p in model.parameters()),
            "Require the selected trained model before its first optimizer update")
    fingerprint = state_sha256(model.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    generator = torch.Generator(device=parameter.device).manual_seed(202610292)
    try:
        with torch.random.fork_rng(devices=[parameter.device.index]):
            audio = .03 * torch.randn(8, 2, CROP_SAMPLES, device=parameter.device, generator=generator)
            context = compare_context(model, audio, WARMUP_SAMPLES)
            del audio
            loss = check_loss(parameter.device, generator)
            torch.cuda.synchronize()
    finally:
        model.zero_grad(set_to_none=True)
    require(state_sha256(model.state_dict()) == fingerprint
            and torch.equal(cpu_rng, torch.get_rng_state())
            and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
            "Long-context qualification changed model weights or RNG states")
    return {**context, "logical_batch_loss": loss, "original_model_state_sha256": fingerprint,
            "trained_parent_weights_unchanged": True, "cpu_and_cuda_rng_restored": True,
            "precision": "BF16 learned operations; FP32 FFT, states and loss", "microbatch_size": 8,
            "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
            "training_optimizer_updates": 0, "inference_architecture_changed": False,
            "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}
