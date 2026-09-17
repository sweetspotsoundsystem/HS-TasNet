"""Qualify two-second B8 context gradients and B16 loss accumulation on CUDA."""
import torch

from research.direct.check_latency58_branch_memory import compare_context
from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
from research.direct.latency58_branch_sdr_blend import objective as full_objective
from research.direct.latency58_logical_batch_loss import objective, prepare_reduction
from research.direct.latency58_long_context_data import WARMUP_SAMPLES, SCORED_SAMPLES, CROP_SAMPLES
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256


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
