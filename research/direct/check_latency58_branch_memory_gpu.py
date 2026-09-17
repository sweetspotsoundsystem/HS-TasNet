"""Check branch-memory BF16 warmup with all 40 gradients at batch size 16."""
import torch

from research.direct.check_latency58_branch_memory import compare_context
from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256


def check_gpu(model):
    parameter = next(model.parameters())
    require(type(model) is Latency58BranchMemoryModel and parameter.device.type == "cuda"
            and model.training and model.training_precision == "bf16"
            and all(p.requires_grad and p.grad is None for p in model.parameters()),
            "Require a fresh BF16 branch-memory model before the first optimizer update")
    fingerprint = state_sha256(model.state_dict())
    projections = (model.spec_memory_output.weight, model.waveform_memory_output.weight)
    saved = [value.detach().clone() for value in projections]
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    generator = torch.Generator(device=parameter.device).manual_seed(202610212)
    try:
        with torch.random.fork_rng(devices=[parameter.device.index]):
            with torch.no_grad():
                for value in projections:
                    value.copy_(.002 * torch.randn(value.shape, device=value.device, generator=generator))
                audio = .03 * torch.randn(16, 2, 88064 + 8 * 128,
                                         device=parameter.device, generator=generator)
            result = compare_context(model, audio, 88064)
            torch.cuda.synchronize()
    finally:
        with torch.no_grad():
            for value, original in zip(projections, saved, strict=True):
                value.copy_(original)
        model.zero_grad(set_to_none=True)
    require(state_sha256(model.state_dict()) == fingerprint
            and torch.equal(cpu_rng, torch.get_rng_state())
            and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
            "GPU fixture changed production weights or random generator state")
    return {**result, "precision": "BF16 learned operations; FP32 FFT, public states and loss",
            "temporary_nonzero_projections_restored": True, "original_model_state_sha256": fingerprint,
            "trained_parent_attention_unchanged": True, "cpu_and_cuda_rng_restored": True,
            "training_optimizer_updates": 0,
            "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}
