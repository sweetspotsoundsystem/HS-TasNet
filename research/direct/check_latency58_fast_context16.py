"""Exact scored-gradient check for the full sixteen-example GPU batch."""
import torch

from research.direct.check_latency58_fast_context import compare
from research.direct.run_latency58_quality import require


def check_gpu(model):
    require(next(model.parameters()).device.type == "cuda" and model.training_precision == "bf16",
            "Require training BF16 CUDA model")
    generator = torch.Generator(device="cuda").manual_seed(202609121)
    audio = .03 * torch.randn(16, 2, 88064 + 8 * 128, device="cuda", generator=generator)
    result = compare(model, audio, 88064)
    torch.cuda.synchronize()
    return {**result, "precision": "bf16 learned operations, FP32 FFT/state/loss",
            "training_optimizer_updates": 0, "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}
