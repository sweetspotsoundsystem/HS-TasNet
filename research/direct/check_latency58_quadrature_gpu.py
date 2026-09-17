"""Resource-stage check of quadrature warmup at the full training batch size."""
import torch

from research.direct.check_latency58_quadrature import compare_context
from research.direct.latency58_quadrature import Latency58QuadratureModel
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256


def check_gpu(model):
    parameter = next(model.parameters())
    require(type(model) is Latency58QuadratureModel and parameter.device.type == "cuda"
            and model.training and model.training_precision == "bf16"
            and all(p.requires_grad and p.grad is None for p in model.parameters()),
            "Require a fresh BF16 quadrature training model before its first optimizer update")
    fingerprint = state_sha256(model.state_dict())
    saved_expansion = model.phase_expand.weight.detach().clone()
    generator = torch.Generator(device=parameter.device).manual_seed(202609126)
    # A nonzero local fixture exercises both factors. Restoring the exact original
    # expansion keeps resource and production prefixes identical from zero init.
    try:
        with torch.random.fork_rng(devices=[parameter.device.index]), torch.no_grad():
            model.phase_expand.weight.copy_(.002 * torch.randn(
                model.phase_expand.weight.shape, device=parameter.device, generator=generator))
            audio = .03 * torch.randn(16, 2, 88064 + 8 * 128, device=parameter.device, generator=generator)
        result = compare_context(model, audio, 88064)
        torch.cuda.synchronize()
    finally:
        with torch.no_grad():
            model.phase_expand.weight.copy_(saved_expansion)
        model.zero_grad(set_to_none=True)
    require(state_sha256(model.state_dict()) == fingerprint, "GPU fixture changed production initialization")
    return {**result, "precision": "BF16 learned operations; FP32 FFT, public states and loss",
            "temporary_nonzero_expansion_restored": True, "original_model_state_sha256": fingerprint,
            "training_optimizer_updates": 0,
            "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}
