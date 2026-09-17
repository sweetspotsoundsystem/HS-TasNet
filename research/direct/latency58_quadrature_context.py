"""Detached warmup with final-frame decoding for the quadrature prototype."""
import torch
from torch.nn import functional as F

from research.direct.latency58_sdr_context import ContextOutput
from research.direct.latency58_quadrature import Latency58QuadratureModel
from research.direct.run_latency58_quality import require


def render_scored_context(model, mixture, *, warmup_samples, carry_state):
    require(carry_state is True and type(warmup_samples) is int and warmup_samples > 0
            and warmup_samples % 128 == 0 and mixture.ndim == 3 and mixture.shape[1] == 2
            and mixture.shape[-1] > warmup_samples and mixture.dtype == torch.float32,
            "Invalid quadrature training context")
    require(type(model) is Latency58QuadratureModel, "Require the quadrature model")
    state = model.warm_state(mixture[..., :warmup_samples]).detached()
    require(all(not value.requires_grad and value.grad_fn is None for value in state), "Warmup retained gradients")
    score = mixture[..., warmup_samples:]
    count, padding = score.shape[-1], (-score.shape[-1]) % 128
    result = model.render(F.pad(score, (0, padding + 128)), state)
    raw, deployed, physical = (value[..., 128:128 + count] for value in
                               (result.raw, result.deployed, result.delayed_mixture))
    require(raw.shape == deployed.shape == (mixture.shape[0], 4, 2, count)
            and torch.equal(physical, score), "Quadrature context lost physical alignment")
    return ContextOutput(raw, deployed, physical, warmup_samples, count, True, True,
                         (count + padding) // 128, 1)
