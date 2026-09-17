"""Training-only state warmup on preceding audio, with an unchanged student."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from research.direct.latency58_checkpoint import require


@dataclass(frozen=True)
class ContextOutput:
    raw: torch.Tensor
    deployed: torch.Tensor
    physical_mixture: torch.Tensor
    warmup_samples: int
    scored_samples: int
    carried_state: bool
    initial_state_detached: bool
    data_hops: int
    flush_hops: int


def render_scored_context(model, mixture, *, warmup_samples, carry_state):
    """Score the same suffix after either detached real history or a reset.

    Both arms receive the same complete augmented crop. The reset control
    ignores its prefix, while the warm arm advances all four native states
    without constructing a gradient graph for that prefix. No intermediate
    flush, mode switch, gain change, or inference architecture change occurs.
    """
    require(type(warmup_samples) is int and warmup_samples > 0 and warmup_samples % 128 == 0
            and type(carry_state) is bool and mixture.ndim == 3 and mixture.shape[0] > 0
            and mixture.shape[1] == 2 and mixture.shape[-1] > warmup_samples
            and mixture.dtype == torch.float32 and model.hop_samples == model.graph_alignment_samples == 128,
            "Invalid scored context request")
    score = mixture[..., warmup_samples:]
    count = score.shape[-1]
    if carry_state:
        with torch.no_grad():
            state = model.render(mixture[..., :warmup_samples]).state.detached()
    else:
        state = model.initial_state(mixture.shape[0], device=mixture.device)
    detached = all(not value.requires_grad and value.grad_fn is None for value in state)
    require(detached, "Warmup retained an autograd graph")
    padding = (-count) % 128
    output = model.render(F.pad(score, (0, padding + 128)), state)
    raw, deployed, physical = (value[..., 128:128 + count] for value in
                               (output.raw, output.deployed, output.delayed_mixture))
    require(raw.shape == deployed.shape == (mixture.shape[0], 4, 2, count)
            and physical.shape == score.shape, "Scored context shape differs")
    return ContextOutput(raw, deployed, physical, warmup_samples, count, carry_state,
                         detached, (count + padding) // 128, 1)


def physical_context_teacher(teacher, mixture, *, kind, warmup_samples):
    """Give both student arms the same teacher targets with preceding audio."""
    from research.direct.latency58_sdr_teacher import physical_targets
    require(type(warmup_samples) is int and 0 < warmup_samples < mixture.shape[-1]
            and warmup_samples % 128 == 0, "Invalid teacher context boundary")
    target = physical_targets(teacher, mixture, kind=kind)[..., warmup_samples:]
    require(not target.requires_grad and target.grad_fn is None
            and target.shape == (mixture.shape[0], 4, 2, mixture.shape[-1] - warmup_samples),
            "Teacher context targets are live or misaligned")
    return target
