"""Isolated crop alignment and native raw-four-head objective for OLA512.

This module has no trainer, checkpoint loading, augmentation, optimizer, or
device selection.  The caller supplies an OLA512 model and an already augmented
crop, its exact physical targets, and the corresponding derangement flags.
Torch and the existing projection implementation are imported only on use.

Training crops must contain a positive multiple of 256 real samples.  Each crop
starts from one fresh state, receives one final zero hop, and loses only the
global negative-time output hop.  Group boundaries do not detach the graph or
reset the state.  Smaller groups alone do not bound full-crop backward memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from research.direct.latency_ola512 import OLA512Model, OLA512State


HOP = 256
SOURCES = 4
CHANNELS = 2


@dataclass(frozen=True)
class OLACropPlan:
    """All offsets use the physical crop origin, before graph alignment."""

    target_samples: int
    partial_hop_padding: int
    input_samples: int
    output_start: int
    output_stop: int
    call_slices: tuple[tuple[int, int], ...]
    training: bool

    @property
    def data_hops(self) -> int:
        return (self.target_samples + self.partial_hop_padding) // HOP

    @property
    def flush_hops(self) -> int:
        return 1


def plan_ola512_crop(
    samples: int, *, training: bool = True, group_hops: int | None = None,
) -> OLACropPlan:
    """Plan one reset crop with a single global alignment cut.

    ``training=False`` permits a final partial data hop.  Its padding is
    distinct from the mandatory additional zero flush hop.  An empty crop is
    rejected in both modes.  ``group_hops=None`` renders all input in one call.
    """
    if type(samples) is not int or samples <= 0:
        raise ValueError("Crop sample count must be a positive integer")
    if type(training) is not bool:
        raise ValueError("training must be a bool")
    if group_hops is not None and (type(group_hops) is not int or group_hops <= 0):
        raise ValueError("group_hops must be None or a positive integer")
    padding = (-samples) % HOP
    if training and padding:
        raise ValueError("Training crop length must be a positive multiple of 256")
    input_samples = samples + padding + HOP
    call_samples = input_samples if group_hops is None else group_hops * HOP
    call_slices = tuple(
        (start, min(start + call_samples, input_samples))
        for start in range(0, input_samples, call_samples)
    )
    return OLACropPlan(samples, padding, input_samples, HOP, HOP + samples,
                       call_slices, training)


@dataclass(frozen=True)
class AlignedOLACrop:
    """Exact physical [0,T) audio and state advanced through the flush input.

    The final state is diagnostic: it includes synthetic tail input.  Do not
    use this reset-crop API to continue a real stream.  Both output variants
    retain the native source gains already applied by the supplied model.
    """

    raw: Tensor
    deployed: Tensor
    delayed_mixture: Tensor
    final_state: OLA512State
    plan: OLACropPlan


def aligned_ola512_crop(
    model: OLA512Model,
    mixture: Tensor,
    *,
    training: bool = True,
    group_hops: int | None = None,
) -> AlignedOLACrop:
    """Render raw/deployed four-head outputs on the original crop timeline.

    This is differentiable in both modes; ``training=False`` only relaxes the
    length rule.  An evaluation caller owns eval()/inference_mode().  No model
    mode, parameter, source gain, or global precision setting is changed.
    The 256-sample host queue is not part of this offline crop alignment.
    """
    import torch

    if (not isinstance(mixture, torch.Tensor) or mixture.ndim != 3
            or mixture.shape[0] < 1 or mixture.shape[1] != CHANNELS
            or mixture.dtype != torch.float32):
        raise ValueError("Mixture must be float32 [B,2,T] with B positive")
    plan = plan_ola512_crop(mixture.shape[-1], training=training, group_hops=group_hops)
    if (getattr(model, "hop_samples", None) != HOP
            or getattr(model, "graph_alignment_samples", None) != HOP
            or getattr(model, "flush_hops", None) != 1):
        raise ValueError("This helper requires the OLA512 graph-delay/flush contract")

    state = model.initial_state(mixture.shape[0], device=mixture.device)
    # One optional partial-hop pad followed by exactly one full zero flush hop.
    zeros = mixture.new_zeros((*mixture.shape[:-1], plan.partial_hop_padding + HOP))
    padded = torch.cat((mixture, zeros), dim=-1)
    raw_parts, deployed_parts, mixture_parts = [], [], []
    for start, stop in plan.call_slices:
        output = model.render(padded[..., start:stop], state)
        expected_raw = (mixture.shape[0], SOURCES, CHANNELS, stop - start)
        expected_mix = (mixture.shape[0], CHANNELS, stop - start)
        if (tuple(output.raw.shape) != expected_raw
                or tuple(output.deployed.shape) != expected_raw
                or tuple(output.delayed_mixture.shape) != expected_mix):
            raise ValueError("OLA output shape differs from the grouped render contract")
        raw_parts.append(output.raw)
        deployed_parts.append(output.deployed)
        mixture_parts.append(output.delayed_mixture)
        # Preserve all four state graphs; there is no truncation at a call edge.
        state = output.state

    def aligned(parts):
        joined = torch.cat(parts, dim=-1)
        return joined[..., plan.output_start:plan.output_stop]

    return AlignedOLACrop(aligned(raw_parts), aligned(deployed_parts),
                          aligned(mixture_parts), state, plan)


@dataclass(frozen=True)
class OLARaw4Loss:
    total: Tensor
    waveform_l1: Tensor
    projection: Tensor
    projection_contribution: Tensor


def raw4_native_objective(
    estimates: Tensor,
    targets: Tensor,
    vocal_derangement: Tensor,
    *,
    projection: bool = True,
) -> OLARaw4Loss:
    """Native-gain raw4 L1 plus the existing capped deranged projection only.

    Pass ``aligned_ola512_crop(...).raw`` and the original [B,4,2,T] targets.
    Targets are never padded, shifted, curtailed, or rescaled.  Augmentation
    belongs to the caller and must precede alignment.  Projection retains the
    production final-one-second activity mask and its detached 0.5%-L1 cap.
    This helper does not perform backward, clipping, or an optimizer update.
    """
    import torch

    if (not isinstance(estimates, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or estimates.ndim != 4 or estimates.shape[0] < 1
            or estimates.shape[1:3] != (SOURCES, CHANNELS) or estimates.shape[-1] < 1
            or estimates.shape != targets.shape
            or estimates.dtype != torch.float32 or targets.dtype != torch.float32
            or estimates.device != targets.device):
        raise ValueError("Objective requires matching float32 [B,4,2,T] estimates and physical targets")
    if (not isinstance(vocal_derangement, torch.Tensor)
            or vocal_derangement.shape != (estimates.shape[0],)
            or vocal_derangement.dtype != torch.bool
            or vocal_derangement.device != estimates.device):
        raise ValueError("Derangement flags must be bool [B] on the audio device")
    if type(projection) is not bool:
        raise ValueError("projection must be a bool")

    with torch.autocast(estimates.device.type, enabled=False):
        waveform_loss = torch.nn.functional.l1_loss(estimates.float(), targets.float())
        projection_loss = waveform_loss.new_zeros(())
        contribution = waveform_loss.new_zeros(())
        if projection:
            from research import experiment

            projection_loss = experiment._deranged_vocal_projection_loss(
                estimates=estimates,
                targets=targets,
                vocal_derangement=vocal_derangement,
            )
            weight = torch.minimum(
                projection_loss.new_tensor(experiment.DERANGED_PROJECTION_LOSS_WEIGHT),
                experiment.DERANGED_PROJECTION_MAX_L1_FRACTION
                * waveform_loss.detach() / projection_loss.detach().clamp_min(1e-8),
            )
            contribution = weight * projection_loss
        total = waveform_loss + contribution
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite OLA crop objective")
    return OLARaw4Loss(total, waveform_loss, projection_loss, contribution)
