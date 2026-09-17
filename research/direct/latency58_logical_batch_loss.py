"""Accumulate the existing SDR blend using whole-batch activity denominators.

Reconstruction and raw-head anchors average over examples. SDR and absence
instead average each stem over its eligible one-second windows, then over
eligible stems. Averaging independent microbatch losses changes that objective
when active or absent window counts differ between microbatches.
"""
from dataclasses import dataclass

import torch

from research.direct.latency58_branch_sdr_blend import BlendedLoss, SDR_WEIGHT, VERSION as OBJECTIVE_VERSION
from research.direct.latency58_direct_sdr import WINDOW, ACTIVITY_POWER, ABSENCE_WEIGHT, ANCHOR_WEIGHT
from research.direct.latency58_wave_spectral import objective as reconstruction

VERSION = "latency58-whole-batch-active-absence-normalized-accumulation-v1"


@dataclass(frozen=True)
class BatchReduction:
    examples: int
    samples: int
    active: torch.Tensor
    absent: torch.Tensor


def activity_counts(targets):
    if (targets.ndim != 4 or targets.shape[0] < 1 or targets.shape[1:3] != (4, 2)
            or targets.shape[-1] < WINDOW or targets.dtype != torch.float32 or targets.requires_grad):
        raise ValueError("Require fixed FP32 four-stem stereo references")
    windows = targets.shape[-1] // WINDOW
    with torch.no_grad(), torch.autocast(targets.device.type, enabled=False):
        reference = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        signal = reference.square().sum((2, 4))
        active = signal / (2 * WINDOW) > ACTIVITY_POWER
        return active.sum((0, 2)), (~active).sum((0, 2))


def prepare_reduction(targets):
    active, absent = activity_counts(targets)
    return BatchReduction(targets.shape[0], targets.shape[-1], active, absent)


def objective(raw, deployed, targets, mixture, reduction):
    """Return this microbatch's contribution; sum contributions before Adam.

The reduction is computed on the training device from every reference in the
logical batch. Neither the returned loss nor its gradient needs a further
division by the number of microbatches.
"""
    if (not isinstance(reduction, BatchReduction) or raw.ndim != 4
            or raw.shape != deployed.shape or raw.shape != targets.shape
            or raw.shape[1:3] != (4, 2) or not 0 < raw.shape[0] <= reduction.examples
            or raw.shape[-1] != reduction.samples or reduction.samples < WINDOW
            or mixture.shape != (raw.shape[0], 2, raw.shape[-1])
            or any(v.dtype != torch.float32 or v.device != raw.device for v in (raw, deployed, targets, mixture))
            or targets.requires_grad or mixture.requires_grad
            or any(v.shape != (4,) or v.dtype != torch.int64 or v.device != raw.device
                   or v.requires_grad or bool((v < 0).any()) for v in (reduction.active, reduction.absent))
            or not bool(torch.all(reduction.active + reduction.absent ==
                                  reduction.examples * (reduction.samples // WINDOW)))):
        raise ValueError("Require matching microbatch audio and whole-batch activity counts")
    fraction = raw.shape[0] / reduction.examples
    base = reconstruction(raw, deployed, targets, mixture)
    windows = raw.shape[-1] // WINDOW
    with torch.autocast(raw.device.type, enabled=False):
        reference = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        estimate = deployed[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        raw_windows = raw[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        physical = mixture[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        signal = reference.square().sum((2, 4))
        error = (estimate - reference).square().sum((2, 4))
        active = signal / (2 * WINDOW) > ACTIVITY_POWER
        active_counts, absent_counts = active.sum((0, 2)), (~active).sum((0, 2))
        if bool((active_counts > reduction.active).any()) or bool((absent_counts > reduction.absent).any()):
            raise ValueError("Microbatch activity exceeds the declared logical batch")
        values = (10 * torch.log10((error + 1e-12) / (signal + 1e-12))).clamp(-60, 60)
        per_stem = torch.where(active, values, 0).sum((0, 2)) / reduction.active.clamp_min(1)
        primary = per_stem.sum() / (reduction.active > 0).sum().clamp_min(1)
        mixture_power = physical.square().mean((1, 3))
        leakage = estimate.square().mean((2, 4))
        absence_values = 10 * torch.log10(1 + leakage / mixture_power[:, None].clamp_min(ACTIVITY_POWER))
        absence_per_stem = torch.where(~active, absence_values, 0).sum((0, 2)) / reduction.absent.clamp_min(1)
        absence = absence_per_stem.sum() / (reduction.absent > 0).sum().clamp_min(1)
        scale = torch.maximum(signal / (2 * WINDOW), .01 * mixture_power[:, None]).clamp_min(ACTIVITY_POWER).sqrt()
        anchor = ((raw_windows - reference).abs().mean((2, 4)) / scale).mean() * fraction
        direct = primary + ABSENCE_WEIGHT * absence + ANCHOR_WEIGHT * anchor
        reconstruction_loss = base.total * fraction
        total = reconstruction_loss + SDR_WEIGHT * direct
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite globally normalized microbatch loss")
    return BlendedLoss(total, base.waveform * fraction, base.spectral * fraction,
                       base.raw_anchor * fraction, primary, per_stem, active_counts, absent_counts,
                       reconstruction_loss, direct, absence, anchor)


def policy():
    return {"version": VERSION, "objective_version": OBJECTIVE_VERSION,
            "direct_sdr_weight": SDR_WEIGHT,
            "reconstruction_and_raw_anchor": "example-weighted mean across the logical batch",
            "sdr_and_absence": "whole-batch eligible-window counts followed by eligible-stem mean",
            "activity_device": "training device before splitting the reference batch",
            "microbatch_backward": "sum already-normalized contributions; no extra division",
            "gradient_clip_and_adam_and_ema": "once after the complete logical batch"}
