"""Weak, scale-dependent error auxiliary for physically aligned native outputs.

This term supplements the existing raw4 waveform and teacher objectives. It
does not replace their absence penalties or the raw Other head's gradient.
No model, gain, teacher target, crop boundary or inference operation is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


VERSION = "latency58-active-window-log-relative-error-v1"
WINDOW_SAMPLES = 44100
ACTIVITY_POWER = 1e-5
EPSILON = 1e-12
AUXILIARY_WEIGHT = 0.01


@dataclass(frozen=True)
class LogRelativeLoss:
    total: Tensor
    per_stem: Tensor
    per_example_stem: Tensor
    active_window_counts: Tensor
    active_examples_per_stem: Tensor


def log_relative_error(deployed: Tensor, targets: Tensor) -> LogRelativeLoss:
    """Average log(1 + error_power / target_power) on active one-second windows.

    Windows start at the physical scored-crop origin. A final partial window
    uses only real samples. Active windows are averaged within each example
    and stem, active examples within each stem, then active stems equally.
    Reference power above 1e-5 defines activity. Absent windows contribute no
    auxiliary gradient; the caller retains raw4 L1 and teacher L1 for them.

    The logarithm is a smooth training surrogate, not the evaluation SDR in
    decibels. There is no fitted projection or scale-invariant alignment.
    """
    import torch

    if (not isinstance(deployed, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or deployed.ndim != 4 or deployed.shape[0] < 1
            or deployed.shape[1:3] != (4, 2) or deployed.shape[-1] < 1
            or deployed.shape != targets.shape or deployed.device != targets.device
            or deployed.dtype != torch.float32 or targets.dtype != torch.float32
            or targets.requires_grad):
        raise ValueError("Require aligned float32 [B,4,2,T] estimates and fixed physical targets")
    if not bool(torch.isfinite(deployed).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("Audio must be finite")
    with torch.autocast(deployed.device.type, enabled=False):
        losses, masks = [], []
        for start in range(0, deployed.shape[-1], WINDOW_SAMPLES):
            reference = targets[..., start:start + WINDOW_SAMPLES]
            error = deployed[..., start:start + WINDOW_SAMPLES] - reference
            reference_power = reference.square().mean(dim=(-2, -1))
            error_power = error.square().mean(dim=(-2, -1))
            active = reference_power > ACTIVITY_POWER
            value = torch.log1p(error_power / reference_power.clamp_min(ACTIVITY_POWER).add(EPSILON))
            losses.append(torch.where(active, value, torch.zeros_like(value)))
            masks.append(active)
        active_windows = torch.stack(masks, dim=-1)
        counts = active_windows.sum(dim=-1)
        per_example = torch.stack(losses, dim=-1).sum(dim=-1) / counts.clamp_min(1)
        active_examples = counts > 0
        per_stem_count = active_examples.sum(dim=0)
        per_stem = per_example.sum(dim=0) / per_stem_count.clamp_min(1)
        total = per_stem.sum() / (per_stem_count > 0).sum().clamp_min(1)
    if not all(bool(torch.isfinite(value).all()) for value in (total, per_stem, per_example)):
        raise FloatingPointError("Nonfinite logarithmic relative-error loss")
    return LogRelativeLoss(total, per_stem, per_example, counts, per_stem_count)
