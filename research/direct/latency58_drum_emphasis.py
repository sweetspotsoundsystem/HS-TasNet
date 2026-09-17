"""Prospective normalized drum weighting for the unchanged training outputs.

The raw supervised and deployed teacher L1 terms use stem weights [2,1,1,1]
divided by five. The existing deranged-vocal projection and its cap remain
computed from the original unweighted raw4 objective. No inference operation
or validation metric is changed, and this module does not authorize training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

VERSION = "latency58-normalized-drums-double-l1-v1"
DRUM_WEIGHT = 2.0
TEACHER_WEIGHT = 0.5


@dataclass(frozen=True)
class DrumEmphasisLoss:
    total: Tensor
    waveform_l1: Tensor
    teacher_l1: Tensor
    unweighted_waveform_l1: Tensor
    unweighted_teacher_l1: Tensor
    raw_drum_l1: Tensor
    teacher_drum_l1: Tensor
    projection: Tensor
    projection_contribution: Tensor


def drum_emphasized_objective(raw, deployed, targets, teacher_targets, vocal_derangement,
                             *, drum_weight=DRUM_WEIGHT):
    """Retain average scale while doubling the drum weight in both L1 terms.

Weight one is an exact identity branch for qualification. The original
projection cap is preserved in both branches, including its detached weight.
All references describe the same physical FP32 [B,4,2,T] samples.
"""
    import torch
    from research.direct.latency_ola512_training import raw4_native_objective
    from research.direct.latency58_teacher import deployed_teacher_l1

    values = (raw, deployed, targets, teacher_targets)
    if (not all(isinstance(value, torch.Tensor) for value in values)
            or raw.ndim != 4 or raw.shape[0] < 1 or raw.shape[1:3] != (4, 2) or raw.shape[-1] < 1
            or any(value.shape != raw.shape or value.dtype != torch.float32 or value.device != raw.device
                   or not bool(torch.isfinite(value).all()) for value in values)
            or targets.requires_grad or teacher_targets.requires_grad
            or type(drum_weight) not in (int, float) or drum_weight not in (1, DRUM_WEIGHT)):
        raise ValueError("Require aligned finite FP32 raw/deployed audio, fixed references and drum weight 1 or 2")
    with torch.autocast(raw.device.type, enabled=False):
        original = raw4_native_objective(raw, targets, vocal_derangement, projection=True)
        original_teacher = deployed_teacher_l1(deployed, teacher_targets)
        drum_raw = torch.nn.functional.l1_loss(raw[:, 0], targets[:, 0])
        drum_teacher = torch.nn.functional.l1_loss(deployed[:, 0], teacher_targets[:, 0])
        if drum_weight == 1:
            waveform, teacher = original.waveform_l1, original_teacher
            total = original.total + TEACHER_WEIGHT * original_teacher
        else:
            waveform = (4 * original.waveform_l1 + drum_raw) / 5
            teacher = (4 * original_teacher + drum_teacher) / 5
            total = waveform + original.projection_contribution + TEACHER_WEIGHT * teacher
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite drum-emphasized objective")
    return DrumEmphasisLoss(total, waveform, teacher, original.waveform_l1, original_teacher,
                            drum_raw, drum_teacher, original.projection, original.projection_contribution)
