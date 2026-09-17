"""Prospective vocal-focused crop views without changing the base RNG draws.

Each group of four absolute examples contains one instrumental view, one
vocals-only view and two unchanged examples from the original augmentation.
Both controlled views use pristine source targets throughout the full crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from research.direct.run_latency58_quality import require

VERSION = "latency58-quarter-instrumental-quarter-vocal-half-original-v1"
VIEW_NAMES = ("instrumental", "vocals_only", "original", "original")


@dataclass(frozen=True)
class VocalFocusBatch:
    mixture: Tensor
    targets: Tensor
    vocal_derangement: Tensor
    view_codes: tuple[int, ...]
    original_augmentation: tuple


def augment_vocal_focus(mixture, targets, *, first_sample_index: int, enabled: bool):
    import torch
    from research import experiment
    require(type(enabled) is bool and type(first_sample_index) is int and first_sample_index >= 0
            and targets.ndim == 4 and targets.shape[0] == 4 and targets.shape[1:3] == (4, 2)
            and mixture.shape == (4, 2, targets.shape[-1]) and targets.shape[-1] > 0
            and mixture.dtype == targets.dtype == torch.float32 and mixture.device == targets.device
            and not targets.requires_grad and not mixture.requires_grad
            and bool(torch.isfinite(mixture).all()) and bool(torch.isfinite(targets).all()),
            "Require four finite physical FP32 examples and absolute data addresses")
    original = experiment._augment_training_distribution(mixture=mixture, targets=targets)
    if not enabled:
        return VocalFocusBatch(*original, (2, 2, 2, 2), original)
    mixed, desired, flags = (value.clone() for value in original)
    codes = tuple((first_sample_index + i) % 4 for i in range(4))
    for index, code in enumerate(codes):
        if code in (0, 1):
            desired[index].zero_()
            if code == 0:
                desired[index, 0] = targets[index, 0]
                desired[index, 1] = targets[index, 1]
                desired[index, 3] = targets[index, 3]
            else:
                desired[index, 2] = targets[index, 2]
            mixed[index] = desired[index].sum(dim=0)
            flags[index] = False
    return VocalFocusBatch(mixed, desired, flags, codes, original)
