"""Optional loss-domain removal of fixed deployment calibration.

This never changes model forwarding, internal features, gains, decoder weights,
or deployed auxiliary losses. Stored source scales include a factor of one half
for the two-branch sum; effective gains are twice those scales.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def uncalibrated_raw_l1(estimates, targets, effective_source_scales):
    """Mean four-head L1 in a fixed inverse-gain output coordinate system."""
    if estimates.shape != targets.shape or estimates.ndim != 4 or estimates.shape[1] != 4:
        raise ValueError("Require matching [batch,4,channels,samples] predictions and targets")
    scales = effective_source_scales.detach().to(device=estimates.device, dtype=torch.float32)
    gains = 2.0 * scales
    if tuple(gains.shape) != (4,) or not bool(torch.isfinite(gains).all()) or not bool((gains > 0).all()):
        raise ValueError("Require four finite positive effective source scales")
    return F.l1_loss(estimates.float() / gains.view(1, 4, 1, 1), targets.float())
