"""Deployed-four L1 with the unchanged raw-four-derived projection contribution.

Inputs are already aligned and at native output gains. The caller constructs
deployed Other through the actual model path; no target or gain is rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

VERSION = "cropped1024-deployed4-l1-rawcap-v1"


@dataclass(frozen=True)
class DeployedL1Loss:
    total: Tensor
    waveform_l1: Tensor
    raw_waveform_l1: Tensor
    projection: Tensor
    projection_contribution: Tensor
    projection_cap: Tensor


def deployed_l1_objective(
    raw: Tensor, deployed: Tensor, targets: Tensor, deranged: Tensor,
    *, projection: bool = True,
) -> DeployedL1Loss:
    """mean(abs(deployed-targets)) + original_raw_objective.contribution.

    The detached cap is 0.005*raw_L1, recomputed on the current predictions.
    Keeping that contribution directly avoids changing its weighting or its
    two detach points. All reconstruction samples and stems have equal weight.
    """
    import torch
    from research.direct.latency_ola512_training import raw4_native_objective

    if (not isinstance(raw, torch.Tensor) or not isinstance(deployed, torch.Tensor)
            or raw.shape != deployed.shape or raw.device != deployed.device
            or deployed.dtype != torch.float32 or deployed.layout != torch.strided):
        raise ValueError("raw/deployed must match as dense native-gain float32 audio")
    # Original helper validates the raw/target geometry, flags and bool option.
    original = raw4_native_objective(raw, targets, deranged, projection=projection)
    if not bool(torch.isfinite(deployed).all()):
        raise FloatingPointError("Non-finite deployed audio")
    with torch.autocast(deployed.device.type, enabled=False):
        reconstruction = torch.nn.functional.l1_loss(deployed, targets)
        total = reconstruction + original.projection_contribution
        cap = 0.005 * original.waveform_l1.detach()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Non-finite deployed L1 objective")
    return DeployedL1Loss(total, reconstruction, original.waveform_l1,
                          original.projection, original.projection_contribution, cap)
