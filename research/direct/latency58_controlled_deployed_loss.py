"""Add source truth for deployed outputs on explicit controlled training views."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from research.direct.latency58_counterfactual_teacher import CounterfactualTeacherLoss

from research.direct.run_latency58_quality import require

VERSION = "latency58-controlled-deployed-truth-half-full-batch-v1"
DEPLOYED_TRUTH_WEIGHT = 0.5


@dataclass(frozen=True)
class ControlledDeployedLoss:
    total: Tensor
    base: CounterfactualTeacherLoss
    controlled_deployed_l1: Tensor
    controlled_deployed_contribution: Tensor


def controlled_deployed_objective(raw, deployed, targets, teacher_targets, vocal_derangement,
                                 *, view_codes, weight=DEPLOYED_TRUTH_WEIGHT):
    """Keep ordinary-only distillation and add native deployed source supervision.

    The fixed candidate coefficient is 0.5; zero is an exact identity fixture.
    Explicit instrumental/vocal-only examples contribute with the original
    full-batch divisor. Ordinary examples retain their original coefficients.
    Raw supervision, teacher targets and the existing projection cap remain.
    There is no amplitude threshold, reference normalization or inference change.
    """
    import torch
    from research.direct.latency58_counterfactual_teacher import counterfactual_teacher_objective

    require(type(weight) in (int, float) and weight in (0, DEPLOYED_TRUTH_WEIGHT),
            "Use the fixed additional weight or its exact zero control")
    base = counterfactual_teacher_objective(raw, deployed, targets, teacher_targets, vocal_derangement,
                                           view_codes=view_codes, mode="ordinary_only")
    with torch.autocast(raw.device.type, enabled=False):
        errors = (deployed - targets).abs().mean(dim=(2, 3))
        stem_weights = raw.new_tensor([2, 1, 1, 1])
        per_example = (errors * stem_weights[None]).sum(dim=1) / 5
        controlled = raw.new_tensor([code < 2 for code in view_codes])
        auxiliary = (per_example * controlled).mean()
        contribution = weight * auxiliary
        total = base.total if weight == 0 else base.total + contribution
    require(bool(torch.isfinite(total)), "Controlled deployed objective is nonfinite")
    return ControlledDeployedLoss(total, base, auxiliary, contribution)
