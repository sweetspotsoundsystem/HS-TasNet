"""Keep ordinary-mixture distillation while isolating known source-view truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from research.direct.run_latency58_quality import require

VERSION = "latency58-ordinary-only-counterfactual-teacher-v1"


@dataclass(frozen=True)
class CounterfactualTeacherLoss:
    total: Tensor
    waveform_l1: Tensor
    teacher_l1: Tensor
    uniform_teacher_l1: Tensor
    projection: Tensor
    projection_contribution: Tensor
    controlled_examples: int
    ordinary_examples: int
    mode: str


def counterfactual_teacher_objective(raw, deployed, targets, teacher_targets, vocal_derangement,
                                     *, view_codes, mode):
    """Change only teacher participation, retaining the original batch divisor.

    Modes are an exact existing-loss control and ordinary-only distillation.
    The latter still supervises every original source target at native level;
    no amplitude threshold, output gate or target normalization is introduced.
    """
    import torch
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective, TEACHER_WEIGHT

    require(mode in ("all_views", "ordinary_only") and type(view_codes) is tuple
            and len(view_codes) == raw.shape[0]
            and all(type(code) is int and code in (0, 1, 2, 3) for code in view_codes),
            "Require the recorded fixed-view assignments and explicit loss mode")
    original = drum_emphasized_objective(raw, deployed, targets, teacher_targets, vocal_derangement)
    controlled = sum(code < 2 for code in view_codes)
    for index, code in enumerate(view_codes):
        if code < 2:
            require(not bool(vocal_derangement[index]), "Controlled views cannot be vocal-deranged")
            excluded = (2,) if code == 0 else (0, 1, 3)
            require(not bool(torch.count_nonzero(targets[index, list(excluded)])),
                    "Controlled-view target contains an excluded source")
    if mode == "all_views" or controlled == 0:
        teacher_l1, total = original.teacher_l1, original.total
    else:
        with torch.autocast(raw.device.type, enabled=False):
            stem_errors = (deployed - teacher_targets).abs().mean(dim=(2, 3))
            weights = raw.new_tensor([2, 1, 1, 1])
            per_example = (stem_errors * weights[None]).sum(dim=1) / 5
            keep = raw.new_tensor([code >= 2 for code in view_codes])
            # Do not renormalize by the number of ordinary examples: their
            # individual coefficient must stay at the original 0.5 / B.
            teacher_l1 = (per_example * keep).mean()
            total = original.waveform_l1 + original.projection_contribution + TEACHER_WEIGHT * teacher_l1
    require(bool(torch.isfinite(total)), "Counterfactual teacher objective is nonfinite")
    return CounterfactualTeacherLoss(total, original.waveform_l1, teacher_l1, original.teacher_l1,
                                    original.projection, original.projection_contribution,
                                    controlled, len(view_codes) - controlled, mode)
