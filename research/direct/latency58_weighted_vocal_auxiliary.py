"""Quarter-weight vocals-only supervision with the existing two-view reduction."""
from dataclasses import dataclass
import math

import torch

from research.direct.run_latency58_quality import require
from research.direct.latency58_logical_batch_loss import prepare_reduction, objective as contribution
from research.direct.latency58_grouped_vocal_auxiliary import policy as original_policy


VERSION = "latency58-joint-source-views-instrumental-one-vocals-quarter-v1"
VIEW_WEIGHTS = (1., .25)


@dataclass(frozen=True)
class WeightedAuxiliaryLoss:
    total: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor
    unweighted_view_contributions: torch.Tensor
    weighted_view_contributions: torch.Tensor


def objective(raw, deployed, targets, mixture, *, weights=VIEW_WEIGHTS):
    """Weight the two complete-group contributions without changing counts.

    Reconstruction keeps its original example denominator of two. Direct SDR
    and absence keep the active/absent counts and eligible-stem counts of both
    views. The outer group coefficient of 0.1 is applied by the accumulator.
    Explicit weights support CPU controls; production uses the fixed default.
    """
    require(raw.ndim == 4 and raw.shape == deployed.shape == targets.shape
            and raw.shape[:3] == (2, 4, 2) and mixture.shape == (2, 2, raw.shape[-1])
            and type(weights) is tuple and len(weights) == 2
            and all(type(w) is float and math.isfinite(w) for w in weights)
            and weights[0] == 1. and 0 <= weights[1] <= 1.,
            "Require exactly the ordered source-view pair and valid fixed view weights")
    require(torch.count_nonzero(targets[0, 2]).item() == 0
            and torch.count_nonzero(targets[1, [0, 1, 3]]).item() == 0,
            "Source-view target ordering or removed stems changed")
    reduction = prepare_reduction(targets)
    terms = [contribution(raw[i:i + 1], deployed[i:i + 1], targets[i:i + 1], mixture[i:i + 1], reduction)
             for i in (0, 1)]
    unweighted = torch.stack([term.total for term in terms])
    weighted = unweighted * unweighted.new_tensor(weights)
    total = weighted.sum()
    require(bool(torch.isfinite(total)), "Nonfinite weighted source-view objective")
    return WeightedAuxiliaryLoss(total, reduction.active, reduction.absent, unweighted, weighted)


def policy():
    return {**original_policy(), "version": VERSION,
            "view_contribution_multipliers": list(VIEW_WEIGHTS),
            "effective_coefficients_in_joint_reduction": [.1, .025],
            "view_weight_renormalization": False,
            "auxiliary_reduction": "Existing complete two-view activity and example denominators, then view weights and outer 0.1",
            "ordinary_objective_changed": False,
            "scientific_change_selected": True}
