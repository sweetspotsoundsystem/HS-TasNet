"""Prospective source-view supervision without replacing ordinary mixtures.

The current sixteen-example objective keeps its own complete-batch reduction.
Two additional source views have a separate reduction and fixed weight 0.1.
This changes training only; it does not select a parent or start a GPU trial.
"""
from dataclasses import dataclass

import torch

from research.direct.latency58_logical_batch_loss import BatchReduction, prepare_reduction, objective

VERSION = "latency58-ordinary16-plus-two-source-views-tenth-loss-v1"
AUXILIARY_WEIGHT = .1
VIEW_INDICES = (14, 15)
VIEW_NAMES = ("instrumental", "vocals_only")


@dataclass(frozen=True)
class Groups:
    ordinary: BatchReduction
    auxiliary: BatchReduction


def source_views(mixture, targets):
    """Derive additional full-context inputs after ordinary augmentation.

Call on the entire warmup-plus-scored crop. Every view needs freshly computed
warmup states; never remove a source only in the scored suffix or reuse states
from the corresponding ordinary mixture. Inputs are never modified in place.
"""
    if (targets.ndim != 4 or targets.shape[:3] != (16, 4, 2)
            or mixture.shape != (16, 2, targets.shape[-1])
            or targets.dtype != torch.float32 or mixture.dtype != torch.float32
            or targets.requires_grad or mixture.requires_grad or targets.device != mixture.device):
        raise ValueError("Require sixteen fixed FP32 augmented stereo mixtures and references")
    if not bool(torch.isfinite(targets).all()) or not bool(torch.isfinite(mixture).all()):
        raise ValueError("Nonfinite source-view inputs")
    selected = targets[list(VIEW_INDICES)].clone()
    keep = torch.tensor([[True, True, False, True], [False, False, True, False]],
                        dtype=torch.bool, device=targets.device)
    selected *= keep[:, :, None, None]
    return selected.sum(1), selected


def prepare_groups(ordinary_targets, auxiliary_targets):
    if ordinary_targets.shape[0] != 16 or auxiliary_targets.shape[0] != 2:
        raise ValueError("Keep sixteen ordinary examples and two auxiliary views")
    if ordinary_targets.shape[1:] != auxiliary_targets.shape[1:]:
        raise ValueError("Both groups must share scored geometry")
    return Groups(prepare_reduction(ordinary_targets), prepare_reduction(auxiliary_targets))


def contribution(group, raw, deployed, targets, mixture, groups):
    """Sum returned contributions, then clip/update/average weights once.

Groups have independent activity denominators. The ordinary contribution is
unchanged, even when all auxiliary vocals or instrumental sources are silent.
The second result retains unweighted components and integer activity counts
for auditing; only the first result should enter the accumulated scalar loss.
"""
    if not isinstance(groups, Groups) or group not in ("ordinary", "auxiliary"):
        raise ValueError("Choose an explicit group with prepared reductions")
    terms = objective(raw, deployed, targets, mixture, getattr(groups, group))
    return terms.total if group == "ordinary" else AUXILIARY_WEIGHT * terms.total, terms


def policy():
    return {"version": VERSION, "ordinary_examples": 16, "auxiliary_examples": 2,
        "auxiliary_weight": AUXILIARY_WEIGHT, "view_indices_in_augmented_batch": list(VIEW_INDICES),
        "view_names": list(VIEW_NAMES), "ordinary_mixture_and_targets_unchanged": True,
        "ordinary_loss_and_activity_denominators_unchanged": True,
        "auxiliary_reduction": "Independent whole-view-group counts, then fixed weight 0.1",
        "source_removal": "Entire received warmup and scored context, before computing any carried states",
        "states": "Independent fresh warmup per view; no states reused from ordinary input",
        "optimizer": "One gradient clip, Adam update and EMA update after both groups",
        "additional_forward_examples_fraction": 2 / 16,
        "inference_architecture_changed": False, "production_recipe_selected": False}
