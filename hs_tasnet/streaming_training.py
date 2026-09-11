"""Training views, aligned context and native-level separation losses.

This retains the released model's raw four-source supervision, normalized
2:1:1:1 stem weights, capped deranged-vocal projection, ordinary-view teacher
participation and controlled-view deployed supervision. Teacher targets are
supplied as physically aligned, detached tensors by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F
from .streaming_model import require

SAMPLE_RATE = 44100
STEM_SUBSET_AUGMENT_PROBABILITY = 0.25
VOCAL_DERANGEMENT_PROBABILITY = 0.125
VOCAL_SOURCE_INDEX = 2
OFF_TARGET_SOURCE_INDICES = (0, 1, 3)
ACTIVITY_POWER_EPSILON = 1e-5


@dataclass(frozen=True)
class StreamingLoss:
    total: Tensor
    raw_l1: Tensor
    projection: Tensor
    projection_contribution: Tensor
    teacher_l1: Tensor
    controlled_deployed_l1: Tensor


def streaming_objective(raw, deployed, targets, vocal_derangement, *, view_codes,
                        teacher_targets=None, teacher_weight=0., deployed_truth_weight=.5):
    """Average all terms over the complete microbatch, including masked rows."""
    import math
    require(raw.ndim == 4 and raw.shape[0] > 0 and raw.shape[1:3] == (4, 2) and raw.shape[-1] > 0
            and all(value.shape == raw.shape and value.dtype == torch.float32 and value.device == raw.device
                    and bool(torch.isfinite(value).all()) for value in (raw, deployed, targets))
            and not targets.requires_grad, "Loss requires aligned finite FP32 source audio")
    require(vocal_derangement.shape == (raw.shape[0],) and vocal_derangement.dtype == torch.bool
            and vocal_derangement.device == raw.device and len(view_codes) == raw.shape[0]
            and all(type(code) is int and code in (0, 1, 2, 3) for code in view_codes), "Invalid source-view assignments")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (teacher_weight, deployed_truth_weight)), "Loss weights must be finite and nonnegative")
    if teacher_weight:
        require(teacher_targets is not None and teacher_targets.shape == raw.shape
                and teacher_targets.dtype == torch.float32 and teacher_targets.device == raw.device
                and not teacher_targets.requires_grad and teacher_targets.grad_fn is None
                and not torch.is_inference(teacher_targets) and bool(torch.isfinite(teacher_targets).all()),
                "Positive distillation weight requires detached, physically aligned teacher targets")
    for index, code in enumerate(view_codes):
        if code < 2:
            excluded = (2,) if code == 0 else (0, 1, 3)
            require(not bool(vocal_derangement[index]) and not bool(torch.count_nonzero(targets[index, list(excluded)])),
                    "Controlled view contains an excluded source or a derangement")
    with torch.autocast(raw.device.type, enabled=False):
        unweighted = F.l1_loss(raw, targets)
        drum = F.l1_loss(raw[:, 0], targets[:, 0])
        raw_l1 = (4 * unweighted + drum) / 5
        projection = _deranged_vocal_projection_loss(estimates=raw, targets=targets,
                                                    vocal_derangement=vocal_derangement)
        projection_weight = torch.minimum(projection.new_tensor(.01), .005 * unweighted.detach()
                                          / projection.detach().clamp_min(1e-8))
        projection_contribution = projection_weight * projection
        weights = raw.new_tensor([2, 1, 1, 1])
        teacher_l1 = raw.new_zeros(())
        if teacher_weight:
            if all(code >= 2 for code in view_codes):
                teacher_l1 = (4 * F.l1_loss(deployed, teacher_targets)
                              + F.l1_loss(deployed[:, 0], teacher_targets[:, 0])) / 5
            else:
                errors = (deployed - teacher_targets).abs().mean(dim=(2, 3))
                per_example = (errors * weights[None]).sum(dim=1) / 5
                keep = raw.new_tensor([code >= 2 for code in view_codes])
                teacher_l1 = (per_example * keep).mean()
        errors = (deployed - targets).abs().mean(dim=(2, 3))
        per_example = (errors * weights[None]).sum(dim=1) / 5
        keep = raw.new_tensor([code < 2 for code in view_codes])
        controlled_l1 = (per_example * keep).mean()
        base = raw_l1 + projection_contribution + teacher_weight * teacher_l1
        total = base if deployed_truth_weight == 0 else base + deployed_truth_weight * controlled_l1
    require(bool(torch.isfinite(total)), "Nonfinite separation loss")
    return StreamingLoss(total, raw_l1, projection, projection_contribution, teacher_l1, controlled_l1)


def _augment_training_distribution(
    *, mixture: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix disjoint stem-subset and all-active vocal-derangement examples."""

    batch_size, source_count = targets.shape[:2]
    if source_count != 4:
        raise ValueError("training augmentation requires four ordered stems")

    augmentation_draw = torch.rand(batch_size, device=targets.device)
    subset_augment = augmentation_draw < STEM_SUBSET_AUGMENT_PROBABILITY
    vocal_derangement = (
        augmentation_draw >= STEM_SUBSET_AUGMENT_PROBABILITY
    ) & (
        augmentation_draw
        < STEM_SUBSET_AUGMENT_PROBABILITY + VOCAL_DERANGEMENT_PROBABILITY
    )
    if batch_size < 2:
        vocal_derangement = torch.zeros_like(vocal_derangement)
    keep_count = torch.randint(
        1,
        source_count,
        (batch_size,),
        device=targets.device,
    )
    random_order = torch.rand(
        batch_size, source_count, device=targets.device
    ).argsort(dim=1)
    ranks = random_order.argsort(dim=1)
    subset_mask = ranks < keep_count[:, None]
    keep_mask = torch.where(
        subset_augment[:, None],
        subset_mask,
        torch.ones_like(subset_mask),
    )

    augmented_targets = targets * keep_mask[:, :, None, None]
    donor_vocals = targets[:, VOCAL_SOURCE_INDEX].roll(shifts=1, dims=0)
    remixed_vocals = torch.where(
        vocal_derangement[:, None, None],
        donor_vocals,
        augmented_targets[:, VOCAL_SOURCE_INDEX],
    )
    augmented_targets[:, VOCAL_SOURCE_INDEX].copy_(remixed_vocals)

    augmented_mixture = augmented_targets.sum(dim=1)
    altered = subset_augment | vocal_derangement
    augmented_mixture = torch.where(
        altered[:, None, None], augmented_mixture, mixture
    )
    return augmented_mixture, augmented_targets, vocal_derangement


def _deranged_vocal_projection_loss(
    *,
    estimates: torch.Tensor,
    targets: torch.Tensor,
    vocal_derangement: torch.Tensor,
) -> torch.Tensor:
    """Discourage accompaniment projection only when vocal identity was shuffled."""

    sample_count = min(SAMPLE_RATE, estimates.shape[-1])
    vocal = estimates[:, VOCAL_SOURCE_INDEX, :, -sample_count:].float()
    desired_vocal = targets[
        :, VOCAL_SOURCE_INDEX, :, -sample_count:
    ].float()
    references = targets[
        :, OFF_TARGET_SOURCE_INDICES, :, -sample_count:
    ].float()

    desired_vocal_energy = desired_vocal.square().sum(dim=(-2, -1))
    reference_energy = references.square().sum(dim=(-2, -1))
    vocal_energy = vocal.square().sum(dim=(-2, -1))
    correlations = (references * vocal[:, None]).sum(dim=(-2, -1))
    squared_cosine = correlations.square() / (
        reference_energy * vocal_energy[:, None]
    ).clamp_min(1e-12)

    channel_sample_count = vocal.shape[-2] * sample_count
    active_references = (
        reference_energy / float(channel_sample_count)
        > ACTIVITY_POWER_EPSILON
    )
    active_desired_vocal = (
        desired_vocal_energy / float(channel_sample_count)
        > ACTIVITY_POWER_EPSILON
    )
    weights = (
        vocal_derangement[:, None]
        & active_desired_vocal[:, None]
        & active_references
    ).to(dtype=squared_cosine.dtype)
    return (squared_cosine * weights).sum() / weights.sum().clamp_min(1.0)

@dataclass(frozen=True)
class AugmentedBatch:
    mixture: Tensor
    targets: Tensor
    vocal_derangement: Tensor
    view_codes: tuple[int, ...]
    original_augmentation: tuple


def augment_training_batch(mixture, targets, *, first_sample_index: int, enabled: bool):
    import torch
    require(type(enabled) is bool and type(first_sample_index) is int and first_sample_index >= 0
            and targets.ndim == 4 and targets.shape[0] == 4 and targets.shape[1:3] == (4, 2)
            and mixture.shape == (4, 2, targets.shape[-1]) and targets.shape[-1] > 0
            and mixture.dtype == targets.dtype == torch.float32 and mixture.device == targets.device
            and not targets.requires_grad and not mixture.requires_grad
            and bool(torch.isfinite(mixture).all()) and bool(torch.isfinite(targets).all()),
            "Require four finite physical FP32 examples and absolute data addresses")
    original = _augment_training_distribution(mixture=mixture, targets=targets)
    if not enabled:
        return AugmentedBatch(*original, (2, 2, 2, 2), original)
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
    return AugmentedBatch(mixed, desired, flags, codes, original)

@dataclass(frozen=True)
class ContextOutput:
    raw: torch.Tensor
    deployed: torch.Tensor
    physical_mixture: torch.Tensor
    warmup_samples: int
    scored_samples: int
    carried_state: bool
    initial_state_detached: bool
    data_hops: int
    flush_hops: int


def render_scored_context(model, mixture, *, warmup_samples, carry_state):
    """Score the same suffix after either detached real history or a reset.

    Both arms receive the same complete augmented crop. The reset control
    ignores its prefix, while the warm arm advances all four native states
    without constructing a gradient graph for that prefix. No intermediate
    flush, mode switch, gain change, or inference architecture change occurs.
    """
    require(type(warmup_samples) is int and warmup_samples > 0 and warmup_samples % 128 == 0
            and type(carry_state) is bool and mixture.ndim == 3 and mixture.shape[0] > 0
            and mixture.shape[1] == 2 and mixture.shape[-1] > warmup_samples
            and mixture.dtype == torch.float32 and model.hop_samples == model.graph_alignment_samples == 128,
            "Invalid scored context request")
    score = mixture[..., warmup_samples:]
    count = score.shape[-1]
    if carry_state:
        with torch.no_grad():
            state = model.render(mixture[..., :warmup_samples]).state.detached()
    else:
        state = model.initial_state(mixture.shape[0], device=mixture.device)
    detached = all(not value.requires_grad and value.grad_fn is None for value in state)
    require(detached, "Warmup retained an autograd graph")
    padding = (-count) % 128
    output = model.render(F.pad(score, (0, padding + 128)), state)
    raw, deployed, physical = (value[..., 128:128 + count] for value in
                               (output.raw, output.deployed, output.delayed_mixture))
    require(raw.shape == deployed.shape == (mixture.shape[0], 4, 2, count)
            and physical.shape == score.shape, "Scored context shape differs")
    return ContextOutput(raw, deployed, physical, warmup_samples, count, carry_state,
                         detached, (count + padding) // 128, 1)
