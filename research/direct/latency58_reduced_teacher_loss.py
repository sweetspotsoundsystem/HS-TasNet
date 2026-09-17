"""Reduce ordinary-mixture distillation while retaining the cleanup objective."""
from __future__ import annotations

from dataclasses import replace
import math

from research.direct.run_latency58_quality import require
from research.direct.latency58_controlled_deployed_loss import controlled_deployed_objective

VERSION = "latency58-ordinary-teacher-quarter-v1"


def objective(raw, deployed, targets, teacher_targets, vocal_derangement, *, view_codes,
              teacher_weight=.25, weight=.5):
    """Weight .5 is an exact control; .25 changes only ordinary distillation.

    Source losses, controlled-view participation, stem and batch divisors, and
    the original projection cap are unchanged. There is no inference change.
    """
    import torch

    require(type(teacher_weight) in (int, float) and teacher_weight in (.25, .5),
            "Require teacher coefficient .25 or its .5 control")
    original = controlled_deployed_objective(raw, deployed, targets, teacher_targets, vocal_derangement,
                                            view_codes=view_codes, weight=weight)
    if teacher_weight == .5:
        return original
    with torch.autocast(raw.device.type, enabled=False):
        base = replace(original.base, total=original.base.waveform_l1
                       + original.base.projection_contribution + teacher_weight * original.base.teacher_l1)
        total = base.total if weight == 0 else base.total + original.controlled_deployed_contribution
    require(bool(torch.isfinite(total)), "Reduced-teacher objective is nonfinite")
    return replace(original, base=base, total=total)


def loss_evidence(raw, deployed, targets, teacher_targets, terms, view_codes, weight, teacher_weight):
    from research.direct.latency58_controlled_deployed_journal import loss_evidence as original_evidence
    return {**original_evidence(raw, deployed, targets, teacher_targets, terms, view_codes, weight),
            "teacher_weight": teacher_weight, "reduced_teacher_version": VERSION}


def validate_microbatch(micro, teacher_weight, weight):
    """Reconstruct the new total independently from detached per-source errors."""
    from research.direct.latency58_counterfactual_journal import validate_microbatch as validate_teacher
    from research.direct.latency58_controlled_deployed_loss import VERSION as CONTROLLED_VERSION

    require(type(teacher_weight) in (int, float) and teacher_weight in (.25, .5)
            and micro["teacher_weight"] == teacher_weight and micro["reduced_teacher_version"] == VERSION
            and type(weight) in (int, float) and weight in (0, .5)
            and micro["additional_loss_weight"] == weight
            and micro["additional_loss_version"] == CONTROLLED_VERSION,
            "Different reduced-teacher recipe")
    validate_teacher(micro, "ordinary_only")
    keep = [code < 2 for code in micro["view_codes"]]
    require(micro["controlled_deployed_keep"] == keep
            and all(type(v) is bool for v in micro["controlled_deployed_keep"]), "Wrong source-truth mask")
    errors = micro["deployed_truth_per_example_stem_l1"]
    require(len(errors) == 4 and all(len(row) == 4 for row in errors)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for row in errors for v in row),
            "Malformed deployed source errors")
    controlled = sum(2 * row[0] + sum(row[1:]) for row, active in zip(errors, keep) if active) / 20
    base = micro["supervised_loss"] + teacher_weight * micro["teacher_l1"]
    for key, expected in (("controlled_deployed_l1", controlled),
                          ("controlled_deployed_contribution", weight * controlled),
                          ("base_loss", base), ("loss", base + weight * controlled)):
        require(math.isfinite(micro[key]) and micro[key] >= 0 and abs(micro[key] - expected) < 1e-7,
                "Reduced-teacher arithmetic differs: " + key)
