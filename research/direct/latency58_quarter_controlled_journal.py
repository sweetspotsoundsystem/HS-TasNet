"""Reconstruct controlled deployed supervision for variable view counts."""
from __future__ import annotations

import math
from research.direct.run_latency58_quality import require
from research.direct.latency58_controlled_deployed_loss import VERSION
from research.direct.latency58_variable_counterfactual_journal import loss_evidence as base_evidence
from research.direct.latency58_variable_counterfactual_journal import validate_microbatch as validate_base_microbatch

EXTRA_LOSS_KEYS = ("base_loss", "controlled_deployed_l1", "controlled_deployed_contribution")


def loss_evidence(raw, deployed, targets, teacher_targets, terms, view_codes, weight):
    import torch
    base = base_evidence(raw, deployed, targets, teacher_targets, terms.base, view_codes)
    with torch.no_grad(), torch.autocast(raw.device.type, enabled=False):
        errors = (deployed.detach() - targets).abs().mean(dim=(2, 3))
        return {**base, "base_loss": float(terms.base.total.detach()),
                "controlled_deployed_l1": float(terms.controlled_deployed_l1.detach()),
                "controlled_deployed_contribution": float(terms.controlled_deployed_contribution.detach()),
                "deployed_truth_per_example_stem_l1": errors.cpu().tolist(),
                "controlled_deployed_keep": [code < 2 for code in view_codes],
                "additional_loss_version": VERSION, "additional_loss_weight": weight}


def validate_microbatch(micro, weight):
    require(type(weight) in (int, float) and weight in (0, .5)
            and micro["additional_loss_weight"] == weight
            and micro["additional_loss_version"] == VERSION, "Different deployed truth policy")
    validate_base_microbatch({**micro, "loss": micro["base_loss"]}, "ordinary_only")
    keep = [code < 2 for code in micro["view_codes"]]
    require(micro["controlled_deployed_keep"] == keep
            and all(type(x) is bool for x in micro["controlled_deployed_keep"]), "Wrong controlled truth mask")
    errors = micro["deployed_truth_per_example_stem_l1"]
    require(len(errors) == 4 and all(len(row) == 4 for row in errors)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for row in errors for v in row),
            "Malformed deployed source errors")
    # Independent Python arithmetic: weights [2,1,1,1]/5, complete batch divisor 4.
    expected = sum((2 * row[0] + sum(row[1:])) for row, active in zip(errors, keep) if active) / 20
    for key, value in (("controlled_deployed_l1", expected),
                       ("controlled_deployed_contribution", weight * expected),
                       ("base_loss", micro["supervised_loss"] + .5 * micro["teacher_l1"]),
                       ("loss", micro["base_loss"] + weight * expected)):
        require(math.isfinite(micro[key]) and micro[key] >= 0 and abs(micro[key] - value) < 1e-7,
                "Controlled deployed loss, coefficient or full-batch divisor differs: " + key)
    if weight == 0:
        require(micro["loss"] == micro["base_loss"] and micro["controlled_deployed_contribution"] == 0,
                "Zero-weight objective did not preserve base exactly")
