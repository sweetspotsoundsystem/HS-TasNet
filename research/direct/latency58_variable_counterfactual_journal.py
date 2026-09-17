"""Validate ordinary-only distillation with recorded variable controlled counts."""
from __future__ import annotations

import math

from research.direct.run_latency58_quality import require
from research.direct.latency58_counterfactual_teacher import VERSION

UNIFORM_KEYS = ("uniform_loss", "uniform_teacher_l1", "uniform_unweighted_teacher_l1",
                "uniform_teacher_drum_l1")


def loss_evidence(raw, deployed, targets, teacher_targets, terms, view_codes):
    """Read detached loss diagnostics without changing the optimization graph."""
    import torch
    with torch.no_grad(), torch.autocast(raw.device.type, enabled=False):
        errors = (deployed.detach() - teacher_targets).abs().mean(dim=(2, 3))
        keep = [terms.mode == "all_views" or code >= 2 for code in view_codes]
        mask = errors.new_tensor(keep)
        uniform_unweighted = torch.nn.functional.l1_loss(deployed.detach(), teacher_targets)
        uniform_drum = torch.nn.functional.l1_loss(deployed.detach()[:, 0], teacher_targets[:, 0])
        unweighted = uniform_unweighted if terms.mode == "all_views" else (errors.mean(dim=1) * mask).mean()
        drum = uniform_drum if terms.mode == "all_views" else (errors[:, 0] * mask).mean()
        return {
            "unweighted_waveform_l1": float(torch.nn.functional.l1_loss(raw.detach(), targets)),
            "raw_drum_l1": float(torch.nn.functional.l1_loss(raw.detach()[:, 0], targets[:, 0])),
            "unweighted_teacher_l1": float(unweighted), "teacher_drum_l1": float(drum),
            "uniform_teacher_l1": float(terms.uniform_teacher_l1.detach()),
            "uniform_unweighted_teacher_l1": float(uniform_unweighted),
            "uniform_teacher_drum_l1": float(uniform_drum),
            "uniform_loss": float((terms.waveform_l1 + terms.projection_contribution
                                   + .5 * terms.uniform_teacher_l1).detach()),
            "teacher_per_example_stem_l1": errors.cpu().tolist(), "teacher_keep": keep,
            "controlled_examples": terms.controlled_examples, "ordinary_examples": terms.ordinary_examples,
            "teacher_mode": terms.mode, "counterfactual_version": VERSION,
        }


def validate_microbatch(micro, mode):
    require(mode in ("all_views", "ordinary_only") and micro["teacher_mode"] == mode
            and micro["counterfactual_version"] == VERSION, "Different teacher policy in the journal")
    codes = micro["view_codes"]
    require(len(codes) == 4 and all(type(code) is int and code in (0, 1, 2, 3) for code in codes)
            and type(micro["controlled_examples"]) is int and type(micro["ordinary_examples"]) is int
            and micro["controlled_examples"] == sum(code < 2 for code in codes)
            and micro["ordinary_examples"] == sum(code >= 2 for code in codes),
            "Invalid view codes or inconsistent controlled-view counts")
    keep = [mode == "all_views" or code >= 2 for code in codes]
    require(micro["teacher_keep"] == keep and all(type(x) is bool for x in micro["teacher_keep"]),
            "Teacher participation differs from the recorded source views")
    errors = micro["teacher_per_example_stem_l1"]
    require(len(errors) == 4 and all(len(row) == 4 for row in errors)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for row in errors for v in row),
            "Malformed per-example teacher errors")
    # Reconstruct using Python doubles and the original full B=4 divisor.
    # FP32 reduction order may differ; a kept-example divisor must still fail.
    unweighted = sum(sum(row) for row, include in zip(errors, keep) if include) / 16
    drum = sum(row[0] for row, include in zip(errors, keep) if include) / 4
    uniform_unweighted = sum(map(sum, errors)) / 16
    uniform_drum = sum(row[0] for row in errors) / 4
    expected = {"unweighted_teacher_l1": unweighted, "teacher_drum_l1": drum,
                "teacher_l1": (4 * unweighted + drum) / 5,
                "uniform_unweighted_teacher_l1": uniform_unweighted,
                "uniform_teacher_drum_l1": uniform_drum,
                "uniform_teacher_l1": (4 * uniform_unweighted + uniform_drum) / 5}
    for key, value in expected.items():
        require(math.isfinite(micro[key]) and micro[key] >= 0
                and abs(micro[key] - value) < 1e-7, "Recorded teacher loss or full-batch divisor differs: " + key)
    require(abs(micro["uniform_loss"] - micro["supervised_loss"] - .5 * micro["uniform_teacher_l1"]) < 1e-7
            and micro["teacher_l1"] <= micro["uniform_teacher_l1"] + 1e-7,
            "Uniform counterfactual loss or selected teacher coefficient differs")
    if mode == "all_views":
        require(micro["loss"] == micro["uniform_loss"]
                and micro["teacher_l1"] == micro["uniform_teacher_l1"]
                and micro["unweighted_teacher_l1"] == micro["uniform_unweighted_teacher_l1"]
                and micro["teacher_drum_l1"] == micro["uniform_teacher_drum_l1"],
                "Existing-loss control changed")


def compare_control_prefix(rows, resource):
    """Require all original journal fields and first-update gradients to replay."""
    reference = resource["matching_production_updates"]
    require(len(reference) == 2 and len(rows) >= 2, "Need the complete two-update control replay")
    for actual, expected in zip(rows[:2], reference, strict=True):
        for key in set(expected) - {"elapsed_seconds", "peak_vram_gib", "stage_start_step", "microbatches"}:
            require(actual[key] == expected[key], "Original control update differs: " + key)
        for micro, old in zip(actual["microbatches"], expected["microbatches"], strict=True):
            require(all(micro[key] == value for key, value in old.items()), "Original control microbatch differs")


def rng_state_sha256():
    import hashlib
    import pickle
    import random
    import numpy as np
    import torch
    values = {"python": pickle.dumps(random.getstate()), "numpy": pickle.dumps(np.random.get_state()),
              "torch_cpu": torch.get_rng_state().numpy().tobytes(),
              "torch_cuda": torch.cuda.get_rng_state().cpu().numpy().tobytes()}
    return {key: hashlib.sha256(value).hexdigest() for key, value in values.items()}
