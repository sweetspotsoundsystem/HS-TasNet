"""Measure the existing grouped objective without changing model weights.

Auxiliary contributions are VJPs of the joint two-view objective. They are not
gradients of separately normalized one-view losses. No optimizer is created.
"""
from __future__ import annotations

import math

import torch

from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_grouped_vocal_canonical import accumulate_groups
from research.direct.latency58_grouped_vocal_auxiliary import AUXILIARY_WEIGHT


VERSION = "latency58-fixed-training-group-gradient-probe-v1"
GROUPS = ("ordinary", "instrumental", "vocals_only", "auxiliary")
PAIRS = (("ordinary", "auxiliary"), ("ordinary", "instrumental"),
         ("ordinary", "vocals_only"), ("instrumental", "vocals_only"))


def policy():
    return {"version": VERSION, "device": "cpu", "precision": "fp32",
            "ordinary_microbatch": 1, "auxiliary_microbatch": 1,
            "ordinary_examples": 16, "auxiliary_examples": 2,
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "auxiliary_contributions": "Partition the existing joint two-view output VJP by example",
            "loss_denominators": "Unchanged complete ordinary16 and complete auxiliary2 groups",
            "warmup": "Fresh detached full-context warmup for each input view",
            "updates": "No clipping, optimizer, EMA update or checkpoint write",
            "gradient_storage": "FP32 tensors in memory only; scalar statistics and hashes may be saved",
            "interpretation": "Local FP32 gradient geometry before clipping and Adam preconditioning"}


def _snapshot(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                "Missing or nonfinite probe gradient: " + name)
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def collect(model, mixture, targets, *, warmup_samples, progress=None):
    """Return four gradient maps, leaving the model and inputs unchanged.

    Observe and clear each auxiliary example only after its complete VJP.
    Its output derivatives still come from the joint two-example objective.
    Fresh gradient buffers avoid subtracting nearly equal accumulated gradients.
    """
    parameters = list(model.parameters())
    require(len(parameters) == 40 and model.training and model.training_precision == "fp32"
            and torch.is_grad_enabled()
            and all(p.device.type == "cpu" and p.dtype == torch.float32 and p.requires_grad
                    and p.grad is None for p in parameters),
            "Require a fresh CPU FP32 training model with all 40 gradients clear")
    require(mixture.ndim == 3 and targets.ndim == 4
            and mixture.device.type == targets.device.type == "cpu"
            and mixture.dtype == targets.dtype == torch.float32
            and mixture.shape == (16, 2, targets.shape[-1])
            and targets.shape[:3] == (16, 4, 2)
            and not mixture.requires_grad and not targets.requires_grad
            and bool(torch.isfinite(mixture).all()) and bool(torch.isfinite(targets).all())
            and type(warmup_samples) is int and warmup_samples > 0 and warmup_samples % 128 == 0
            and mixture.shape[-1] - warmup_samples >= 44100,
            "Require complete fixed FP32 ordinary inputs and valid scored context")
    before = state_sha256(model.state_dict())
    input_before = state_sha256({"mixture": mixture, "targets": targets})
    rng_before = torch.get_rng_state().clone()
    gradients, observed = {}, []

    def observe(phase, group, offset):
        if group == "auxiliary" and phase == "canonical_backward":
            require(offset in (0, 1) and offset == len(observed), "Auxiliary VJP order changed")
            name = ("instrumental", "vocals_only")[offset]
            gradients[name] = _snapshot(model)
            observed.append(offset)
            model.zero_grad(set_to_none=True)
        if progress is not None:
            progress(phase, group, offset)

    def after_group(group, row):
        if group == "ordinary":
            require(not gradients and row["examples"] == 16, "Ordinary group boundary changed")
            gradients[group] = _snapshot(model)
            model.zero_grad(set_to_none=True)
        else:
            require(observed == [0, 1] and all(p.grad is None for p in parameters),
                    "Both auxiliary contributions were not captured independently")

    try:
        rows = accumulate_groups(model, mixture, targets, warmup_samples=warmup_samples,
                                 ordinary_microbatch=1, auxiliary_microbatch=1,
                                 after_group=after_group, progress=observe)
        gradients["auxiliary"] = {
            name: gradients["instrumental"][name] + gradients["vocals_only"][name]
            for name, _ in model.named_parameters()}
        require(state_sha256(model.state_dict()) == before
                and state_sha256({"mixture": mixture, "targets": targets}) == input_before
                and torch.equal(rng_before, torch.get_rng_state()), "Probe changed model, inputs or RNG")
        return gradients, {"policy": policy(), "model_state_sha256": before,
                           "input_sha256": input_before, "groups": rows,
                           "weights_inputs_and_rng_unchanged": True,
                           "auxiliary_contributions_from_joint_objective": True,
                           "optimizer_updates": 0}
    finally:
        model.zero_grad(set_to_none=True)


def _pair(left_squared, right_squared, dot):
    left, right = math.sqrt(left_squared), math.sqrt(right_squared)
    cosine = dot / (left * right) if left and right else None
    require(cosine is None or -1 - 1e-12 <= cosine <= 1 + 1e-12, "Invalid gradient cosine")
    return {"left_l2": left, "right_l2": right, "dot": dot,
            "cosine": None if cosine is None else max(-1., min(1., cosine)),
            "right_to_left_l2_ratio": right / left if left else None,
            "opposed": dot < 0}


def summarize(gradients):
    """Use FP64 reductions without concatenating full model-sized vectors."""
    require(set(gradients) == set(GROUPS) and bool(gradients["ordinary"]), "Incomplete gradient groups")
    names = list(gradients["ordinary"])
    require(all(list(gradients[group]) == names for group in GROUPS), "Gradient parameter orders differ")
    rows = {}
    for name in names:
        tensors = {group: gradients[group][name] for group in GROUPS}
        require(all(t.device.type == "cpu" and t.dtype == torch.float32 and not t.requires_grad
                    and t.shape == tensors["ordinary"].shape and bool(torch.isfinite(t).all())
                    for t in tensors.values()), "Malformed gradient tensor: " + name)
        require(torch.equal(tensors["auxiliary"], tensors["instrumental"] + tensors["vocals_only"]),
                "Auxiliary contribution sum differs: " + name)
        values = {group: tensor.double() for group, tensor in tensors.items()}
        norms = {group: float(value.square().sum()) for group, value in values.items()}
        dots = {left + "_vs_" + right: float((values[left] * values[right]).sum())
                for left, right in PAIRS}
        count = tensors["ordinary"].numel()
        rows[name] = {"shape": list(tensors["ordinary"].shape), "elements": count,
                      "fp32_parameter_bytes": count * 4, "two_fp32_adam_moment_bytes": count * 8,
                      "squared_l2": norms, "dots": dots,
                      "pairs": {left + "_vs_" + right: _pair(norms[left], norms[right],
                                                             dots[left + "_vs_" + right])
                                for left, right in PAIRS}}
    norms = {group: math.fsum(row["squared_l2"][group] for row in rows.values()) for group in GROUPS}
    dots = {left + "_vs_" + right: math.fsum(row["dots"][left + "_vs_" + right] for row in rows.values())
            for left, right in PAIRS}
    pairs = {left + "_vs_" + right: _pair(norms[left], norms[right], dots[left + "_vs_" + right])
             for left, right in PAIRS}
    return {"parameter_tensors": len(rows), "parameter_elements": sum(row["elements"] for row in rows.values()),
            "group_l2": {group: math.sqrt(value) for group, value in norms.items()},
            "pairs": pairs, "per_parameter": rows,
            "ordinary_directional_derivative_along_negative_combined_gradient":
                -(norms["ordinary"] + dots["ordinary_vs_auxiliary"]),
            "auxiliary_directional_derivative_along_negative_combined_gradient":
                -(norms["auxiliary"] + dots["ordinary_vs_auxiliary"]),
            "opposed_parameter_tensors": {key: sum(row["pairs"][key]["opposed"] for row in rows.values())
                                           for key in pairs},
            "gradient_sha256": {group: state_sha256(gradients[group]) for group in GROUPS},
            "limitations": ["Auxiliary vectors already include the fixed 0.1 weight and joint group normalization.",
                            "Directional derivatives describe infinitesimal plain gradient descent before clipping; not Adam steps.",
                            "FP32 CPU geometry does not replay the BF16 CUDA training trajectory.",
                            "Fixed training examples do not establish validation quality or a causal account of its changes."]}
