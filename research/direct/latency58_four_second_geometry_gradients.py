"""Check full four-second neural gradients on a retained model, without updates.

The independent complete-group output VJP and canonical implementation use
the explicitly requested ordinary and auxiliary microbatch sizes. The original absolute, relative and relative-L2
tolerances remain unchanged. No recipe or model quality is selected here.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import time

import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, AUXILIARY_WEIGHT
from research.direct.latency58_branch_sdr_blend import objective as whole_objective
from research.direct.latency58_weighted_vocal_auxiliary import objective as weighted_objective
from research.direct.latency58_weighted_vocal_canonical import accumulate_groups, policy as accumulation_policy
from research.direct.latency58_four_second_data import CROP_SAMPLES, WARMUP_SAMPLES, SCORED_SAMPLES
from research.direct.latency58_four_second_storage import snapshot, ARTIFACT_ROOT

def compare_group_gradients(model, mixture_cpu, targets_cpu, *, warmup_samples, ordinary_microbatch, auxiliary_microbatch, progress=None):
    device = next(model.parameters()).device
    fingerprint = state_sha256(model.state_dict())
    auxiliary = source_views(mixture_cpu, targets_cpu)
    inputs = {"ordinary": tuple(v.to(device) for v in (mixture_cpu, targets_cpu)),
              "auxiliary": tuple(v.to(device) for v in auxiliary)}
    groups = prepare_groups(inputs["ordinary"][1][..., warmup_samples:], inputs["auxiliary"][1][..., warmup_samples:])
    outputs, derivatives, losses, ordinary_reference = {}, {}, {}, {}
    for group, microbatch in (("ordinary", ordinary_microbatch), ("auxiliary", auxiliary_microbatch)):
        audio, truth = inputs[group]
        raw, deployed = [], []
        for offset in range(0, len(audio), microbatch):
            result = render_scored_context(model, audio[offset:offset + microbatch], warmup_samples=warmup_samples, carry_state=True)
            raw.append(result.raw.detach().clone()); deployed.append(result.deployed.detach().clone())
            del result
            if progress is not None:
                progress("reference_capture", group, offset)
        raw, deployed = torch.cat(raw).requires_grad_(), torch.cat(deployed).requires_grad_()
        objective = whole_objective if group == "ordinary" else weighted_objective
        value = objective(raw, deployed, truth[..., warmup_samples:], audio[..., warmup_samples:]).total
        if group == "auxiliary":
            value = value * AUXILIARY_WEIGHT
        losses[group] = float(value.detach())
        derivatives[group] = tuple(v.detach() for v in torch.autograd.grad(value, (raw, deployed)))
        outputs[group] = raw.detach(), deployed.detach()
        del raw, deployed, value
    try:
        model.zero_grad(set_to_none=True)
        for group, microbatch in (("ordinary", ordinary_microbatch), ("auxiliary", auxiliary_microbatch)):
            audio, truth = inputs[group]
            for offset in range(0, len(audio), microbatch):
                end = offset + microbatch
                physical = audio[offset:end].detach().clone().requires_grad_()
                result = render_scored_context(model, physical, warmup_samples=warmup_samples, carry_state=True)
                require(torch.equal(result.raw, outputs[group][0][offset:end]) and torch.equal(result.deployed, outputs[group][1][offset:end]),
                        "Reference output replay changed")
                torch.autograd.backward((result.raw, result.deployed), tuple(v[offset:end] for v in derivatives[group]))
                require(physical.grad is not None and bool(torch.isfinite(physical.grad).all())
                        and torch.count_nonzero(physical.grad[..., :warmup_samples]) == 0
                        and torch.count_nonzero(physical.grad[..., warmup_samples:]) > 0, "Reference warmup gradients differ")
                del result, physical
                if progress is not None:
                    progress("independent_whole_group_vjp", group, offset)
            if group == "ordinary":
                ordinary_reference.update({name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()})
        expected = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
        del outputs, derivatives, inputs, groups
        model.zero_grad(set_to_none=True)

        def verify_ordinary(group, row):
            if group == "ordinary":
                require(len(ordinary_reference) == 40 and all(
                    torch.equal(p.grad.detach().cpu(), ordinary_reference[name])
                    for name, p in model.named_parameters()), "Ordinary gradients changed")
                ordinary_reference.clear()

        actual_groups = accumulate_groups(model, mixture_cpu, targets_cpu, warmup_samples=warmup_samples,
                                         ordinary_microbatch=ordinary_microbatch, auxiliary_microbatch=auxiliary_microbatch,
                                         verify_input_gradients=True, progress=progress, after_group=verify_ordinary)
        errors = {}
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing finite canonical gradient: " + name)
            actual, reference = parameter.grad.detach().cpu(), expected[name]
            require(float(actual.norm()) > 0 and float(reference.norm()) > 0, "Unexercised canonical gradient: " + name)
            relative_l2 = float((actual - reference).double().norm() / reference.double().norm())
            errors[name] = {"maximum_absolute_error": float((actual - reference).abs().max()),
                           "relative_l2_error": relative_l2, "reference_norm": float(reference.norm()),
                           "bitwise_equal": torch.equal(actual, reference)}
            require(torch.allclose(actual, reference, atol=1e-7, rtol=1e-4) and relative_l2 < 5e-5,
                    "Canonical parameter gradient differs from whole-group VJP: " + name)
        require(len(errors) == 40 and state_sha256(model.state_dict()) == fingerprint, "Canonical parent or parameter inventory changed")
        for group in actual_groups:
            require(abs(actual_groups[group]["weighted_loss"] - losses[group]) < 3e-6
                    and actual_groups[group]["replay_outputs_bit_exact"], "Canonical whole-group loss differs")
        return {"status": "pass", "model_state_sha256": fingerprint, "device": str(device),
            "precision": model.training_precision, "ordinary_microbatch": ordinary_microbatch, "auxiliary_microbatch": auxiliary_microbatch,
            "warmup_samples": warmup_samples, "scored_samples": mixture_cpu.shape[-1] - warmup_samples,
            "all_40_gradients": errors, "reference_losses": losses,
            "actual_losses": {g: row["weighted_loss"] for g, row in actual_groups.items()},
            "activity": {g: {"active": row["active_windows"], "absent": row["absent_windows"]} for g, row in actual_groups.items()},
            "absolute_gradient_tolerance": 1e-7, "relative_gradient_tolerance": 1e-4,
            "relative_l2_tolerance": 5e-5, "loss_absolute_tolerance": 3e-6,
            "warmup_input_gradients_zero": True, "weights_unchanged": True, "optimizer_updates": 0,
            "ordinary_all_40_gradients_bit_exact_against_unmodified_reference": True,
            "accumulation_policy": accumulation_policy(), "canonical_replay_outputs_bit_exact": True}
    finally:
        model.zero_grad(set_to_none=True)
