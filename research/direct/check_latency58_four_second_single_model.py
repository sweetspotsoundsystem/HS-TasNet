"""Check full four-second neural gradients on a retained model, without updates.

The independent complete-group output VJP and canonical implementation use
ordinary microbatches of one. The original absolute, relative and relative-L2
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

def compare_group_gradients(model, mixture_cpu, targets_cpu, *, warmup_samples, progress=None):
    device = next(model.parameters()).device
    fingerprint = state_sha256(model.state_dict())
    auxiliary = source_views(mixture_cpu, targets_cpu)
    inputs = {"ordinary": tuple(v.to(device) for v in (mixture_cpu, targets_cpu)),
              "auxiliary": tuple(v.to(device) for v in auxiliary)}
    groups = prepare_groups(inputs["ordinary"][1][..., warmup_samples:], inputs["auxiliary"][1][..., warmup_samples:])
    outputs, derivatives, losses, ordinary_reference = {}, {}, {}, {}
    for group, microbatch in (("ordinary", 1), ("auxiliary", 1)):
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
        for group, microbatch in (("ordinary", 1), ("auxiliary", 1)):
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
                                         ordinary_microbatch=1, auxiliary_microbatch=1,
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
            "precision": model.training_precision, "ordinary_microbatch": 1, "auxiliary_microbatch": 1,
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

def main():
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.check_latency58_branch_long_context_gpu_b4 import compare_context
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.plan.resolve().is_relative_to(ARTIFACT_ROOT)
            and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require frozen CPU1 qualification with CUDA hidden")
    plan = read(args.plan)
    require(plan["torch_version"] == torch.__version__ and plan["ordinary_microbatch"] == plan["auxiliary_microbatch"] == 1
            and plan["warmup_samples"] == WARMUP_SAMPLES and plan["scored_samples"] == SCORED_SAMPLES
            and plan["new_training_recipe_selected"] is False, "Qualification geometry changed")
    verify_inputs(plan)
    require(not (args.plan.parent / "result.json").exists(), "Preserve completed qualification")
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    before = snapshot()
    model, _ = load_model(plan["fixture_checkpoint"])
    model.train().requires_grad_(True); model.training_precision = "fp32"
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == plan["fixture_model_state_sha256"], "Fixture weights changed")
    rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(plan["synthetic_seed"])
    def progress(phase, group, offset):
        print(json.dumps({"event": "four_second_model_progress", "phase": phase, "group": group,
            "offset": offset, "elapsed_seconds": time.monotonic() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}), flush=True)
    audio = .03 * torch.randn(1, 2, CROP_SAMPLES, generator=generator)
    progress("context_comparison_start", "ordinary", 0)
    context = compare_context(model, audio, WARMUP_SAMPLES)
    del audio
    gc.collect()
    progress("context_comparison_pass", "ordinary", 0)
    truth = .02 * torch.randn(16, 4, 2, CROP_SAMPLES, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    auxiliary_mix, auxiliary_truth = source_views(truth.sum(1), truth)
    auxiliary_contexts = []
    for index, name in enumerate(("instrumental", "vocals_only")):
        progress("context_comparison_start", name, index)
        auxiliary_contexts.append(compare_context(model, auxiliary_mix[index:index + 1], WARMUP_SAMPLES))
        gc.collect()
        progress("context_comparison_pass", name, index)
    del auxiliary_mix, auxiliary_truth
    gradients = compare_group_gradients(model, truth.sum(1), truth,
                                        warmup_samples=WARMUP_SAMPLES, progress=progress)
    del truth
    gc.collect()
    require(state_sha256(model.state_dict()) == fingerprint and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized() and all(p.grad is None for p in model.parameters()),
            "Qualification changed weights or RNG, left gradients or initialized CUDA")
    verify_inputs(plan)
    result = {"status": "pass", "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
        "fixture_model_state_sha256": fingerprint, "context_comparison": context,
        "auxiliary_context_comparisons": auxiliary_contexts,
        "whole_group_neural_gradient_comparison": gradients,
        "ordinary_microbatch": 1, "auxiliary_microbatch": 1, "logical_batch_size": 16,
        "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
        "gpu_used": False, "optimizer_updates": 0, "model_weights_unchanged": True,
        "rng_unchanged": True, "quality_measured": False, "new_training_recipe_selected": False,
        "elapsed_seconds": time.monotonic() - began,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_before": before, "budget_after": snapshot(),
        "limits": ["Synthetic CPU FP32 fixture on retained 012 EMA; no quality measurement.",
                   "Full training update/restart, CUDA BF16 and GPU resource qualification remain separate requirements."]}
    write(args.plan.parent / "result.json", result)
    print(json.dumps({"status": "pass", "elapsed_seconds": result["elapsed_seconds"],
        "peak_rss_bytes": result["peak_rss_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
