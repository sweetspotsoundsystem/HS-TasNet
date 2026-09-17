"""Qualify four-second output losses without loading or changing a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from research.direct.latency58_branch_sdr_blend import objective as whole_objective
from research.direct.latency58_logical_batch_loss import objective as contribution, prepare_reduction
from research.direct.latency58_weighted_vocal_auxiliary import objective as weighted_objective
from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_m4_followup_budget import snapshot

SCORED = 176512


def gradient_errors(actual, expected):
    require(all(torch.isfinite(a).all() and torch.isfinite(b).all()
                and torch.allclose(a, b, atol=2e-9, rtol=3e-5)
                for a, b in zip(actual, expected, strict=True)),
            "Four-second output-coordinate gradients changed")
    return [float((a - b).abs().max()) for a, b in zip(actual, expected, strict=True)]


def ordinary_case(generator, case, batch):
    target = .03 * torch.randn(batch, 4, 2, SCORED, generator=generator)
    target[:batch // 2, 1] = 0
    target[batch // 2:, 2] = 0
    target[0, 3, :, :44100] = .0001
    if case == "globally_absent_stem":
        target[:, 0] = 0
    if case == "all_silence":
        target.zero_()
    mixture = target.sum(1)
    noise = .008 * torch.randn(target.shape, generator=generator)
    noise[batch // 2:] *= 3
    if case == "all_silence":
        noise.zero_()
    raw, deployed = (target + noise).requires_grad_(), (target + .7 * noise).requires_grad_()
    reference = whole_objective(raw, deployed, target, mixture)
    expected = torch.autograd.grad(reference.total, (raw, deployed))
    reduction = prepare_reduction(target)
    actual = tuple(torch.zeros_like(value) for value in expected)
    term_sums = {}
    naive = 0.
    names = ("total", "waveform", "spectral", "raw_anchor", "negative_sdr_db",
             "reconstruction_loss", "direct_sdr_loss", "absence_db", "direct_raw_anchor",
             "per_stem_negative_sdr_db")
    active, absent = torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
    for offset in range(0, batch, 2):
        stop = offset + 2
        a, b = raw[offset:stop].detach().requires_grad_(), deployed[offset:stop].detach().requires_grad_()
        part = contribution(a, b, target[offset:stop], mixture[offset:stop], reduction)
        grads = torch.autograd.grad(part.total, (a, b))
        for destination, value in zip(actual, grads, strict=True):
            destination[offset:stop].copy_(value)
        for name in names:
            term_sums[name] = term_sums.get(name, 0) + getattr(part, name).detach()
        active += part.active_window_counts
        absent += part.absent_window_counts
        with torch.no_grad():
            naive += float(whole_objective(a, b, target[offset:stop], mixture[offset:stop]).total) * 2 / batch
        del part, grads, a, b
    errors = {name: float((value - getattr(reference, name)).detach().abs().max())
              for name, value in term_sums.items()}
    require(all(torch.allclose(value, getattr(reference, name), atol=3e-6, rtol=3e-6)
                for name, value in term_sums.items())
            and torch.equal(active, reduction.active) and torch.equal(absent, reduction.absent),
            "Four-second microbatch loss or eligible-window counts changed")
    derivatives = gradient_errors(actual, expected)
    naive_error = naive - float(reference.total.detach())
    if case == "heterogeneous_b16":
        require(abs(naive_error) > 1e-3, "The fixture must reject naive microbatch averaging")
    if case == "all_silence":
        require(float(reference.total.detach()) == 0 and all(torch.count_nonzero(g) == 0 for g in actual),
                "Exact silence acquired a loss or gradient")
    return {"case": case, "batch_size": batch, "microbatch_size": 2,
        "scored_samples": SCORED, "complete_one_second_windows_per_example": SCORED // 44100,
        "active_windows": active.tolist(), "absent_windows": absent.tolist(),
        "term_max_abs_errors": errors, "raw_and_deployed_gradient_max_abs_errors": derivatives,
        "incorrect_naive_mean_loss_difference": naive_error}


def auxiliary_case(generator):
    target = .03 * torch.randn(2, 4, 2, SCORED, generator=generator)
    target[0, 2] = 0
    target[1, [0, 1, 3]] = 0
    target[0, 1, :, :2 * 44100] = 0
    target[1, 2, :, :44100] = .0001
    mixture = target.sum(1)
    noise = .008 * torch.randn(target.shape, generator=generator)
    raw, deployed = (target + noise).requires_grad_(), (target + .7 * noise).requires_grad_()
    reference = whole_objective(raw, deployed, target, mixture)
    expected = torch.autograd.grad(reference.total, (raw, deployed))
    rows = []
    for weights in ((1., 1.), (1., .25)):
        result = weighted_objective(raw, deployed, target, mixture, weights=weights)
        actual = torch.autograd.grad(result.total, (raw, deployed))
        scaled = tuple(value * value.new_tensor(weights)[:, None, None, None] for value in expected)
        errors = gradient_errors(actual, scaled)
        if weights == (1., 1.):
            require(torch.allclose(result.total, reference.total, atol=3e-6, rtol=3e-6),
                    "Unit view weights changed the complete-group scalar")
        rows.append({"weights": list(weights), "raw_and_deployed_gradient_max_abs_errors": errors,
            "active_windows": result.active_window_counts.tolist(),
            "absent_windows": result.absent_window_counts.tolist(),
            "whole_group_output_derivatives_scaled_by_view": True})
    return {"case": "ordered_source_views_four_seconds", "scored_samples": SCORED, "controls": rows,
        "reference": "Independent unchanged complete-group objective, then scale each view's output derivative"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.plan.resolve().is_relative_to(PHASE)
            and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require a frozen CPU1 four-second qualification")
    plan = read(args.plan)
    require(plan["scored_samples"] == SCORED and plan["torch_version"] == torch.__version__
            and plan["new_training_recipe_selected"] is False, "Qualification scope changed")
    verify_inputs(plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(20261102)
    before = snapshot()
    cases = []
    for case, batch in (("heterogeneous_b16", 16), ("globally_absent_stem", 4), ("all_silence", 4)):
        row = ordinary_case(generator, case, batch)
        cases.append(row)
        print(json.dumps({"event": "case_pass", **row}), flush=True)
    auxiliary = auxiliary_case(generator)
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "CPU qualification changed global RNG or initialized CUDA")
    verify_inputs(plan)
    result = {"status": "pass", "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
        "cases": cases, "source_view_derivatives": auxiliary, "gpu_used": False,
        "model_loaded": False, "optimizer_updates": 0, "quality_measured": False,
        "new_training_recipe_selected": False, "elapsed_seconds": time.monotonic() - began,
        "budget_before": before, "budget_after": snapshot(),
        "limits": ["Output-coordinate CPU qualification only; no neural-gradient, data-pipeline or GPU resource qualification.",
                   "Four-second training geometry is a prospective option, not a selected next experiment."]}
    write(args.plan.parent / "result.json", result)
    print(json.dumps({"status": "pass", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
