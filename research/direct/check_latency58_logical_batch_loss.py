"""Compare microbatch values and gradients with the unchanged full-batch loss."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from research.direct.latency58_branch_sdr_blend import objective as full_objective
from research.direct.latency58_logical_batch_loss import objective, prepare_reduction, policy
from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def check():
    generator = torch.Generator().manual_seed(20261029)
    rows = []
    for case, batch, samples in (("heterogeneous_b16_two_seconds", 16, 88320),
                                 ("globally_absent_stem", 4, 88320),
                                 ("one_second_compatibility", 4, 44160),
                                 ("all_silence", 4, 88320)):
        target = .03 * torch.randn(batch, 4, 2, samples, generator=generator)
        half = batch // 2
        target[:half, 1] = 0
        target[half:, 2] = 0
        target[0, 3, :, :44100] = .0001
        if case == "globally_absent_stem":
            target[:, 0] = 0
        if case == "all_silence":
            target.zero_()
        mixture = target.sum(1)
        noise = .008 * torch.randn(target.shape, generator=generator)
        noise[half:] *= 3
        if case == "all_silence":
            noise.zero_()
        raw = (target + noise).requires_grad_()
        deployed = (target + .7 * noise).requires_grad_()
        reference = full_objective(raw, deployed, target, mixture)
        reference_gradients = torch.autograd.grad(reference.total, (raw, deployed))
        reduction = prepare_reduction(target)
        raw_micro = raw.detach().clone().requires_grad_()
        deployed_micro = deployed.detach().clone().requires_grad_()
        pieces = [objective(raw_micro[i:i + half], deployed_micro[i:i + half], target[i:i + half],
                            mixture[i:i + half], reduction) for i in (0, half)]
        summed = sum(part.total for part in pieces)
        gradients = torch.autograd.grad(summed, (raw_micro, deployed_micro))
        names = ("total", "waveform", "spectral", "raw_anchor", "negative_sdr_db",
                 "reconstruction_loss", "direct_sdr_loss", "absence_db", "direct_raw_anchor",
                 "per_stem_negative_sdr_db")
        term_errors = {name: float((sum(getattr(p, name) for p in pieces) - getattr(reference, name)).detach().abs().max())
                       for name in names}
        gradient_errors = [float((actual - expected).abs().max())
                           for actual, expected in zip(gradients, reference_gradients, strict=True)]
        require(all(torch.allclose(sum(getattr(p, name) for p in pieces), getattr(reference, name),
                                   atol=3e-6, rtol=3e-6) for name in names)
                and all(torch.allclose(actual, expected, atol=2e-9, rtol=3e-5)
                        for actual, expected in zip(gradients, reference_gradients, strict=True))
                and torch.equal(sum(p.active_window_counts for p in pieces), reduction.active)
                and torch.equal(sum(p.absent_window_counts for p in pieces), reduction.absent),
                "Microbatch objective or its audio gradients differ from the original full batch")
        with torch.no_grad():
            naive = sum(full_objective(raw[i:i + half], deployed[i:i + half], target[i:i + half],
                                       mixture[i:i + half]).total for i in (0, half)) / 2
        naive_error = float((naive - reference.total).detach())
        if case == "heterogeneous_b16_two_seconds":
            require(abs(naive_error) > 1e-3, "Fixture must expose the incorrect naive microbatch mean")
        if case == "all_silence":
            require(float(summed.detach()) == 0 and all(torch.count_nonzero(g) == 0 for g in gradients),
                    "Silent references and estimates acquired a loss or gradient")
        rows.append({"case": case, "batch_size": batch, "microbatch_size": half, "scored_samples": samples,
                     "active_windows": reduction.active.tolist(), "absent_windows": reduction.absent.tolist(),
                     "term_max_abs_errors": term_errors, "raw_and_deployed_gradient_max_abs_errors": gradient_errors,
                     "incorrect_naive_mean_loss_difference": naive_error})
    require(not torch.cuda.is_initialized(), "CPU loss qualification initialized CUDA")
    return {"status": "pass", "policy": policy(), "cases": rows,
            "unchanged_full_batch_objective_and_audio_gradients_match": True,
            "unequal_activity_exposes_naive_mean_error": True, "gpu_used": False,
            "model_quality_measured": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and args.plan.resolve().is_relative_to(PHASE)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require an authenticated CPU qualification plan")
    plan = read(args.plan)
    verify_inputs(plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    began = time.monotonic()
    result = check()
    verify_inputs(plan)
    result.update(plan_sha256=args.plan_sha256, source_bindings_unchanged=True,
                  elapsed_seconds=time.monotonic() - began)
    write(args.plan.parent / "result.json", result)
    print(json.dumps(result), flush=True)
