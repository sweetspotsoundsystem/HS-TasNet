"""Check the proposed auxiliary's values, gradients and preserved base loss."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Functional plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-softcap-functional-v1"
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Functional sources changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    import numpy as np
    import torch
    from research.direct.latency58_sdr_softcap import (
        ACTIVITY_POWER, AUXILIARY_WEIGHT, EPSILON, ERROR_RATIO_FLOOR, VERSION, WINDOW_SAMPLES, sdr_softcap_error,
    )
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.latency_ola512_training import raw4_native_objective

    require(ERROR_RATIO_FLOOR == plan["error_ratio_floor"] == 0.01
            and AUXILIARY_WEIGHT == plan["auxiliary_weight"] == 0.003,
            "Soft cap or auxiliary coefficient differs from the frozen recipe")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()
    length = WINDOW_SAMPLES + 17
    target = torch.empty(2, 4, 2, length, dtype=torch.float32)
    target[..., :WINDOW_SAMPLES] = torch.tensor([[.1, .002, .2, .003], [.03, .04, .05, .06]])[..., None, None]
    target[..., WINDOW_SAMPLES:] = torch.tensor([[.05, .1, .002, .004], [.002, .04, 0., .06]])[..., None, None]
    estimate = (1.5 * target).requires_grad_()
    result = sdr_softcap_error(estimate, target)
    gradient = torch.autograd.grad(result.total, estimate)[0]
    reference_np = target.numpy().astype(np.float64)
    error_np = estimate.detach().numpy().astype(np.float64) - reference_np
    losses, masks, powers, errors = [], [], [], []
    slices = [(0, WINDOW_SAMPLES), (WINDOW_SAMPLES, length)]
    for start, end in slices:
        power = np.square(reference_np[..., start:end]).mean(axis=(-2, -1))
        error = np.square(error_np[..., start:end]).mean(axis=(-2, -1))
        active = power > ACTIVITY_POWER
        losses.append(np.where(active, np.log((error + ERROR_RATIO_FLOOR * (np.maximum(power, ACTIVITY_POWER) + EPSILON)) / (ERROR_RATIO_FLOOR * (np.maximum(power, ACTIVITY_POWER) + EPSILON))), 0.))
        masks.append(active)
        powers.append(power)
        errors.append(error)
    count = np.stack(masks, axis=-1).sum(axis=-1)
    per_example = np.stack(losses, axis=-1).sum(axis=-1) / np.maximum(count, 1)
    per_stem_count = (count > 0).sum(axis=0)
    per_stem = per_example.sum(axis=0) / np.maximum(per_stem_count, 1)
    expected = per_stem.sum() / np.maximum((per_stem_count > 0).sum(), 1)
    value_error = abs(float(result.total) - expected)
    require(value_error < 1e-6 and np.array_equal(result.active_window_counts.numpy(), count)
            and np.max(np.abs(result.per_stem.detach().numpy() - per_stem)) < 1e-6,
            "Active/partial-window values or weighting differ from the independent calculation")
    expected_gradient = np.zeros_like(error_np)
    for index, (start, end) in enumerate(slices):
        weight = masks[index] / np.maximum(count, 1) / np.maximum(per_stem_count[None], 1) / (per_stem_count > 0).sum()
        derivative = 2 * error_np[..., start:end] / (2 * (end - start))
        derivative /= (ERROR_RATIO_FLOOR * (np.maximum(powers[index], ACTIVITY_POWER) + EPSILON) + errors[index])[..., None, None]
        expected_gradient[..., start:end] = derivative * weight[..., None, None]
    gradient_error = float(np.max(np.abs(gradient.numpy() - expected_gradient)))
    require(np.allclose(gradient.numpy(), expected_gradient, rtol=1e-5, atol=1e-9)
            and bool(torch.isfinite(gradient).all()) and float((gradient * (estimate.detach() - target)).sum()) > 0,
            "Auxiliary gradient differs from the analytic derivative")
    for index, (start, end) in enumerate(slices):
        require(bool((gradient[..., start:end][torch.from_numpy(~masks[index])] == 0).all()),
                "Absent windows receive an auxiliary gradient")
    perfect = target.clone().requires_grad_()
    perfect_loss = sdr_softcap_error(perfect, target).total
    require(float(perfect_loss) == 0 and not bool(torch.autograd.grad(perfect_loss, perfect)[0].any())
            and float(result.total) > 0, "Perfect reconstruction or scale sensitivity differs")
    absent = torch.full_like(target, .1, requires_grad=True)
    absent_loss = sdr_softcap_error(absent, torch.zeros_like(target)).total
    require(float(absent_loss) == 0 and not bool(torch.autograd.grad(absent_loss, absent)[0].any()),
            "Entirely absent targets should leave the auxiliary inactive")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_loss = sdr_softcap_error(estimate, target).total
    require(autocast_loss.dtype == torch.float32 and torch.equal(autocast_loss, result.total),
            "Auxiliary escaped its FP32 precision scope")

    raw = (target * 1.5).requires_grad_()
    mixture = target.sum(dim=1) + .001
    deployed = torch.cat((raw[:, :3], (mixture - raw[:, :3].sum(dim=1))[:, None]), dim=1)
    teacher = (target * .95).detach()
    teacher_before = teacher.clone()
    base_terms = raw4_native_objective(raw, target, torch.tensor([True, False]), projection=True)
    teacher_l1 = deployed_teacher_l1(deployed, teacher)
    base = base_terms.total + .5 * teacher_l1
    auxiliary = sdr_softcap_error(deployed, target)
    combined = base + AUXILIARY_WEIGHT * auxiliary.total
    base_gradient = torch.autograd.grad(base, raw, retain_graph=True)[0]
    combined_gradient = torch.autograd.grad(combined, raw)[0]
    require(torch.equal(base + 0 * auxiliary.total, base)
            and abs(float(combined - base) - AUXILIARY_WEIGHT * float(auxiliary.total)) < 1e-7
            and torch.equal(base_gradient[:, 3], combined_gradient[:, 3])
            and bool(base_gradient[:, 3].abs().sum() > 0)
            and bool(torch.isfinite(combined_gradient).all())
            and bool(((combined_gradient - base_gradient)[:, :3].abs().sum()) > 0)
            and torch.equal(teacher_before, teacher),
            "The addition changed base coefficients, raw Other gradient or fixed teacher targets")
    rejected = []
    for label, prediction, reference in (
        ("dtype", estimate.detach().half(), target.half()),
        ("live_target", estimate.detach(), target.clone().requires_grad_()),
        ("shape", estimate.detach()[:, :3], target),
        ("nonfinite", torch.full_like(target, float("nan")), target),
    ):
        try:
            sdr_softcap_error(prediction, reference)
        except ValueError:
            rejected.append(label)
        else:
            raise RuntimeError("Malformed input was accepted: " + label)
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Functional check changed RNG, CPU scope or source bindings")
    output = Path(plan["output_directory"]) / "result.json"
    report = {"schema": "latency58-sdr-softcap-functional-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "version": VERSION, "auxiliary_weight": AUXILIARY_WEIGHT,
              "error_ratio_floor": ERROR_RATIO_FLOOR, "soft_sdr_ceiling_db": -10 * np.log10(ERROR_RATIO_FLOOR),
              "independent_value_error": value_error, "analytic_gradient_max_abs_error": gradient_error,
              "active_window_counts": count.tolist(), "partial_window_samples": 17,
              "active_windows_weighted_within_example_then_stem": True,
              "absent_auxiliary_gradients_zero": True, "perfect_reconstruction_gradient_zero": True,
              "scale_error_penalized": True, "autocast_kept_fp32": True,
              "raw4_and_teacher_coefficients_preserved": True, "raw_other_gradient_bit_exact": True,
              "teacher_targets_unchanged": True, "rejected_inputs": rejected,
              "model_training_executed": False, "quality_selected": False}
    write(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_bindings"}), flush=True)


if __name__ == "__main__":
    main()
