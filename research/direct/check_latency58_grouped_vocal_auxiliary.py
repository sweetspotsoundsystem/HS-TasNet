"""CPU data and output-gradient checks for the prospective two-group loss."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write


def main():
    import torch
    from research.direct.train_latency58 import verify_inputs
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    from research.direct.latency58_branch_sdr_blend import objective as complete_objective
    from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, contribution, policy, AUXILIARY_WEIGHT
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Require CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    names = ("check_latency58_grouped_vocal_auxiliary.py", "latency58_grouped_vocal_auxiliary.py",
        "latency58_logical_batch_loss.py", "latency58_branch_sdr_blend.py", "latency58_direct_sdr.py",
        "latency58_wave_spectral.py", "train_latency58.py", "run_latency58_quality.py",
        "run_latency58_deployed_vocal_views.py")
    bindings = {str(ROOT / "research/direct" / n): sha(ROOT / "research/direct" / n) for n in names}
    baseline_review = PHASE / "deployed-vocal-baseline-review-001/result.json"
    bindings[str(baseline_review)] = sha(baseline_review)
    budget_plan = {**read(PHASE / "deployed-vocal-views-001/plan.json"), "diagnostic_artifact_allowance_bytes": 2_050_000_000}
    before = budget_snapshot(budget_plan)
    out = PHASE / "grouped-vocal-auxiliary-cpu-001"
    require(not out.exists(), "Preserve prior group-loss checks")
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    plan = {"schema": "latency58-grouped-vocal-auxiliary-cpu-v1", "source_bindings": bindings,
        "policy": policy(), "budget_before": before, "training_data_used": False, "validation_audio_used": False,
        "model_parent_selected": False, "quality_measured": False, "gpu_used": False}
    write(out / "plan.json", plan)
    generator = torch.Generator().manual_seed(20260914)
    samples = 2 * 44100 + 128
    targets = .025 * torch.randn((16, 4, 2, samples), generator=generator)
    # Uneven activity across microbatches and the two complete scored windows.
    targets[:3, 0] = 0
    targets[4:8, 1, :, :44100] = 0
    targets[10:13, 2] = 0
    targets[13, 3, :, 44100:] = 0
    mixture = targets.sum(1) + .000123
    target_before, mixture_before = targets.clone(), mixture.clone()
    rng_before = torch.get_rng_state().clone()
    auxiliary_mix, auxiliary = source_views(mixture, targets)
    require(torch.equal(targets, target_before) and torch.equal(mixture, mixture_before)
            and not torch.equal(mixture, targets.sum(1)) and torch.equal(torch.get_rng_state(), rng_before),
            "Ordinary recorded-mixture mismatch, inputs or RNG changed")
    require(torch.equal(auxiliary_mix, auxiliary.sum(1)) and torch.count_nonzero(auxiliary[0, 2]) == 0
            and torch.count_nonzero(auxiliary[1, [0, 1, 3]]) == 0
            and torch.equal(auxiliary[0, [0, 1, 3]], targets[14, [0, 1, 3]])
            and torch.equal(auxiliary[1, 2], targets[15, 2]), "Wrong source removal or kept native samples")
    require(all(torch.equal(a, b) for a, b in zip(source_views(mixture, targets), (auxiliary_mix, auxiliary), strict=True)),
            "Source-view replay differs")
    groups = prepare_groups(targets, auxiliary)
    ordinary_noise = .006 * torch.randn(targets.shape, generator=generator)
    auxiliary_noise = .006 * torch.randn(auxiliary.shape, generator=generator)
    # Distinct leaf coordinates expose both raw-head and deployed supervision.
    leaves = [(targets + .8 * ordinary_noise).requires_grad_(), (targets + ordinary_noise).requires_grad_(),
              (auxiliary + .8 * auxiliary_noise).requires_grad_(), (auxiliary + auxiliary_noise).requires_grad_()]
    expected = complete_objective(leaves[0], leaves[1], targets, mixture).total
    expected = expected + AUXILIARY_WEIGHT * complete_objective(leaves[2], leaves[3], auxiliary, auxiliary_mix).total
    expected_gradient = torch.autograd.grad(expected, leaves)
    observed_parts = []
    for start in range(0, 16, 4):
        observed_parts.append(contribution("ordinary", leaves[0][start:start+4], leaves[1][start:start+4],
            targets[start:start+4], mixture[start:start+4], groups)[0])
    for start in range(2):
        observed_parts.append(contribution("auxiliary", leaves[2][start:start+1], leaves[3][start:start+1],
            auxiliary[start:start+1], auxiliary_mix[start:start+1], groups)[0])
    observed = sum(observed_parts)
    actual_gradient = torch.autograd.grad(observed, leaves)
    loss_error = abs(float((observed - expected).detach()))
    gradient_errors = [float((a - b).abs().max()) for a, b in zip(actual_gradient, expected_gradient, strict=True)]
    require(loss_error <= 2e-6 and all(torch.allclose(a, b, atol=2e-10, rtol=2e-5)
            for a, b in zip(actual_gradient, expected_gradient, strict=True)), "Accumulation changed the independent complete-group objective")
    require(all(bool(torch.isfinite(g).all()) for g in actual_gradient)
            and float((actual_gradient[3][0, 2] * auxiliary_noise[0, 2]).sum()) > 0
            and float((actual_gradient[3][1, 2] * auxiliary_noise[1, 2]).sum()) > 0,
            "Absent or wanted vocals lost their finite restoring gradient")
    # Auxiliary support must not enter ordinary denominators or gradients.
    silent = torch.zeros_like(auxiliary)
    altered = prepare_groups(targets, silent)
    ordinary_first = sum(contribution("ordinary", leaves[0][i:i+4], leaves[1][i:i+4], targets[i:i+4], mixture[i:i+4], groups)[0]
                         for i in range(0, 16, 4))
    ordinary_second = sum(contribution("ordinary", leaves[0][i:i+4], leaves[1][i:i+4], targets[i:i+4], mixture[i:i+4], altered)[0]
                          for i in range(0, 16, 4))
    require(torch.equal(ordinary_first, ordinary_second), "Auxiliary activity renormalized ordinary loss")
    grad_first = torch.autograd.grad(ordinary_first, leaves[:2])
    grad_second = torch.autograd.grad(ordinary_second, leaves[:2])
    require(all(torch.equal(a, b) for a, b in zip(grad_first, grad_second, strict=True)), "Auxiliary support changed ordinary gradients")
    all_silent = prepare_groups(torch.zeros_like(targets), silent)
    require(contribution("ordinary", torch.zeros_like(targets), torch.zeros_like(targets), torch.zeros_like(targets),
            torch.zeros_like(mixture), all_silent)[0].item() == 0
            and contribution("auxiliary", silent, silent, silent, silent.sum(1), all_silent)[0].item() == 0,
            "Perfect all-silent output must have zero loss")
    with torch.no_grad():
        improved = complete_objective(auxiliary + .4 * auxiliary_noise, auxiliary + .5 * auxiliary_noise, auxiliary, auxiliary_mix).total
        initial = complete_objective(leaves[2], leaves[3], auxiliary, auxiliary_mix).total
        require(improved < initial, "Reducing auxiliary source error failed to improve loss")
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "CPU qualification used GPU")
    result = {"status": "pass", "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "policy": policy(), "independent_complete_group_loss_error": loss_error,
        "maximum_raw_and_deployed_gradient_errors": gradient_errors,
        "ordinary_inputs_and_recorded_mismatch_preserved": True, "source_views_exact_for_entire_supplied_context": True,
        "ordinary_loss_and_gradients_bit_exact_when_auxiliary_activity_changes": True,
        "absent_and_wanted_vocals_have_restoring_gradients": True, "perfect_silence_loss_zero": True,
        "reduced_error_improves_auxiliary_loss": True, "global_rng_unchanged_by_view_augmentation": True,
        "gpu_used": False, "training_updates": 0, "quality_measured": False, "production_recipe_selected": False,
        "model_state_or_warmup_qualification_completed": False, "budget_after": budget_snapshot(budget_plan),
        "limitations": "Synthetic data and output-coordinate gradients only. Recorded training data replay, independent carried-state warmup, selected-parent parameter gradients, GPU resources, saved full14 quality, real-vocal preservation and native timing are still required."}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "independent_complete_group_loss_error", "maximum_raw_and_deployed_gradient_errors",
        "ordinary_loss_and_gradients_bit_exact_when_auxiliary_activity_changes", "production_recipe_selected")}))


if __name__ == "__main__":
    main()
