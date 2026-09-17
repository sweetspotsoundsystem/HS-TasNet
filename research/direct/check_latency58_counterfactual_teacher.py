"""Verify selective distillation preserves ordinary examples and quiet truth."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_vocal_focus_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Changed functional plan or cwd")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-counterfactual-teacher-functional-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the CPU1 functional check")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve functional result")
    before = require_space(plan, 1_000_000)
    import torch
    from research.direct.latency58_counterfactual_teacher import counterfactual_teacher_objective, VERSION
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = torch.get_rng_state().clone()
    sample = torch.arange(4096, dtype=torch.float32) / 44100
    sources = torch.stack([torch.stack((torch.sin(2 * torch.pi * frequency * sample),
                          .8 * torch.cos(2 * torch.pi * frequency * sample))) for frequency in (93, 139, 317, 683)])
    truth = torch.tensor([.03, .0001, .04, .02])[:, None, None, None] * sources[None]
    truth[0, 2] = 0
    truth[1, [0, 1, 3]] = 0
    mixture = truth.sum(dim=1)
    initial = .5 * truth + .03 * mixture[:, None]
    teacher = .2 * truth
    teacher[:, 3] = mixture - teacher[:, :3].sum(dim=1)
    flags = torch.tensor([False, False, True, False])
    codes = (0, 1, 2, 3)
    originals = [x.clone() for x in (initial, truth, teacher, flags)]

    def evaluate(kind, *, teacher_value=teacher, assignments=codes):
        raw = initial.clone().requires_grad_(True)
        deployed = torch.cat((raw[:, :3], mixture[:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
        if kind in ("base", "supervised"):
            terms = drum_emphasized_objective(raw, deployed, truth, teacher_value, flags)
        else:
            terms = counterfactual_teacher_objective(raw, deployed, truth, teacher_value, flags,
                                                     view_codes=assignments, mode=kind)
        total = terms.waveform_l1 + terms.projection_contribution if kind == "supervised" else terms.total
        total.backward()
        require(bool(torch.isfinite(raw.grad).all()), "Nonfinite output gradient")
        return terms, raw.grad.clone(), deployed.detach(), total.detach()

    base, base_grad, deployed, _ = evaluate("base")
    control, control_grad, _, _ = evaluate("all_views")
    ordinary, ordinary_grad, _, _ = evaluate("ordinary_only", assignments=(2, 2, 2, 2))
    for value, gradient in ((control, control_grad), (ordinary, ordinary_grad)):
        require(torch.equal(base.total, value.total) and torch.equal(base.teacher_l1, value.teacher_l1)
                and torch.equal(base.projection_contribution, value.projection_contribution)
                and torch.equal(base_grad, gradient), "Existing-loss control or all-ordinary batch changed")
    focused, focused_grad, _, _ = evaluate("ordinary_only")
    # Independent B=1 calls retain the original teacher definition, then
    # divide the two kept examples by the full B=4, not by two.
    expected_teacher = sum(drum_emphasized_objective(initial[i:i+1], deployed[i:i+1], truth[i:i+1],
                           teacher[i:i+1], flags[i:i+1]).teacher_l1 for i in (2, 3)) / 4
    require(abs(float((focused.teacher_l1 - expected_teacher).detach())) < 1e-8
            and focused.controlled_examples == focused.ordinary_examples == 2
            and torch.equal(focused.waveform_l1, base.waveform_l1)
            and torch.equal(focused.projection_contribution, base.projection_contribution),
            "Selective teacher changed batch weighting, source supervision or the original projection cap")
    ordinary_gradient_error = float((focused_grad[2:] - base_grad[2:]).abs().max())
    require(ordinary_gradient_error < 1e-10, "Ordinary examples no longer receive their original gradients")
    _, supervised_grad, _, _ = evaluate("supervised")
    controlled_gradient_error = float((focused_grad[:2] - supervised_grad[:2]).abs().max())
    require(controlled_gradient_error < 1e-10, "A controlled example still receives a teacher gradient")
    corrupted = teacher.clone()
    corrupted[:2] = -3 * corrupted[:2] + .05
    changed, changed_grad, _, _ = evaluate("ordinary_only", teacher_value=corrupted)
    require(torch.equal(changed.total, focused.total) and torch.equal(changed.teacher_l1, focused.teacher_l1)
            and torch.equal(changed_grad, focused_grad), "Changing excluded teacher targets changed the loss or gradients")
    altered_ordinary = teacher.clone()
    altered_ordinary[2:] *= -2
    changed_ordinary, _, _, _ = evaluate("ordinary_only", teacher_value=altered_ordinary)
    require(not torch.equal(changed_ordinary.total, focused.total), "Ordinary teacher targets are being ignored")
    quiet_rms_dbfs = float(20 * torch.log10(truth[1, 2].square().mean().sqrt()))
    quiet_derivative = float((focused_grad[1, 2] * (truth[1, 2] - initial[1, 2])).sum())
    require(quiet_rms_dbfs < -50 and quiet_derivative < 0 and bool(torch.count_nonzero(focused_grad[1, 2])),
            "Quiet wanted vocals lost their native ground-truth supervision")
    rejected = 0
    for bad_codes, bad_flags, bad_targets in (
        ((0, 1, 4, 3), flags, truth),
        (codes, torch.tensor([True, False, True, False]), truth),
        (codes, flags, truth + .001),
    ):
        try:
            counterfactual_teacher_objective(initial, deployed, bad_targets, teacher, bad_flags,
                                             view_codes=bad_codes, mode="ordinary_only")
        except RuntimeError:
            rejected += 1
    require(rejected == 3, "Invalid view assignments or targets were accepted")
    require(all(torch.equal(a, b) for a, b in zip((initial, truth, teacher, flags), originals, strict=True))
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Functional check mutated source tensors, RNG or CPU scope")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-counterfactual-teacher-functional-v1", "status": "pass",
          "version": VERSION, "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
          "source_bindings_unchanged": True, "existing_loss_control_and_gradients_exact": True,
          "all_ordinary_batch_loss_and_gradients_exact": True, "original_raw_supervision_and_projection_exact": True,
          "original_batch_divisor_preserved": True, "controlled_teacher_perturbation_no_effect_exact": True,
          "ordinary_teacher_perturbation_changes_loss": True, "ordinary_output_gradient_max_abs_error": ordinary_gradient_error,
          "controlled_output_gradient_max_abs_error_against_supervised": controlled_gradient_error,
          "quiet_vocal_target_rms_dbfs": quiet_rms_dbfs, "quiet_vocal_truth_directional_derivative": quiet_derivative,
          "invalid_requests_rejected": rejected, "input_tensors_and_rng_unchanged": True,
          "uniform_total_loss": float(base.total.detach()), "selective_total_loss": float(focused.total.detach()),
          "uniform_teacher_l1": float(base.teacher_l1.detach()), "selective_teacher_l1": float(focused.teacher_l1.detach()),
          "model_instances": 0, "optimizer_instances": 0, "training_updates_executed": 0, "cuda_initialized": False,
          "quality_selected": False, "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
          "limitations": ["Synthetic FP32 output-gradient qualification only; no real-data training, GPU rehearsal or quality improvement is established.",
                          "The ongoing three-arm pilot still uses its original frozen objective."]})
    print({"status": "pass", "ordinary_gradient_error": ordinary_gradient_error,
           "controlled_gradient_error": controlled_gradient_error, "quiet_vocal_rms_dbfs": quiet_rms_dbfs}, flush=True)


if __name__ == "__main__":
    main()
