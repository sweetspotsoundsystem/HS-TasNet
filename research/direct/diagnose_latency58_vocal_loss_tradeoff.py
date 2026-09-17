"""Check a vocal-output flat direction in the frozen raw4/distillation loss."""
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
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Different loss diagnostic plan")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-loss-tradeoff-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require the CPU1 synthetic diagnostic")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve loss diagnostic")
    before = require_space(plan, 1_000_000)
    import torch
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective, DRUM_WEIGHT, TEACHER_WEIGHT
    require(DRUM_WEIGHT == 2.0 and TEACHER_WEIGHT == 0.5, "The current pilot loss weights differ")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = torch.get_rng_state().clone()
    time = torch.arange(2048, dtype=torch.float32) / 44100
    wave = torch.stack((torch.sin(2 * torch.pi * 317 * time) + .2 * torch.cos(2 * torch.pi * 731 * time),
                        -.8 * torch.sin(2 * torch.pi * 317 * time) + .15 * torch.cos(2 * torch.pi * 977 * time)))
    mixture = torch.tensor([.07, .02, .0003, .00005])[:, None, None] * wave[None]
    truth = torch.zeros(4, 4, 2, 2048)
    truth[:, 2] = mixture
    teacher = torch.zeros_like(truth)
    teacher[:, 0] = .03 * mixture
    teacher[:, 1] = -.02 * mixture
    teacher[:, 2] = .1 * mixture + .03 * mixture.flip(-1)
    teacher[:, 3] = mixture - teacher[:, :3].sum(dim=1)
    flags = torch.zeros(4, dtype=torch.bool)
    direction = truth[:, 2] - teacher[:, 2]
    weights = truth.new_tensor([2, 1, 1, 1])[None, :, None, None]
    shape = (truth.shape[0], 1, truth.shape[2], truth.shape[3])
    denominator = 5 * truth.shape[0] * truth.shape[2] * truth.shape[3]
    expected_slope = -float(direction.abs().sum()) / denominator

    def weighted(a, b):
        return ((a - b).abs() * weights).sum() / denominator

    cases = []
    for fraction in (0.0, .125, .25, .5, .75, .875, 1.0):
        for variant in ("current", "teacher_half_strength", "controlled_view_ground_truth_only", "add_deployed_truth_half"):
            raw = truth.clone()
            raw[:, :2] = teacher[:, :2]
            raw[:, 2] = teacher[:, 2] + fraction * direction
            raw.requires_grad_(True)
            deployed = torch.cat((raw[:, :3], mixture[:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
            require(deployed.shape == truth.shape and deployed[:, 3:4].shape == shape, "Residual output geometry differs")
            terms = drum_emphasized_objective(raw, deployed, truth, teacher, flags)
            require(float(terms.projection_contribution) == 0, "Forced unshuffled view unexpectedly has a projection penalty")
            independent = weighted(raw, truth) + .5 * weighted(deployed, teacher)
            require(abs(float(terms.total - independent)) < 1e-8, "Independent scalar objective differs from the frozen loss")
            if variant == "current":
                loss = terms.total
                expected = 0.0
            elif variant == "teacher_half_strength":
                loss = terms.waveform_l1 + .25 * terms.teacher_l1
                expected = .5 * expected_slope
            elif variant == "controlled_view_ground_truth_only":
                loss = terms.waveform_l1
                expected = expected_slope
            else:
                loss = terms.total + .5 * weighted(deployed, truth)
                # Teacher's small DB offset can change Other's absolute-error
                # sign near the truth endpoint. Check this variant's measured
                # decrease separately; do not impose the flat-direction oracle.
                expected = None
            loss.backward()
            derivative = float((raw.grad[:, 2] * direction).sum())
            if 0 < fraction < 1 and expected is not None:
                require(abs(derivative - expected) < 1e-8, "Vocal directional derivative differs from the L1 oracle")
            cases.append({"fraction_teacher_to_truth": fraction, "variant": variant,
                          "total_loss": float(loss), "raw_supervised_l1": float(terms.waveform_l1),
                          "deployed_teacher_l1": float(terms.teacher_l1),
                          "vocal_directional_derivative": derivative,
                          "expected_interior_derivative": expected,
                          "vocal_gradient_max_abs": float(raw.grad[:, 2].abs().max()),
                          "other_raw_gradient_max_abs": float(raw.grad[:, 3].abs().max()),
                          "finite_all_output_gradients": bool(torch.isfinite(raw.grad).all()),
                          "output_sum_max_abs": float((deployed.sum(dim=1) - mixture).abs().max())})
    current = [row for row in cases if row["variant"] == "current"]
    require(max(row["total_loss"] for row in current) - min(row["total_loss"] for row in current) < 1e-8
            and all(row["vocal_gradient_max_abs"] < 1e-10 for row in current if 0 < row["fraction_teacher_to_truth"] < 1),
            "Current loss is not flat along the specified vocal-output line")
    alternatives = {}
    for variant in ("teacher_half_strength", "controlled_view_ground_truth_only", "add_deployed_truth_half"):
        rows = [row for row in cases if row["variant"] == variant]
        require(all(a["total_loss"] > b["total_loss"] for a, b in zip(rows, rows[1:]))
                and all(row["vocal_directional_derivative"] < 0 for row in rows if 0 < row["fraction_teacher_to_truth"] < 1),
                "Alternative does not restore preference for the wanted vocal in this fixture")
        alternatives[variant] = {"loss_decreases_toward_truth": True,
                                 "teacher_endpoint_loss": rows[0]["total_loss"], "truth_endpoint_loss": rows[-1]["total_loss"]}
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(row["finite_all_output_gradients"] and row["output_sum_max_abs"] <= 1e-6 for row in cases),
            "Diagnostic changed RNG/CUDA scope or produced invalid gradients")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-vocal-loss-tradeoff-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "cases": cases, "current_loss_flat_in_vocal_direction": True, "alternatives": alternatives,
          "fixture": {"batch": 4, "stereo_samples": 2048, "amplitudes": [.07, .02, .0003, .00005],
                      "teacher_vocal": ".1 * truth + .03 * time-reversed truth", "teacher_drum_gain": .03,
                      "teacher_bass_gain": -.02, "teacher_other": "mixture minus DBV", "raw_other": "ground truth zero",
                      "fixed_student_DB": "teacher DB", "student_vocal": "linear interpolation from teacher to truth",
                      "vocal_derangement": False},
          "expected_raw_only_interior_directional_derivative": expected_slope,
          "training_updates_executed": 0, "model_instances": 0, "optimizer_instances": 0,
          "cuda_initialized": False, "rng_unchanged": True, "quality_selected": False,
          "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
          "limitations": ["Output-space synthetic fixture, not a model-parameter gradient or proof of the pilot's training trajectory.",
                          "Other heads are fixed while Vocals varies; shared model parameters can alter several heads together.",
                          "Alternative objective values are diagnostics only; no training recipe was changed or selected.",
                          "Source-isolated correctness and quiet-vocal preservation still require real-data validation and listening."]})
    print({"status": "pass", "current_loss_flat_in_vocal_direction": True, "alternatives": alternatives}, flush=True)


if __name__ == "__main__":
    main()
