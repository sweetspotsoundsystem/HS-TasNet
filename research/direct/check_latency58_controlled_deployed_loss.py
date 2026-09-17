"""Check the proposed deployed-source training term without model updates."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 diagnostic")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-controlled-deployed-loss-check-plan-v1", "Different check scope")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve functional evidence")
    from research.direct.latency58_sdr_checkpoint import require_space
    before = require_space(plan, 1_000_000)
    import torch
    from research.direct.latency58_counterfactual_teacher import counterfactual_teacher_objective
    from research.direct.latency58_controlled_deployed_loss import controlled_deployed_objective, VERSION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = torch.get_rng_state().clone()
    time = torch.arange(2048, dtype=torch.float32) / 44100
    carriers = torch.stack([torch.stack((torch.sin(2 * torch.pi * f * time),
                                        .8 * torch.cos(2 * torch.pi * f * time))) for f in (83, 173, 317, 731)])
    codes = (0, 1, 2, 3)
    flags = torch.tensor([False, False, True, False])
    rows = []
    for amplitude in (.07, .0003, .00005, 0.0):
        truth = amplitude * carriers[None].repeat(4, 1, 1, 1)
        truth[0, 2] = 0
        truth[1, [0, 1, 3]] = 0
        mixture = truth.sum(dim=1, keepdim=True)
        teacher = .65 * truth
        teacher[:, 3:4] = mixture - teacher[:, :3].sum(dim=1, keepdim=True)
        fixture = .73 * truth + .00002 * carriers.flip(0)[None]

        def evaluate(kind):
            raw = fixture.clone().requires_grad_(True)
            deployed = torch.cat((raw[:, :3], mixture - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
            if kind == "base":
                terms = counterfactual_teacher_objective(raw, deployed, truth, teacher, flags,
                                                         view_codes=codes, mode="ordinary_only")
                total, base = terms.total, terms
            else:
                terms = controlled_deployed_objective(raw, deployed, truth, teacher, flags, view_codes=codes,
                                                     weight=0 if kind == "zero" else .5)
                total, base = terms.total, terms.base
                if kind == "independent":
                    weights = truth.new_tensor([2, 1, 1, 1])[None, :, None, None]
                    divisor = 5 * raw.shape[0] * raw.shape[2] * raw.shape[3]
                    auxiliary = ((deployed[:2] - truth[:2]).abs() * weights).sum() / divisor
                    total = base.total + .5 * auxiliary
            total.backward()
            require(bool(torch.isfinite(raw.grad).all()), "Nonfinite output gradient")
            return total.detach(), raw.grad, base

        base, zero, candidate, independent = [evaluate(kind) for kind in ("base", "zero", "candidate", "independent")]
        require(torch.equal(base[0], zero[0]) and torch.equal(base[1], zero[1]), "Zero coefficient changed base loss or gradient")
        scalar_error = float((candidate[0] - independent[0]).abs())
        gradient_error = float((candidate[1] - independent[1]).abs().max())
        require(scalar_error < 1e-8 and gradient_error < 1e-10, "Deployed truth term differs from independent full-batch arithmetic")
        require(torch.equal(candidate[1][2:], base[1][2:]) and torch.equal(candidate[1][:, 3], base[1][:, 3])
                and float((candidate[1][:2, :3] - base[1][:2, :3]).abs().max()) > 0,
                "The term must affect controlled deployed heads while preserving ordinary examples and raw Other gradients")
        require(torch.equal(candidate[2].projection_contribution, base[2].projection_contribution)
                and torch.equal(candidate[2].teacher_l1, base[2].teacher_l1), "Existing loss terms changed")
        rows.append({"amplitude": amplitude, "base_loss": float(base[0]), "candidate_loss": float(candidate[0]),
                     "scalar_oracle_max_abs": scalar_error, "gradient_oracle_max_abs": gradient_error,
                     "zero_weight_loss_and_gradients_exact": True, "ordinary_output_gradients_exact": True,
                     "raw_other_gradients_exact": True, "existing_projection_contribution": float(base[2].projection_contribution)})
    require(rows[0]["existing_projection_contribution"] > 0, "Fixture did not exercise the existing projection cap")

    directions = []
    for amplitude in (.07, .0003, .00005):
        truth = torch.zeros(4, 4, 2, 2048)
        truth[1, 2] = amplitude * carriers[2]
        mixture = truth.sum(dim=1, keepdim=True)
        for fraction in (.25, .5, .75):
            derivatives = []
            for weight in (0, .5):
                raw = fraction * truth
                raw.requires_grad_(True)
                deployed = torch.cat((raw[:, :3], mixture - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
                loss = controlled_deployed_objective(raw, deployed, truth, truth, torch.zeros(4, dtype=torch.bool),
                                                    view_codes=codes, weight=weight)
                loss.total.backward()
                derivatives.append(float((raw.grad[1, 2] * truth[1, 2]).sum()))
            require(derivatives[0] < 0 and abs(derivatives[1] - 2 * derivatives[0]) < 1e-9,
                    "Vocal recovery direction does not include both deployed Vocal and residual Other errors")
            directions.append({"amplitude": amplitude, "fraction": fraction,
                               "base_vocal_directional_derivative": derivatives[0],
                               "candidate_vocal_directional_derivative": derivatives[1]})
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "Diagnostic changed RNG or initialized CUDA")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-controlled-deployed-loss-check-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "version": VERSION, "native_level_cases": rows, "vocal_recovery_directions": directions,
          "training_updates_executed": 0, "model_instances": 0, "optimizer_instances": 0,
          "cuda_initialized": False, "rng_unchanged": True, "quality_selected": False,
          "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
          "limitations": ["Synthetic output-space semantics; no parameter-gradient, full-crop BF16 or resource qualification.",
                          "Source-view loss behavior does not establish full-mixture quality or audible improvement.",
                          "This check neither selects a training parent or schedule nor changes a running trial."]})
    print({"status": "pass", "native_level_cases": len(rows), "vocal_recovery_directions": len(directions)}, flush=True)


if __name__ == "__main__":
    main()
