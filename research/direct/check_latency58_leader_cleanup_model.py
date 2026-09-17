"""Check the added objective on the real parent model and reject malformed journals."""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_sdr_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a frozen CPU1 model check")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-model-check-plan-v1", "Unknown model check")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve model proof")
    before = require_space(plan, 400_000_000)
    import torch
    from research.direct.latency58_leader_cleanup_checkpoint import load_parent, validate_recipe
    from research.direct.latency58_counterfactual_teacher import counterfactual_teacher_objective
    from research.direct.latency58_controlled_deployed_loss import controlled_deployed_objective, VERSION
    from research.direct.latency58_controlled_deployed_journal import loss_evidence, validate_microbatch
    from research.direct.latency58_sdr_context import render_scored_context

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    control = read(plan["recipe_plan"]["path"])
    verify_inputs(control)
    invalid_recipes = []
    for key, value in (("teacher_mode", "all_views"), ("additional_loss_weight", 0),
                       ("carry_state", False), ("comparison_variable", "loss_weight"),
                       ("initialized_model_state_sha256", "0" * 64),
                       ("matched_ordinary_training_plan", control["reference_training_plan"])):
        wrong = copy.deepcopy(control)
        wrong[key] = value
        invalid_recipes.append(wrong)
    wrong = copy.deepcopy(control)
    wrong["config"]["steps"] = 500
    invalid_recipes.append(wrong)
    wrong = copy.deepcopy(control)
    wrong["config"]["data_start"] += 16
    invalid_recipes.append(wrong)
    recipe_rejections = 0
    for wrong in invalid_recipes:
        try:
            validate_recipe(wrong)
        except RuntimeError:
            recipe_rejections += 1
    require(recipe_rejections == 8, "Changed recipe or misleading control attribution accepted")
    rng = torch.get_rng_state().clone()
    model = load_parent(control).train().requires_grad_(True)
    parent_hash = state_sha256(model.state_dict())
    sample = torch.arange(1024, dtype=torch.float32) / 44100
    wave = torch.stack([torch.stack((torch.sin(2 * torch.pi * f * sample),
                       .8 * torch.cos(2 * torch.pi * f * sample))) for f in (93, 139, 317, 683)])
    truth = torch.tensor([.03, .0001, .04, .02])[:, None, None, None] * wave[None]
    truth[0, 2] = 0
    truth[1, [0, 1, 3]] = 0
    mixture = truth.sum(dim=1)
    targets = truth[..., 512:]
    teacher = .2 * targets
    teacher[:, 3] = mixture[..., 512:] - teacher[:, :3].sum(dim=1)
    codes, flags = (0, 1, 2, 3), torch.tensor([False, False, True, False])
    snapshots = [x.clone() for x in (truth, mixture, targets, teacher, flags)]

    def run(weight):
        model.zero_grad(set_to_none=True)
        output = render_scored_context(model, mixture, warmup_samples=512, carry_state=True)
        if weight is None:
            terms = counterfactual_teacher_objective(output.raw, output.deployed, targets, teacher, flags,
                                                     view_codes=codes, mode="ordinary_only")
            terms.total.backward()
            row = None
        else:
            terms = controlled_deployed_objective(output.raw, output.deployed, targets, teacher, flags,
                                                  view_codes=codes, weight=weight)
            terms.total.backward()
            gradients_before = {n: p.grad.clone() for n, p in model.named_parameters()}
            row = {"loss": float(terms.total.detach()),
                   "supervised_loss": float((terms.base.waveform_l1 + terms.base.projection_contribution).detach()),
                   "teacher_l1": float(terms.base.teacher_l1.detach()), "view_codes": list(codes),
                   **loss_evidence(output.raw, output.deployed, targets, teacher, terms, codes, weight)}
            validate_microbatch(row, weight)
            require(all(torch.equal(p.grad, gradients_before[n]) for n, p in model.named_parameters()),
                    "Detached logging changed model gradients")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "Missing or nonfinite parent gradient")
        return (float(terms.total.detach()), {n: p.grad.clone() for n, p in model.named_parameters()},
                output.raw.detach().clone(), output.deployed.detach().clone(), row)

    base, zero, half = run(None), run(0), run(.5)
    require(base[0] == zero[0] and len(base[1]) == 21
            and all(torch.equal(base[1][n], zero[1][n]) for n in base[1])
            and all(torch.equal(base[i], zero[i]) and torch.equal(base[i], half[i]) for i in (2, 3)),
            "Zero-weight model gradients or forward outputs changed")
    changed = [n for n in base[1] if not torch.equal(base[1][n], half[1][n])]
    require(changed and half[0] > base[0], "Added supervision does not change parent optimization")
    invalid = []
    for key, value in (("controlled_deployed_keep", [True] * 4),
                       ("additional_loss_weight", 1), ("additional_loss_version", "different"),
                       ("loss", half[4]["loss"] + .01), ("base_loss", half[4]["base_loss"] + .01),
                       ("deployed_truth_per_example_stem_l1", [[float("nan")] * 4] * 4)):
        wrong = copy.deepcopy(half[4]); wrong[key] = value; invalid.append(wrong)
    wrong = copy.deepcopy(half[4])
    wrong["controlled_deployed_l1"] *= 2
    wrong["controlled_deployed_contribution"] *= 2
    wrong["loss"] = wrong["base_loss"] + wrong["controlled_deployed_contribution"]
    invalid.append(wrong)
    rejected = 0
    for wrong in invalid:
        try:
            validate_microbatch(wrong, .5)
        except RuntimeError:
            rejected += 1
    require(rejected == 7 and state_sha256(model.state_dict()) == parent_hash
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(torch.equal(a, b) for a, b in zip((truth, mixture, targets, teacher, flags), snapshots, strict=True)),
            "Invalid journal accepted or parent/input/RNG changed")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-leader-cleanup-model-check-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "additional_loss_version": VERSION, "parent_model_state_sha256": parent_hash,
          "zero_weight_loss_and_all_21_parameter_gradients_exact": True,
          "forward_outputs_exact": True, "half_weight_changes_parameter_gradients": changed,
          "detached_logging_preserves_model_gradients": True, "invalid_journals_rejected": rejected,
          "wrong_controlled_example_divisor_rejected": True,
          "changed_recipe_and_misleading_control_plans_rejected": recipe_rejections,
          "zero_gradient_sha256": {n: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for n, v in zero[1].items()},
          "half_gradient_sha256": {n: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for n, v in half[1].items()},
          "examples": {"zero": zero[4], "half": half[4]},
          "parent_and_source_tensors_unchanged": True, "rng_unchanged": True,
          "warmup_samples": 512, "scored_samples": 512, "precision": "float32",
          "training_updates_executed": 0, "optimizer_instances": 0, "cuda_initialized": False,
          "counted_bytes_before": before, "quality_selected": False,
          "limitations": ["Short CPU model fixture with synthetic teacher targets; full-crop BF16 GPU replay remains required."]})
    print({"status": "pass", "parameter_gradients_exact": 21, "half_weight_changed_gradients": len(changed),
           "invalid_journals_rejected": rejected}, flush=True)


if __name__ == "__main__":
    main()
