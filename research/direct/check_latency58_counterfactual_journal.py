"""Check detached logging, the full batch divisor, and malformed teacher journals."""
from __future__ import annotations

import argparse
import copy
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
            "Require a frozen CPU1 journal check")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-counterfactual-journal-check-plan-v1", "Unknown check plan")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve the check")
    import torch
    from research.direct.latency58_counterfactual_teacher import counterfactual_teacher_objective
    from research.direct.latency58_counterfactual_journal import loss_evidence, validate_microbatch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = torch.get_rng_state().clone()
    sample = torch.arange(4096, dtype=torch.float32) / 44100
    wave = torch.stack([torch.stack((torch.sin(2 * torch.pi * f * sample),
                       .8 * torch.cos(2 * torch.pi * f * sample))) for f in (93, 139, 317, 683)])
    truth = torch.tensor([.03, .0001, .04, .02])[:, None, None, None] * wave[None]
    truth[0, 2] = 0
    truth[1, [0, 1, 3]] = 0
    mixture = truth.sum(dim=1)
    teacher = .2 * truth
    teacher[:, 3] = mixture - teacher[:, :3].sum(dim=1)
    codes, flags = (0, 1, 2, 3), torch.tensor([False, False, True, False])
    reports = {}
    for mode in ("all_views", "ordinary_only"):
        raw = (.5 * truth + .03 * mixture[:, None]).requires_grad_(True)
        deployed = torch.cat((raw[:, :3], mixture[:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
        terms = counterfactual_teacher_objective(raw, deployed, truth, teacher, flags, view_codes=codes, mode=mode)
        terms.total.backward()
        gradient, raw_before, deployed_before = raw.grad.clone(), raw.detach().clone(), deployed.detach().clone()
        row = {"loss": float(terms.total.detach()), "supervised_loss": float((terms.waveform_l1 + terms.projection_contribution).detach()),
               "teacher_l1": float(terms.teacher_l1.detach()), "view_codes": list(codes),
               **loss_evidence(raw, deployed, truth, teacher, terms, codes)}
        validate_microbatch(row, mode)
        require(torch.equal(raw.grad, gradient) and torch.equal(raw.detach(), raw_before)
                and torch.equal(deployed.detach(), deployed_before), "Logging changed outputs or gradients")
        reports[mode] = row
    altered = []
    wrong = copy.deepcopy(reports["ordinary_only"])
    for key in ("teacher_l1", "unweighted_teacher_l1", "teacher_drum_l1"):
        wrong[key] *= 2
    altered.append(wrong)
    for key, value in (("teacher_keep", [True] * 4), ("controlled_examples", 0),
                       ("view_codes", [0, 1, 2, 4]), ("counterfactual_version", "different")):
        wrong = copy.deepcopy(reports["ordinary_only"])
        wrong[key] = value
        altered.append(wrong)
    rejected = 0
    for wrong in altered:
        try:
            validate_microbatch(wrong, "ordinary_only")
        except RuntimeError:
            rejected += 1
    require(rejected == 5 and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Invalid journal accepted or CPU/RNG scope changed")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-counterfactual-journal-check-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "both_modes_reconstructed_from_per_example_errors": True, "detached_logging_preserves_outputs_and_gradients": True,
          "wrong_active_example_divisor_rejected": True, "invalid_journals_rejected": rejected,
          "input_rng_unchanged": True, "cuda_initialized": False, "model_instances": 0,
          "training_updates_executed": 0, "quality_selected": False, "examples": reports,
          "limitations": ["Synthetic FP32 journal qualification; real-data GPU replay and training remain unverified."]})
    print({"status": "pass", "invalid_journals_rejected": rejected}, flush=True)


if __name__ == "__main__":
    main()
