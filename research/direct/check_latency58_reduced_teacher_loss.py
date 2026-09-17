"""CPU output-gradient checks for reduced ordinary-mixture distillation."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(),
            "Use CPU and preserve existing evidence")
    import torch
    from research.direct.latency58_reduced_teacher_loss import objective, loss_evidence, validate_microbatch, VERSION
    from research.direct.latency58_controlled_deployed_loss import controlled_deployed_objective
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()
    sample = torch.arange(4096, dtype=torch.float32) / 44100
    sources = torch.stack([torch.stack((torch.sin(2 * torch.pi * f * sample),
                          .8 * torch.cos(2 * torch.pi * f * sample))) for f in (93, 139, 317, 683)])
    truth = torch.tensor([.03, .0001, .04, .02])[:, None, None, None] * sources[None]
    truth[0, 2] = 0
    truth[1, [0, 1, 3]] = 0
    mixture = truth.sum(dim=1)
    initial = .5 * truth + .03 * mixture[:, None]
    teacher = .2 * truth
    teacher[:, 3] = mixture - teacher[:, :3].sum(dim=1)
    flags, codes = torch.tensor([False, False, True, False]), (0, 1, 2, 3)
    snapshots = [v.clone() for v in (truth, initial, teacher, flags)]

    def evaluate(coefficient, teacher_value=teacher, original=False):
        raw = initial.clone().requires_grad_(True)
        deployed = torch.cat((raw[:, :3], mixture[:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
        if original:
            terms = controlled_deployed_objective(raw, deployed, truth, teacher_value, flags, view_codes=codes)
        else:
            terms = objective(raw, deployed, truth, teacher_value, flags, view_codes=codes, teacher_weight=coefficient)
        terms.total.backward()
        require(bool(torch.isfinite(raw.grad).all()), "Nonfinite gradient")
        evidence = loss_evidence(raw, deployed, truth, teacher_value, terms, codes, .5, coefficient)
        evidence.update(loss=float(terms.total.detach()), teacher_l1=float(terms.base.teacher_l1.detach()),
                        supervised_loss=float((terms.base.waveform_l1 + terms.base.projection_contribution).detach()),
                        view_codes=list(codes))
        return terms, raw.grad.clone(), evidence

    previous, old_gradient, _ = evaluate(.5, original=True)
    control, control_gradient, _ = evaluate(.5)
    candidate, new_gradient, evidence = evaluate(.25)
    require(torch.equal(previous.total, control.total) and torch.equal(old_gradient, control_gradient),
            "Half-weight control changed the existing objective")
    require(torch.equal(old_gradient[:2], new_gradient[:2]), "Controlled source gradients changed")
    for name in ("waveform_l1", "teacher_l1", "projection", "projection_contribution"):
        require(torch.equal(getattr(control.base, name), getattr(candidate.base, name)), "Unrelated term changed")
    require(torch.equal(control.controlled_deployed_l1, candidate.controlled_deployed_l1)
            and torch.equal(control.controlled_deployed_contribution, candidate.controlled_deployed_contribution),
            "Controlled deployed supervision changed")
    corrupted = teacher.clone()
    corrupted[:2] = -3 * corrupted[:2] + .05
    altered, altered_gradient, _ = evaluate(.25, corrupted)
    require(torch.equal(candidate.total, altered.total) and torch.equal(new_gradient, altered_gradient),
            "Excluded teacher references affected the candidate")
    validate_microbatch(evidence, .25, .5)
    rejected = 0
    for key, value in (("teacher_weight", .5), ("teacher_keep", [True] * 4),
                       ("controlled_deployed_l1", evidence["controlled_deployed_l1"] * 2),
                       ("loss", evidence["loss"] + .25 * evidence["teacher_l1"])):
        malformed = copy.deepcopy(evidence)
        malformed[key] = value
        try:
            validate_microbatch(malformed, .25, .5)
        except RuntimeError:
            rejected += 1
    require(rejected == 4, "Malformed journal arithmetic accepted")
    # On ordinary mixtures, only the Vocal head moves between teacher and truth.
    # Residual Other supplies the second teacher error that cancels truth at .5.
    directional = {}
    reference = truth[2:3].clone()
    mix = reference.sum(dim=1)
    fixed_teacher = reference.clone()
    fixed_teacher[:, 2] *= .2
    fixed_teacher[:, 3] = mix - fixed_teacher[:, :3].sum(dim=1)
    for coefficient in (.5, .25):
        q = torch.tensor(.5, requires_grad=True)
        vocal = fixed_teacher[:, 2:3] + q * (reference[:, 2:3] - fixed_teacher[:, 2:3])
        raw = torch.cat((reference[:, :2], vocal, reference[:, 3:4]), dim=1)
        deployed = torch.cat((raw[:, :3], mix[:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
        terms = objective(raw, deployed, reference, fixed_teacher, torch.tensor([False]),
                          view_codes=(2,), teacher_weight=coefficient)
        terms.total.backward()
        directional[str(coefficient)] = float(q.grad)
    require(abs(directional["0.5"]) < 1e-8 and directional["0.25"] < -1e-5,
            "Quarter weight did not restore the source-truth direction")
    require(all(torch.equal(a, b) for a, b in zip(snapshots, (truth, initial, teacher, flags), strict=True))
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Inputs, RNG or CPU scope changed")
    result = {"status": "pass", "version": VERSION, "source_bindings": {
              str(p): sha(p) for p in (Path(__file__).resolve(),
              Path(__file__).with_name("latency58_reduced_teacher_loss.py").resolve())},
              "half_weight_control_loss_and_gradients_exact": True,
              "controlled_gradients_and_supervision_exact": True,
              "raw_supervision_and_projection_exact": True,
              "excluded_teacher_references_have_no_effect": True,
              "independent_loss_arithmetic_pass": True, "malformed_journals_rejected": rejected,
              "ordinary_vocal_truth_directional_derivatives": directional,
              "cuda_initialized": False, "training_updates_executed": 0,
              "limitations": ["Synthetic output-space gradients; model improvement requires training and full14 scoring."]}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
