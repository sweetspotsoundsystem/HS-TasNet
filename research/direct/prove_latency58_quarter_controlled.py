"""CPU proof of the prospective quarter-controlled augmentation and loss journal.

This does not run a teacher, student, dataset, validation panel or GPU update.
All existing numeric loss functions remain unchanged. Training needs its own
bounded resource rehearsal and complete source/data/RNG journal audit.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def normalized(path, replacements=()):
    text = Path(path).read_text()
    for old, new in replacements:
        require(old in text, "Missing expected source transformation")
        text = text.replace(old, new)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU proof only")
    import numpy as np
    import torch
    from research import experiment
    from research.direct.latency58_vocal_focus_augmentation import augment_vocal_focus as old_augment
    from research.direct.latency58_quarter_controlled_augmentation import augment_vocal_focus, VERSION
    from research.direct.latency58_controlled_deployed_loss import controlled_deployed_objective
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective
    from research.direct.latency58_quarter_controlled_journal import loss_evidence, validate_microbatch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA was initialized")
    directory = PHASE / "quarter-controlled-functional-001"
    require(not directory.exists(), "Preserve an existing proof")
    source_dir = ROOT / "research/direct"
    reference_plan = PHASE / "cleanup-rebound-prep-001/resource-plan.json"
    plan = {"schema": "latency58-quarter-controlled-functional-v1", "augmentation_version": VERSION,
            "reference_plan": {"path": str(reference_plan), "sha256": sha(reference_plan)},
            "source_bindings": dict(read(reference_plan)["source_bindings"]),
            "scope": "CPU synthetic augmentation, loss arithmetic, gradients and journal rejection only",
            "training_updates": 0, "inference_examples": 0, "quality_scores": 0}
    for name in ("latency58_quarter_controlled_augmentation.py", "latency58_variable_counterfactual_journal.py",
                 "latency58_quarter_controlled_journal.py", Path(__file__).name):
        path = source_dir / name
        plan["source_bindings"][str(path)] = sha(path)
    verify_inputs(plan)
    directory.mkdir()
    write(directory / "plan.json", plan)

    augmentation_replacements = (
        ('latency58-quarter-instrumental-quarter-vocal-half-original-v1', VERSION),
        ('VIEW_NAMES = ("instrumental", "vocals_only", "original", "original")',
         'VIEW_NAMES = ("instrumental", "vocals_only", "original", "original")\nVIEW_CYCLE = (0, 1, 2, 3, 2, 3, 2, 3)'),
        ('tuple((first_sample_index + i) % 4 for i in range(4))',
         'tuple(VIEW_CYCLE[(first_sample_index + i) % 8] for i in range(4))'))
    require(normalized(source_dir / "latency58_vocal_focus_augmentation.py", augmentation_replacements)
            == normalized(source_dir / "latency58_quarter_controlled_augmentation.py"),
            "Augmentation changed beyond the version and absolute-address view cycle")
    old_counts = ('require(len(codes) == 4 and sorted(codes) == [0, 1, 2, 3]\n'
                  '            and micro["controlled_examples"] == micro["ordinary_examples"] == 2,\n'
                  '            "Different fixed controlled-view counts")')
    new_counts = ('require(len(codes) == 4 and all(type(code) is int and code in (0, 1, 2, 3) for code in codes)\n'
                  '            and type(micro["controlled_examples"]) is int and type(micro["ordinary_examples"]) is int\n'
                  '            and micro["controlled_examples"] == sum(code < 2 for code in codes)\n'
                  '            and micro["ordinary_examples"] == sum(code >= 2 for code in codes),\n'
                  '            "Invalid view codes or inconsistent controlled-view counts")')
    require(normalized(source_dir / "latency58_counterfactual_journal.py", ((old_counts, new_counts),))
            == normalized(source_dir / "latency58_variable_counterfactual_journal.py"),
            "Teacher journal arithmetic changed beyond recorded count validation")
    require(normalized(source_dir / "latency58_controlled_deployed_journal.py",
                       (("research.direct.latency58_counterfactual_journal",
                         "research.direct.latency58_variable_counterfactual_journal"),))
            == normalized(source_dir / "latency58_quarter_controlled_journal.py"),
            "Deployed journal arithmetic changed")

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def rng():
        return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
                                           torch.get_rng_state().numpy().tobytes()))).hexdigest()

    generator = torch.Generator(device="cpu").manual_seed(20260923)
    targets = torch.randn(4, 4, 2, 256, generator=generator) * .03
    mixture = targets.sum(dim=1)
    pristine = (mixture.clone(), targets.clone())
    cases, rows, rejection_cases = [], [], []
    original_deranged = original_subset = 0
    max_loss_error = max_gradient_error = 0.

    def expect_rejected(label, record):
        try:
            validate_microbatch(record, .5)
        except (RuntimeError, ValueError, AssertionError):
            rejection_cases.append(label)
        else:
            raise AssertionError("Corrupted journal accepted: " + label)

    for repetition in range(16):
        for offset in range(8):
            address = 976000 + offset
            value = 17000 + 8 * repetition + offset
            seed(value)
            before = rng()
            original = experiment._augment_training_distribution(mixture=mixture, targets=targets)
            after = rng()
            original_deranged += int(original[2].sum())
            original_subset += int((original[1].abs().sum(dim=(2, 3)) == 0).sum())
            for enabled in (False, True):
                seed(value)
                require(rng() == before, "Seed replay differs")
                batch = augment_vocal_focus(mixture, targets, first_sample_index=address, enabled=enabled)
                require(rng() == after and all(torch.equal(a, b) for a, b in zip(
                    batch.original_augmentation, original, strict=True)), "Original augmentation or RNG changed")
                expected_codes = tuple((0, 1, 2, 3, 2, 3, 2, 3)[(offset + i) % 8] for i in range(4))
                if not enabled:
                    expected_codes = (2, 2, 2, 2)
                require(batch.view_codes == expected_codes, "Absolute-address view cycle differs")
                for index, code in enumerate(batch.view_codes):
                    if code >= 2:
                        require(all(torch.equal(a[index], b[index]) for a, b in zip(
                            (batch.mixture, batch.targets, batch.vocal_derangement), original, strict=True)),
                            "Ordinary example or vocal-derangement flag changed")
                    else:
                        expected = targets[index].clone()
                        expected[[2] if code == 0 else [0, 1, 3]] = 0
                        require(torch.equal(batch.targets[index], expected)
                                and torch.equal(batch.mixture[index], expected.sum(dim=0))
                                and not bool(batch.vocal_derangement[index]), "Controlled physical truth differs")
                require(torch.equal(mixture, pristine[0]) and torch.equal(targets, pristine[1]), "Inputs mutated")
                if not enabled:
                    continue
                seed(value)
                old = old_augment(mixture, targets, first_sample_index=address, enabled=True)
                require(rng() == after, "Old and new augmentation RNG differ")
                if offset == 0:
                    require(all(torch.equal(a, b) for a, b in zip(
                        (batch.mixture, batch.targets, batch.vocal_derangement),
                        (old.mixture, old.targets, old.vocal_derangement), strict=True)),
                        "Shared first/third microbatch changed")

                raw = torch.randn(targets.shape, generator=generator).mul(.02).requires_grad_()
                deployed = torch.randn(targets.shape, generator=generator).mul(.02).requires_grad_()
                teacher = torch.randn(targets.shape, generator=generator).mul(.02)
                unchanged_rng = rng()
                terms = controlled_deployed_objective(raw, deployed, batch.targets, teacher,
                    batch.vocal_derangement, view_codes=batch.view_codes, weight=.5)
                base = terms.base
                micro = {"view_codes": list(batch.view_codes), "loss": float(terms.total.detach()),
                         "supervised_loss": float((base.waveform_l1 + base.projection_contribution).detach()),
                         "teacher_l1": float(base.teacher_l1.detach()),
                         **loss_evidence(raw, deployed, batch.targets, teacher, terms, batch.view_codes, .5)}
                validate_microbatch(micro, .5)
                rows.append(micro)
                original_loss = drum_emphasized_objective(raw, deployed, batch.targets, teacher,
                                                          batch.vocal_derangement)
                teacher_sum, truth_sum = raw.new_zeros(()), raw.new_zeros(())
                for i, code in enumerate(batch.view_codes):
                    for s, weight in enumerate((2, 1, 1, 1)):
                        if code >= 2:
                            teacher_sum = teacher_sum + weight * (deployed[i, s] - teacher[i, s]).abs().mean()
                        else:
                            truth_sum = truth_sum + weight * (deployed[i, s] - batch.targets[i, s]).abs().mean()
                expected = (original_loss.waveform_l1 + original_loss.projection_contribution
                            + .5 * teacher_sum / 20 + .5 * truth_sum / 20)
                gradients = torch.autograd.grad(terms.total, (raw, deployed), retain_graph=True)
                expected_gradients = torch.autograd.grad(expected, (raw, deployed), retain_graph=True)
                loss_error = abs(float((terms.total - expected).detach()))
                gradient_error = max(float((a - b).abs().max()) for a, b in zip(gradients, expected_gradients, strict=True))
                require(loss_error < 1e-7 and gradient_error < 1e-8
                        and all(bool(torch.isfinite(g).all()) for g in gradients), "Loss or gradient arithmetic differs")
                max_loss_error = max(max_loss_error, loss_error)
                max_gradient_error = max(max_gradient_error, gradient_error)
                if base.controlled_examples == 0:
                    original_gradients = torch.autograd.grad(original_loss.total, (raw, deployed), retain_graph=True)
                    require(terms.controlled_deployed_l1.item() == terms.controlled_deployed_contribution.item() == 0
                            and torch.equal(terms.total, original_loss.total)
                            and all(torch.equal(a, b) for a, b in zip(gradients, original_gradients, strict=True)),
                            "All-ordinary loss and gradients failed exact identity")
                zero = controlled_deployed_objective(raw, deployed, batch.targets, teacher,
                    batch.vocal_derangement, view_codes=batch.view_codes, weight=0)
                zero_micro = {**micro, "loss": float(zero.total.detach()),
                              **loss_evidence(raw, deployed, batch.targets, teacher, zero, batch.view_codes, 0)}
                validate_microbatch(zero_micro, 0)
                require(torch.equal(zero.total, zero.base.total) and rng() == unchanged_rng,
                        "Zero-weight identity or loss/backward RNG differs")
                cases.append({"seed": value, "address": address, "codes": list(batch.view_codes),
                              "controlled_examples": base.controlled_examples,
                              "ordinary_examples": base.ordinary_examples, "loss_error": loss_error,
                              "gradient_error": gradient_error})

    require(original_deranged > 0 and original_subset > 0, "Did not exercise original distribution branches")
    for offset in range(8):
        codes = [(0, 1, 2, 3, 2, 3, 2, 3)[(offset + i) % 8] for i in range(16)]
        require(codes.count(0) == codes.count(1) == 2 and sum(c >= 2 for c in codes) == 12,
                "Effective B=16 does not contain four controlled and twelve ordinary examples")
    aligned_counts = [sum((0, 1, 2, 3, 2, 3, 2, 3)[(start + i) % 8] < 2 for i in range(4))
                      for start in (0, 4, 8, 12)]
    require(aligned_counts == [2, 0, 2, 0], "Aligned accumulation has unexpected microbatch counts")
    fixture = next(row for row in rows if row["controlled_examples"] == 2)
    for key, value in (("controlled_examples", 0), ("ordinary_examples", 4),
                       ("view_codes", [False, 1, 2, 3]), ("view_codes", [0, 1, 2, 4]),
                       ("teacher_keep", [True] * 4), ("controlled_deployed_keep", [False] * 4),
                       ("teacher_l1", 2 * fixture["teacher_l1"]),
                       ("controlled_deployed_l1", 2 * fixture["controlled_deployed_l1"]),
                       ("loss", fixture["loss"] + .001)):
        bad = copy.deepcopy(fixture)
        bad[key] = value
        expect_rejected(key + ":" + str(value), bad)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "CPU proof touched CUDA")
    result = {"schema": plan["schema"], "status": "pass", "plan_sha256": sha(directory / "plan.json"),
              "source_bindings_unchanged": True, "cuda_initialized": False,
              "augmentation_cases": len(cases) * 2, "loss_gradient_cases": len(cases),
              "loss_functions_unchanged": True, "augmentation_only_changes": "view cycle and version",
              "journal_only_changes": "validate recorded variable counts; retain all divisor arithmetic",
              "effective_batch_size": 16, "microbatch_size": 4, "aligned_controlled_counts": aligned_counts,
              "controlled_per_update": 4, "ordinary_per_update": 12,
              "original_deranged_examples_exercised": original_deranged,
              "original_zeroed_stems_exercised": original_subset,
              "max_loss_error": max_loss_error, "max_gradient_error": max_gradient_error,
              "corrupted_journals_rejected": rejection_cases, "cases": cases,
              "training_updates": 0, "quality_scores": 0,
              "limitations": ["Synthetic CPU proof does not establish GPU determinism, model quality or runtime latency.",
                              "Changing view frequency also changes teacher participation and auxiliary-loss frequency.",
                              "Ordinary means the original augmentation, which includes subset and vocal-derangement examples."]}
    write(directory / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "corrupted_journals_rejected")}), flush=True)


if __name__ == "__main__":
    main()
