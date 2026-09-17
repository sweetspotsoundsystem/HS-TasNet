"""Measure output-gradient alignment on a small, fixed set of training crops."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU diagnostic")
    out = PHASE / "ordinary-gradient-diagnostic-001"
    require(not out.exists(), "Preserve existing diagnostic")
    training_path = PHASE / "latency58-reduced-teacher-001/training-plan.json"
    plan = read(training_path)
    verify_inputs(plan)
    sources = {**plan["source_bindings"], str(training_path): sha(training_path),
               str(Path(__file__).resolve()): sha(Path(__file__))}
    from research.direct.latency58_reduced_teacher_checkpoint import load_parent, require_space
    require_space(plan, 1_000_000)
    import torch
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_vocal_focus_augmentation import augment_vocal_focus
    from research.direct.latency58_reduced_teacher_loss import objective
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = load_parent(plan).eval().requires_grad_(False)
    teacher, identity = load_teacher(plan["teacher_kind"], plan["teacher"])
    parent_state = state_sha256(model.state_dict())
    require(parent_state == plan["parent"]["model_state_sha256"], "Different diagnostic parent")
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"],
        config=read(PRODUCTION / "full_config.json"))
    require(manifest_sha == plan["manifest_sha256"], "Different training corpus")
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=plan["config"]["root_weights"],
        seed=plan["config"]["data_seed"], crop_samples=plan["config"]["crop_samples"],
        vocal_active_probability=plan["config"]["vocal_active_probability"], final_sample_index=980016)
    torch.manual_seed(20260923)
    out.mkdir()
    rows = []
    began = time.monotonic()
    for first in range(980000, 980016, 4):
        examples = [dataset[i] for i in range(first, first + 4)]
        mixture = torch.stack([x[0] for x in examples])
        targets = torch.stack([x[1] for x in examples])
        batch = augment_vocal_focus(mixture, targets, first_sample_index=first, enabled=False)
        mixture, targets, flags = batch.mixture, batch.targets, batch.vocal_derangement
        with torch.no_grad():
            output = render_scored_context(model, mixture, warmup_samples=plan["warmup_samples"], carry_state=True)
            teacher_targets = physical_context_teacher(teacher, mixture, kind=plan["teacher_kind"],
                                                       warmup_samples=plan["warmup_samples"])
        truth = targets[..., plan["warmup_samples"]:]
        physical = mixture[..., plan["warmup_samples"]:]
        require(torch.equal(output.raw[:, :3], output.deployed[:, :3]), "DBV output definition differs")
        fingerprint = hashlib.sha256(mixture.contiguous().numpy().tobytes() + targets.contiguous().numpy().tobytes()).hexdigest()
        row = {"first_sample_index": first, "augmented_batch_sha256": fingerprint,
               "deranged_examples": int(flags.sum()), "variants": {}}
        for name, coefficient, deployed_weight in (("teacher_half", .5, 0), ("teacher_quarter", .25, 0),
                                                   ("quarter_plus_ordinary_deployed_truth", .25, .5)):
            raw = output.raw.detach().clone().requires_grad_(True)
            deployed = torch.cat((raw[:, :3], physical[:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
            require(torch.equal(deployed.detach(), output.deployed), "Diagnostic output algebra differs")
            terms = objective(raw, deployed, truth, teacher_targets, flags,
                              view_codes=(2, 2, 2, 2), teacher_weight=coefficient)
            stem_weights = raw.new_tensor([2, 1, 1, 1])[None, :, None, None]
            deployed_l1 = ((deployed - truth).abs() * stem_weights).mean() * (4 / 5)
            loss = terms.total + deployed_weight * deployed_l1
            mse = (deployed - truth).square().mean()
            reference_gradient, = torch.autograd.grad(mse, raw, retain_graph=True)
            gradient, = torch.autograd.grad(loss, raw)
            require(bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(reference_gradient).all()),
                    "Nonfinite diagnostic gradient")
            gradient, reference_gradient = gradient[:, :3].double(), reference_gradient[:, :3].double()
            stats = {}
            for index, stem in enumerate(("drums", "bass", "vocals")):
                g, target_gradient = gradient[:, index], reference_gradient[:, index]
                dot = (g * target_gradient).sum()
                denominator = (g.square().sum() * target_gradient.square().sum()).sqrt()
                active = target_gradient != 0
                count = int(active.sum())
                stats[stem] = {"gradient_cosine_to_deployed_mse": float(dot / denominator) if denominator > 0 else None,
                               "opposing_scalar_gradients": int(((g * target_gradient < 0) & active).sum()),
                               "zero_scalar_gradients": int(((g == 0) & active).sum()),
                               "supported_scalar_gradients": count}
            row["variants"][name] = {"teacher_weight": coefficient, "ordinary_deployed_truth_weight": deployed_weight,
                                      "loss": float(loss.detach()), "deployed_mse": float(mse.detach()),
                                      "output_gradient_alignment": stats}
        rows.append(row)
        print(json.dumps({"event": "diagnostic_batch", "first_sample_index": first}), flush=True)
    require(state_sha256(model.state_dict()) == parent_state
            and state_sha256(teacher.state_dict()) == identity["model_state_sha256"]
            and all(p.grad is None for p in model.parameters()) and not torch.cuda.is_initialized(),
            "Diagnostic mutated model tensors or used CUDA")
    verify_inputs({"source_bindings": sources})
    write(out / "result.json", {"status": "pass", "source_bindings": sources, "source_bindings_unchanged": True,
          "parent_model_state_sha256": parent_state, "precision": "CPU FP32", "seed": 20260923,
          "sample_indices": [980000, 980016], "rows": rows, "cuda_initialized": False,
          "training_updates_executed": 0, "elapsed_seconds": time.monotonic() - began,
          "limitations": ["Sixteen training crops; no validation SDR or generalization result.",
              "Output-space gradients, not parameter-space updates; CPU augmentation and precision differ from GPU training.",
              "Raw Other has no direct influence on deployed residual Other; alignment covers the three deployed heads."]})
    print(json.dumps({"status": "pass", "result": str(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
