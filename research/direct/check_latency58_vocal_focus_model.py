"""CPU identity and gradient checks for the explicit mixer model class."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Model-class fixture plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-model-functional-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Require CUDA-hidden CPU1")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve model-class fixture")
    import torch
    from research.direct.latency58_drum_accum_parent import load_parent
    from research.direct.latency58_vocal_focus_model import LocalMaskMixerModel, VERSION, parent_parameter_name
    from research.direct.latency58_vocal_focus_augmentation import augment_vocal_focus
    from research.direct.latency58_sdr_context import render_scored_context
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260921)
    torch.use_deterministic_algorithms(True)
    binding = plan["parent_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Preparation parent changed")
    parent_plan = read(binding["path"])
    verify_inputs(parent_plan)
    parent = load_parent(parent_plan).eval().requires_grad_(True)
    parent_fingerprint = state_sha256(parent.state_dict())
    rng = torch.get_rng_state().clone()
    before, began = require_space(plan, 2_000_000), time.monotonic()
    model = LocalMaskMixerModel.from_parent(parent, initialization_seed=20260921).eval().requires_grad_(True)
    repeated = LocalMaskMixerModel.from_parent(parent, initialization_seed=20260921)
    initial_fingerprint = state_sha256(model.state_dict())
    require(initial_fingerprint == state_sha256(repeated.state_dict())
            and torch.equal(torch.get_rng_state(), rng), "Initialization is not reproducible or changed caller RNG")
    del repeated
    metadata = model.architecture_metadata
    require(metadata["version"] == VERSION and metadata["spectral_mask_mixer"]["added_parameters"] == 536
            and all(metadata[key] == parent.architecture_metadata[key] for key in
                ("state_family", "state_names", "feature_history_samples", "graph_alignment_samples",
                 "host_queue_samples", "intended_total_latency_samples", "public_fusion_state_scale")),
            "Model metadata loses its extension or changes streaming geometry")
    generator = torch.Generator().manual_seed(20260921)
    targets = torch.randn(4, 4, 2, 1536, generator=generator) * .02
    targets[3, 2] = 0
    batch = augment_vocal_focus(targets.sum(dim=1), targets, first_sample_index=972000, enabled=True)
    teacher = (batch.targets[..., 512:] + torch.randn(4, 4, 2, 1024, generator=generator) * .001).detach()
    fixture_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        original = parent.render(batch.mixture)
        zero = model.render(batch.mixture)
        require(all(torch.equal(getattr(original, field), getattr(zero, field)) for field in
                    ("raw", "deployed", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(x, y) for x, y in zip(original.state, zero.state, strict=True)),
                "Explicit zero-initialized model differs from its preparation parent")

    def gradients(current):
        current.zero_grad(set_to_none=True)
        output = render_scored_context(current, batch.mixture, warmup_samples=512, carry_state=True)
        terms = drum_emphasized_objective(output.raw, output.deployed, batch.targets[..., 512:],
                                         teacher, batch.vocal_derangement)
        terms.total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in current.parameters()),
                "Missing or nonfinite model-class gradient")
        return float(terms.total.detach())

    parent_loss, zero_loss = gradients(parent), gradients(model)
    parent_parameters = dict(parent.named_parameters())
    gradient_differences = {}
    for name, parameter in model.named_parameters():
        if not name.startswith("to_spec_masks.mixer."):
            reference = parent_parameters[parent_parameter_name(name)].grad
            error = float((parameter.grad - reference).abs().max())
            require(torch.equal(parameter.grad, reference), "Zero mixer changed an inherited CPU gradient: " + name)
            gradient_differences[name] = error
    initial_gradients = {name: float(p.grad.abs().max()) for name, p in model.to_spec_masks.mixer.named_parameters()}
    require(parent_loss == zero_loss and initial_gradients["local.weight"] == initial_gradients["local.bias"] == 0
            and initial_gradients["out.weight"] > 0 and initial_gradients["out.bias"] > 0,
            "Zero mixer loss or expected staged branch gradients differ")
    require(state_sha256(model.state_dict()) == initial_fingerprint, "Backward changed initialized weights")
    with torch.no_grad():
        model.to_spec_masks.mixer.out.weight.copy_(
            torch.randn(model.to_spec_masks.mixer.out.weight.shape, generator=generator) * .003)
    active_fingerprint = state_sha256(model.state_dict())
    active_loss = gradients(model)
    active_gradients = {name: float(p.grad.abs().max()) for name, p in model.to_spec_masks.mixer.named_parameters()}
    require(all(value > 0 for value in active_gradients.values())
            and state_sha256(model.state_dict()) == active_fingerprint
            and state_sha256(parent.state_dict()) == parent_fingerprint
            and torch.equal(torch.get_rng_state(), fixture_rng) and not torch.cuda.is_initialized(),
            "Active fixture lost gradients or changed weights, RNG or CPU scope")
    verify_inputs(plan)
    result = {"schema": "latency58-vocal-focus-model-functional-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "architecture": metadata,
              "preparation_parent_state_sha256": parent_fingerprint,
              "initialized_model_state_sha256": initial_fingerprint, "initialization_seed": 20260921,
              "zero_output_and_state_exact": True, "initialization_replay_exact": True,
              "inherited_gradient_max_absolute_errors": gradient_differences,
              "parent_loss": parent_loss, "zero_loss": zero_loss, "active_fixture_loss": active_loss,
              "initial_mixer_gradient_maxima": initial_gradients, "active_mixer_gradient_maxima": active_gradients,
              "elapsed_seconds": time.monotonic() - began, "counted_bytes_before": before,
              "counted_bytes_after": require_space(plan, 0), "cuda_initialized": False,
              "training_updates_executed": 0, "weights_saved": False, "quality_selected": False,
              "limitations": ["Preparation parent and short synthetic crops with synthetic teacher references; no training-parent choice.",
                              "FP32 CPU gradient semantics only; full-crop BF16 GPU rehearsal remains required.",
                              "Fixed nonzero test coefficients are not trained candidate weights."]}
    write(out / "result.json", result)
    print({"status": "pass", "inherited_gradients_exact": len(gradient_differences),
           "initialized_model_state_sha256": initial_fingerprint}, flush=True)


if __name__ == "__main__":
    main()
