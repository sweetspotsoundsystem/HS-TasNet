"""Compare reconstruction and SDR gradients in native-output coordinates.

Use four complete, predetermined training batches from the complex-mask pilot
and its frozen quadrature parent. This does not backpropagate through neural
parameters, train a model, use validation audio, or estimate held-out quality.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs
from research.direct.diagnose_latency58_quadrature_continuation_learning import selected_rows


def main():
    import torch
    from torch.nn import functional as F
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_remix_augmentation import augment
    from research.direct.latency58_wave_spectral import objective as reconstruction
    from research.direct.latency58_direct_sdr import objective as direct_sdr
    from research.direct.latency58_residual_model import corrected_estimates
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the CPU1 workspace")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training = PHASE / "complex-mask-001"
    training_path = training / "plan.json"
    source = read(training_path)
    verify_inputs(source)
    config = source["config"]
    require(source["schema"] == "latency58-complex-mask-training-plan-v1"
            and config["batch_size"] == config["microbatch_size"] == 16
            and source["warmup_samples"] == 88064 and source["scored_samples"] == 44160,
            "Unexpected training geometry")
    # This diagnostic is deliberately scheduled well before the terminal save.
    require(not (training / "production-run/checkpoint").exists()
            and not (training / "production-run/checkpoint.pending").exists(), "Run this diagnostic during early training")
    require_space(source, 382_000_000)
    steps = [1, 100, 200, 300]
    journal = training / "production-run/metrics.jsonl"
    selected = selected_rows(journal, steps)
    model, _ = load_model(source["parent_checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == source["parent_model_state_sha256"], "Frozen parent changed")
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    corpus = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=corpus)
    require(manifest_sha == source["manifest_sha256"], "Training corpus changed")
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=selected[-1]["next_sample_index"])
    out = PHASE / "quadrature-objective-alignment-001"
    require(not out.exists(), "Preserve earlier diagnostics")
    out.mkdir()
    write(out / "selected-training-rows.json", selected)
    paths = [training_path, Path(__file__).resolve(), out / "selected-training-rows.json",
             ROOT / "research/direct/diagnose_latency58_quadrature_continuation_learning.py",
             PHASE / "complex-mask-rationale-correction-001/result.json"]
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    plan = {"source_bindings": bindings, "training_steps": steps, "examples_per_batch": 16,
            "selection": "Initial update and every hundredth update through 300, independent of losses",
            "parent_checkpoint": source["parent_checkpoint"], "precision": "CPU FP32",
            "gradient_coordinate": "Native four-source output, before the unchanged discrepancy correction",
            "neural_parameter_gradients_measured": False, "training_only": True,
            "validation_or_test_used": False, "checkpoint_quality_measured": False,
            "pending_training_reservation_bytes": 380_000_000}
    write(out / "plan.json", plan)
    began, rows = time.monotonic(), []
    def cosine(a, b):
        # FP64 reduction keeps this diagnostic's dot products reproducible.
        left, right = a.double().flatten(), b.double().flatten()
        denominator = left.norm() * right.norm()
        return None if denominator == 0 else float(torch.dot(left, right) / denominator)
    with (out / "progress.jsonl").open("x", buffering=1) as progress:
        for logged in selected:
            first = logged["first_sample_index"]
            require(first == config["data_start"] + (logged["step"] - 1) * 16, "Training index differs")
            inputs = [dataset[i] for i in range(first, first + 16)]
            mixture, truth = torch.stack([x for x, _ in inputs]), torch.stack([y for _, y in inputs])
            mixture, truth, changed, factors = augment(mixture, truth, seed=config["seed"], first_sample_index=first)
            digest = hashlib.sha256()
            for value in (mixture, truth, changed, factors):
                digest.update(value.contiguous().numpy().tobytes())
            require(digest.hexdigest() == logged["augmented_inputs_sha256"], "Training audio hash differs")
            with torch.no_grad():
                state = model.warm_state(mixture[..., :88064]).detached()
                physical = mixture[..., 88064:]
                rendered = model.render(F.pad(physical, (0, 128)), state)
                native = rendered.native_raw[..., 128:128 + 44160].clone()
                require(torch.equal(rendered.delayed_mixture[..., 128:128 + 44160], physical), "Physical alignment differs")
            native.requires_grad_()
            raw, deployed = corrected_estimates(native, physical, model.fixed_residual_share)
            require(torch.equal(raw, rendered.raw[..., 128:128 + 44160])
                    and torch.equal(deployed, rendered.deployed[..., 128:128 + 44160]),
                    "Native-output reconstruction changed the saved parent")
            targets = truth[..., 88064:]
            recon = reconstruction(raw, deployed, targets, physical)
            direct = direct_sdr(raw, deployed, targets, physical)
            require(direct.active_window_counts.tolist() == logged["microbatches"][0]["active_windows"]
                    and direct.absent_window_counts.tolist() == logged["microbatches"][0]["absent_windows"],
                    "Training activity rule differs")
            def gradient(value):
                grad, = torch.autograd.grad(value, native, retain_graph=True)
                require(bool(torch.isfinite(grad).all()), "Nonfinite output gradient")
                return grad.detach()
            combined = gradient(recon.total)
            primary = gradient(direct.negative_sdr_db)
            components = {name: cosine(gradient(value), primary) for name, value in
                          (("waveform", recon.waveform), ("complex_stft", recon.spectral), ("raw_anchor", recon.raw_anchor))}
            stem_alignment = {stem: cosine(combined, gradient(direct.per_stem_negative_sdr_db[i]))
                              for i, stem in enumerate(("drums", "bass", "vocals", "other"))}
            result = {"step": logged["step"], "augmented_inputs_sha256": digest.hexdigest(),
                      "reconstruction_loss": float(recon.total.detach()),
                      "parent_training_batch_sdr_db": -float(direct.negative_sdr_db.detach()),
                      "reconstruction_vs_primary_sdr_cosine": cosine(combined, primary),
                      "reconstruction_vs_direct_total_cosine": cosine(combined, gradient(direct.total)),
                      "component_vs_primary_sdr_cosine": components,
                      "reconstruction_vs_each_stem_sdr_cosine": stem_alignment,
                      "active_windows": direct.active_window_counts.tolist(),
                      "absent_windows": direct.absent_window_counts.tolist(),
                      "elapsed_seconds": time.monotonic() - began}
            rows.append(result)
            progress.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps(result, allow_nan=False), flush=True)
            del inputs, mixture, truth, changed, factors, state, physical, rendered, native, raw, deployed
            del targets, recon, direct, combined, primary
    require(selected_rows(journal, steps) == selected and state_sha256(model.state_dict()) == fingerprint
            and all(p.grad is None for p in model.parameters()) and not torch.cuda.is_initialized(),
            "Journal, frozen parent or CPU scope changed")
    verify_inputs(plan)
    write(out / "result.json", {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
          "comparisons": rows, "all_64_training_example_hashes_verified": True,
          "native_output_reconstruction_bit_exact": True, "neural_parameters_and_gradients_unchanged": True,
          "training_only": True, "checkpoint_quality_measured": False, "gpu_used": False,
          "elapsed_seconds": time.monotonic() - began,
          "limitation": "Local gradients at the saved parent in output space. Neural-parameter updates can differ after the model Jacobian, optimizer and clipping; this is not a training or validation outcome."})
    require_space(source, 382_000_000)
    print(json.dumps({"status": "pass", "batches": len(rows), "result_sha256": sha(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
