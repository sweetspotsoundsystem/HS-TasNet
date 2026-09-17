"""Measure saved-parent attention on four predetermined training batches.

CPU inference only: record attention concentration, lag mass and correction
magnitude. No validation audio, training update or checkpoint write is used.
"""
from __future__ import annotations

import hashlib
import json
import math
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
    from research.direct.latency58_temporal_attention_checkpoint import load_model
    from research.direct.latency58_temporal_attention import WINDOW, KEY
    from research.direct.latency58 import EMBED
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_remix_augmentation import augment
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU inference with one numerical thread")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training = PHASE / "attention-continuation-001"
    source_path = training / "plan.json"
    source = read(source_path)
    verify_inputs(source)
    config = source["config"]
    require(source["parent_kind"] == "saved_temporal_attention"
            and source["parent_training_updates"] == 21250
            and source["inference_architecture_changed"] is False
            and config["batch_size"] == config["microbatch_size"] == 16
            and source["warmup_samples"] == 88064 and source["scored_samples"] == 44160,
            "Use the frozen selected parent and the complete training geometry")
    require(not (training / "production-run/checkpoint").exists()
            and not (training / "production-run/checkpoint.pending").exists(),
            "Reserve this diagnostic for early training")
    require_space(source, 382_000_000)
    steps = [1, 100, 200, 300]
    journal = training / "production-run/metrics.jsonl"
    selected = selected_rows(journal, steps)
    model, _ = load_model(source["parent_checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == source["parent_model_state_sha256"] and not model.training,
            "Use the unchanged saved parent in evaluation mode")
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
    out = PHASE / "attention-parent-activity-001"
    require(not out.exists(), "Preserve existing diagnostics")
    out.mkdir()
    write(out / "selected-training-rows.json", selected)
    paths = [source_path, Path(__file__).resolve(), out / "selected-training-rows.json",
             ROOT / "research/direct/diagnose_latency58_quadrature_continuation_learning.py"]
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    plan = {"source_bindings": bindings, "training_steps": steps, "examples_per_batch": 16,
            "selection": "Initial update and every hundredth update through 300, independent of losses",
            "parent_checkpoint": source["parent_checkpoint"], "precision": "CPU FP32",
            "training_only": True, "validation_or_test_used": False,
            "checkpoint_quality_measured": False, "pending_training_reservation_bytes": 380_000_000}
    write(out / "plan.json", plan)
    original_attention = model.attention
    observations = []

    def describe(values):
        values = values.double().flatten()
        require(bool(torch.isfinite(values).all()), "Nonfinite attention statistic")
        return {"mean": float(values.mean()), "p05": float(torch.quantile(values, .05)),
                "p50": float(torch.quantile(values, .5)), "p95": float(torch.quantile(values, .95))}

    def observe(fused, past_keys, past_values, *, tail_only=False):
        result = original_attention(fused, past_keys, past_values, tail_only=tail_only)
        # Reconstruct the explicit attention arithmetic without changing the
        # original return values or the saved module's parameter tensors.
        queries = model.temporal_query(fused[:, -1:] if tail_only else fused)
        keys = torch.cat((past_keys, model.temporal_key(fused).float()), dim=1)
        values = torch.cat((past_values, model.temporal_value(fused).float()), dim=1)
        key_windows = (keys[:, -WINDOW:].unsqueeze(1) if tail_only
                       else keys.unfold(1, WINDOW, 1).transpose(-1, -2))
        value_windows = (values[:, -WINDOW:].unsqueeze(1) if tail_only
                         else values.unfold(1, WINDOW, 1).transpose(-1, -2))
        logits = (queries.float().unsqueeze(-2) * key_windows).sum(-1) * (KEY ** -.5)
        weights = torch.softmax(logits, dim=-1)
        correction = model.temporal_output((weights.unsqueeze(-1) * value_windows).sum(-2))
        require(torch.equal(correction, result[0])
                and torch.equal(keys[:, -(WINDOW - 1):], result[1])
                and torch.equal(values[:, -(WINDOW - 1):], result[2]),
                "Diagnostic arithmetic must reproduce the saved attention exactly")
        if not tail_only:
            entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
            maximum = weights.max(-1).values
            relative_rms = lambda x, y: (x.square().mean(-1) / y.square().mean(-1).clamp_min(1e-20)).sqrt()
            lags = torch.arange(WINDOW - 1, -1, -1, dtype=weights.dtype)
            observations.append({"frames_per_example_including_flush": int(fused.shape[1]),
                "normalized_entropy": describe(entropy / math.log(WINDOW)),
                "effective_attended_frames": describe(entropy.exp()),
                "maximum_frame_mass": describe(maximum),
                "fraction_maximum_mass_above_half": float((maximum > .5).double().mean()),
                "fraction_maximum_mass_above_nine_tenths": float((maximum > .9).double().mean()),
                "mean_mass_by_lag_oldest_to_current": weights.double().mean((0, 1)).tolist(),
                "expected_past_lag_frames": describe((weights * lags).sum(-1)),
                "within_frame_logit_std": describe(logits.std(-1, correction=0)),
                "correction_to_refined_feature_rms": describe(relative_rms(correction, fused)),
                "spectral_half_correction_to_refined_rms": describe(relative_rms(correction[..., :EMBED], fused[..., :EMBED])),
                "waveform_half_correction_to_refined_rms": describe(relative_rms(correction[..., EMBED:], fused[..., EMBED:]))})
        return result

    began, rows = time.monotonic(), []
    model.attention = observe
    try:
        with torch.no_grad(), (out / "progress.jsonl").open("x", buffering=1) as progress:
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
                observations.clear()
                state = model.warm_state(mixture[..., :88064]).detached()
                physical = mixture[..., 88064:]
                rendered = model.render(F.pad(physical, (0, 128)), state)
                require(torch.equal(rendered.delayed_mixture[..., 128:128 + 44160], physical)
                        and len(observations) == 1, "Physical alignment or scored observation count differs")
                row = {"step": logged["step"], "augmented_inputs_sha256": digest.hexdigest(),
                       **observations[0], "elapsed_seconds": time.monotonic() - began}
                rows.append(row)
                progress.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps({"step": row["step"], "entropy_mean": row["normalized_entropy"]["mean"],
                      "effective_frames_mean": row["effective_attended_frames"]["mean"],
                      "correction_relative_rms_mean": row["correction_to_refined_feature_rms"]["mean"]}), flush=True)
                del inputs, mixture, truth, changed, factors, state, physical, rendered
    finally:
        del model.attention
    require(selected_rows(journal, steps) == selected and state_sha256(model.state_dict()) == fingerprint
            and all(parameter.grad is None for parameter in model.parameters()) and not torch.cuda.is_initialized(),
            "Journal, saved parent, gradients or CPU scope changed")
    verify_inputs(plan)
    require_space(source, 382_000_000)
    write(out / "result.json", {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
          "observations": rows, "all_four_augmented_batch_hashes_verified": True,
          "training_examples_covered_by_batch_hashes": 64,
          "reconstructed_attention_and_cache_outputs_bit_exact": True,
          "saved_parent_parameters_and_gradients_unchanged": True, "instrumentation_removed": True,
          "training_only": True, "checkpoint_quality_measured": False, "gpu_used": False,
          "elapsed_seconds": time.monotonic() - began,
          "limitation": "Four training batches at the saved parent. Attention concentration and correction size do not establish causal usefulness, generalization or the quality of an alternative architecture. Scored-frame summaries include one flush frame."})
    print(json.dumps({"status": "pass", "batches": len(rows), "result_sha256": sha(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
