"""Compare saved attention with three local ablations on fixed training audio.

This CPU diagnostic changes only the temporary attention correction. It records
training reconstruction loss and SDR, verifies all stream states are unchanged,
and never writes a model or evaluates the held-out validation panel.
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

VARIANTS = ("saved_attention", "disabled_correction", "uniform_history", "current_frame_only")


def main():
    import torch
    from torch.nn import functional as F
    from research.direct.latency58 import SOURCE_ORDER
    from research.direct.latency58_temporal_attention_checkpoint import load_model
    from research.direct.latency58_temporal_attention import WINDOW
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_remix_augmentation import augment
    from research.direct.latency58_wave_spectral import objective
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
    activity_root = PHASE / "attention-parent-activity-001"
    activity = read(activity_root / "result.json")
    activity_execution_path = PHASE / "attention-parent-activity-stage-001/execution.json"
    activity_execution = read(activity_execution_path)
    require(activity["status"] == "pass" and activity["source_bindings_unchanged"]
            and activity["reconstructed_attention_and_cache_outputs_bit_exact"]
            and activity_execution["actual_exit_code"] == 0 and not activity_execution["timed_out"]
            and activity_execution["source_bindings_unchanged"], "Require the completed attention diagnostic")
    verify_inputs(activity)
    require(source["parent_kind"] == "saved_temporal_attention"
            and source["parent_training_updates"] == 21250
            and config["batch_size"] == config["microbatch_size"] == 16
            and source["warmup_samples"] == 88064 and source["scored_samples"] == 44160,
            "Use the selected parent and complete training geometry")
    require(not (training / "production-run/checkpoint").exists()
            and not (training / "production-run/checkpoint.pending").exists(), "Run during early training")
    require_space(source, 383_000_000)
    steps = [1, 100, 200, 300]
    journal = training / "production-run/metrics.jsonl"
    selected = selected_rows(journal, steps)
    require(selected == read(activity_root / "selected-training-rows.json"), "Use the same fixed training batches")
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
    out = PHASE / "attention-parent-ablation-001"
    require(not out.exists(), "Preserve existing ablations")
    out.mkdir()
    write(out / "selected-training-rows.json", selected)
    paths = [source_path, Path(__file__).resolve(), out / "selected-training-rows.json",
             activity_root / "result.json", activity_root / "selected-training-rows.json", activity_execution_path,
             ROOT / "research/direct/diagnose_latency58_quadrature_continuation_learning.py"]
    bindings = {**source["source_bindings"], **activity["source_bindings"],
                **{str(path): sha(path) for path in paths}}
    plan = {"source_bindings": bindings, "training_steps": steps, "examples_per_batch": 16,
            "variants": VARIANTS, "parent_checkpoint": source["parent_checkpoint"],
            "training_only": True, "validation_or_test_used": False, "checkpoint_quality_measured": False,
            "precision": "CPU FP32", "pending_training_reservation_bytes": 380_000_000,
            "selection": "Same four batches as the closed attention-activity diagnostic; no batch reselection"}
    write(out / "plan.json", plan)
    original_attention = model.attention

    def ablated_attention(variant):
        def forward(fused, past_keys, past_values, *, tail_only=False):
            result = original_attention(fused, past_keys, past_values, tail_only=tail_only)
            if variant == "disabled_correction":
                correction = torch.zeros_like(result[0])
            elif variant == "uniform_history":
                values = torch.cat((past_values, model.temporal_value(fused).float()), dim=1)
                windows = (values[:, -WINDOW:].unsqueeze(1) if tail_only
                           else values.unfold(1, WINDOW, 1).transpose(-1, -2))
                correction = model.temporal_output(windows.mean(-2))
            elif variant == "current_frame_only":
                correction = model.temporal_output(model.temporal_value(fused[:, -1:] if tail_only else fused).float())
            else:
                raise ValueError("Unknown diagnostic ablation")
            require(correction.shape == result[0].shape and bool(torch.isfinite(correction).all()),
                    "Ablated correction geometry or finite values differ")
            return correction, result[1], result[2]
        return forward

    began, rows = time.monotonic(), []
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
            state = model.warm_state(mixture[..., :88064]).detached()
            state_digest = state_sha256(state._asdict())
            physical, targets = mixture[..., 88064:], truth[..., 88064:]
            baseline_state, baseline_output = None, None
            variants = {}
            for variant in VARIANTS:
                require(state_sha256(state._asdict()) == state_digest, "A variant mutated its common initial state")
                if variant != "saved_attention":
                    model.attention = ablated_attention(variant)
                try:
                    rendered = model.render(F.pad(physical, (0, 128)), state)
                finally:
                    if variant != "saved_attention":
                        del model.attention
                require(torch.equal(rendered.delayed_mixture[..., 128:128 + 44160], physical),
                        "Physical alignment differs")
                raw = rendered.raw[..., 128:128 + 44160]
                deployed = rendered.deployed[..., 128:128 + 44160]
                require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(deployed).all()),
                        "A diagnostic output is nonfinite")
                if baseline_state is None:
                    baseline_state = tuple(value.clone() for value in rendered.state)
                    baseline_output = deployed.clone()
                require(all(torch.equal(a, b) for a, b in zip(baseline_state, rendered.state, strict=True)),
                        "A local attention ablation changed the stream states")
                terms = objective(raw, deployed, targets, physical)
                row = {"reconstruction_loss": float(terms.total), "waveform_loss": float(terms.waveform),
                       "spectral_loss": float(terms.spectral), "raw_anchor_loss": float(terms.raw_anchor),
                       "training_batch_sdr_db": -float(terms.negative_sdr_db),
                       "per_stem_training_sdr_db": dict(zip(SOURCE_ORDER,
                           (-terms.per_stem_negative_sdr_db).tolist(), strict=True)),
                       "max_output_abs_change": float((deployed - baseline_output).abs().max()),
                       "all_six_stream_states_bit_exact": True,
                       "active_windows": terms.active_window_counts.tolist(),
                       "absent_windows": terms.absent_window_counts.tolist()}
                if variants:
                    baseline = variants["saved_attention"]
                    row["reconstruction_loss_delta_vs_saved"] = row["reconstruction_loss"] - baseline["reconstruction_loss"]
                    row["training_sdr_delta_vs_saved_db"] = row["training_batch_sdr_db"] - baseline["training_batch_sdr_db"]
                    row["per_stem_training_sdr_delta_vs_saved_db"] = {
                        name: row["per_stem_training_sdr_db"][name] - baseline["per_stem_training_sdr_db"][name]
                        for name in SOURCE_ORDER}
                    require(row["active_windows"] == baseline["active_windows"]
                            and row["absent_windows"] == baseline["absent_windows"], "Reference activity rule changed")
                variants[variant] = row
                print(json.dumps({"step": logged["step"], "variant": variant,
                    "training_sdr_db": row["training_batch_sdr_db"],
                    "sdr_delta_vs_saved_db": row.get("training_sdr_delta_vs_saved_db")}), flush=True)
                del rendered, raw, deployed, terms
            item = {"step": logged["step"], "augmented_inputs_sha256": digest.hexdigest(), "variants": variants,
                    "elapsed_seconds": time.monotonic() - began}
            rows.append(item)
            progress.write(json.dumps(item, allow_nan=False) + "\n")
            del inputs, mixture, truth, changed, factors, state, physical, targets, baseline_state, baseline_output
    require(selected_rows(journal, steps) == selected and state_sha256(model.state_dict()) == fingerprint
            and all(parameter.grad is None for parameter in model.parameters()) and not torch.cuda.is_initialized()
            and "attention" not in model.__dict__, "Journal, saved model, gradients or CPU scope changed")
    verify_inputs(plan)
    require_space(source, 383_000_000)
    write(out / "result.json", {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
          "comparisons": rows, "all_four_augmented_batch_hashes_verified": True,
          "training_examples_covered_by_batch_hashes": 64, "all_six_stream_states_unchanged_across_variants": True,
          "saved_parent_parameters_and_gradients_unchanged": True, "instrumentation_removed": True,
          "training_only": True, "checkpoint_quality_measured": False, "gpu_used": False,
          "elapsed_seconds": time.monotonic() - began,
          "limitation": "Local inference ablations of a saved parent on four training batches. No alternative is retrained, no checkpoint is saved, and these scores do not establish validation quality or select a deployment model."})
    print(json.dumps({"status": "pass", "batches": len(rows), "result_sha256": sha(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
