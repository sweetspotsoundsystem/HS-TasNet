"""Compare logged training microbatches with the unchanged parent on exact crops.

The parent runs in CPU FP32; logged training used GPU BF16. This is a training
diagnostic with an initial-weight precision control, not checkpoint validation.
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


def selected_rows(path, steps):
    rows = {row["step"]: row for line in path.read_text().splitlines() if line.strip()
            for row in [json.loads(line)]}
    require(all(step in rows for step in steps), "Requested updates have not completed")
    return [rows[step] for step in steps]


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    import torch
    from research.direct.latency58_full_magnitude_checkpoint import load_model
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_wave_spectral import augment, objective
    from research.direct.latency58_sdr_context import render_scored_context
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training_root = PHASE / "full-magnitude-001"
    training_plan = training_root / "plan.json"
    source = read(training_plan)
    verify_inputs(source)
    config = source["config"]
    out = PHASE / "full-magnitude-learning-001"
    require(not out.exists(), "Preserve completed diagnostics")
    require_space(source, 380_000_000)
    steps = [1, *range(25, 226, 25)]
    journal = training_root / "production-run/metrics.jsonl"
    selected = selected_rows(journal, steps)
    model, _ = load_model(source["parent_checkpoint"])
    parent_state = state_sha256(model.state_dict())
    require(parent_state == source["parent_model_state_sha256"], "Parent initialization differs")
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    source_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=source_config)
    require(manifest_sha == source["manifest_sha256"], "Corpus changed")
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=selected[-1]["next_sample_index"])
    out.mkdir()
    write(out / "selected-training-rows.json", selected)
    bindings = {**source["source_bindings"], str(training_plan): sha(training_plan),
                str(Path(__file__).resolve()): sha(__file__),
                str(out / "selected-training-rows.json"): sha(out / "selected-training-rows.json")}
    plan = {"source_bindings": bindings, "training_steps": steps, "examples_scored_per_step": 4,
            "all16_augmented_input_hash_verified": True, "parent_precision": "CPU FP32",
            "training_precision": "GPU BF16 learned operations and FP32 FFT/loss/state",
            "training_only": True, "validation_or_test_used": False, "checkpoint_quality_measured": False}
    write(out / "plan.json", plan)
    began, comparisons = time.monotonic(), []
    with (out / "progress.jsonl").open("x", buffering=1) as progress, torch.inference_mode():
        for row in selected:
            first = row["first_sample_index"]
            require(first == config["data_start"] + (row["step"] - 1) * 16, "Training index changed")
            inputs = [dataset[index] for index in range(first, first + 16)]
            mixture = torch.stack([x for x, _ in inputs])
            truth = torch.stack([y for _, y in inputs])
            mixture, truth, changed, factors = augment(mixture, truth, seed=config["seed"], first_sample_index=first)
            digest = hashlib.sha256()
            for offset in range(0, 16, 4):
                for tensor in (mixture, truth, changed, factors):
                    digest.update(tensor[offset:offset + 4].contiguous().numpy().tobytes())
            require(digest.hexdigest() == row["augmented_inputs_sha256"], "Matched training audio differs")
            rendered = render_scored_context(model, mixture[:4], warmup_samples=source["warmup_samples"], carry_state=True)
            terms = objective(rendered.raw, rendered.deployed, truth[:4, ..., source["warmup_samples"]:], rendered.physical_mixture)
            logged = row["microbatches"][0]
            require(terms.active_window_counts.tolist() == logged["active_windows"]
                    and terms.absent_window_counts.tolist() == logged["absent_windows"], "Training activity masks differ")
            result = {"step": row["step"], "parent_loss": float(terms.total), "logged_training_loss": logged["loss"],
                      "training_minus_parent_loss": logged["loss"] - float(terms.total),
                      "parent_sdr_db": -float(terms.negative_sdr_db), "logged_training_sdr_db": -logged["negative_sdr_db"],
                      "training_minus_parent_sdr_db": float(terms.negative_sdr_db) - logged["negative_sdr_db"],
                      "per_stem_training_minus_parent_sdr_db": [a - b for a, b in zip(
                          terms.per_stem_negative_sdr_db.tolist(), logged["per_stem_negative_sdr_db"], strict=True)],
                      "augmented_inputs_sha256": digest.hexdigest(), "elapsed_seconds": time.monotonic() - began}
            comparisons.append(result)
            progress.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            del inputs, mixture, truth, changed, factors, rendered, terms
    require(selected_rows(journal, steps) == selected and state_sha256(model.state_dict()) == parent_state
            and not torch.cuda.is_initialized(), "Journal, parent, or CPU scope changed")
    verify_inputs(plan)
    later = comparisons[1:]
    write(out / "result.json", {"status": "complete", "training_only": True, "checkpoint_quality_measured": False,
          "initial_weight_precision_control": comparisons[0], "comparisons": comparisons,
          "mean_later_training_minus_parent_loss": sum(r["training_minus_parent_loss"] for r in later) / len(later),
          "mean_later_training_minus_parent_sdr_db": sum(r["training_minus_parent_sdr_db"] for r in later) / len(later),
          "source_bindings_unchanged": True, "parent_unchanged": True, "matched_all16_input_hashes": True,
          "gpu_used": False, "elapsed_seconds": time.monotonic() - began,
          "limitation": "Different evolving training checkpoints and CPU/GPU precision; this cannot establish held-out SDR.",
          "counted_bytes_after": require_space(source, 370_000_000)})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
