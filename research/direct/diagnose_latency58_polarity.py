"""Measure polarity sensitivity on recorded training crops without fitting weights."""
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
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    from research.direct.latency58_direct_sdr import objective
    from research.direct.latency58_musdb_sdr_data import selection_contract, select_tracks
    from research.direct.latency58_sdr_context import render_scored_context
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = PHASE / "polarity-training-diagnostic-001"
    require(not out.exists(), "Preserve prior diagnostics")
    budget = read(PHASE / "magnitude-sdr-001/plan.json")
    require_space(budget, 1_000_000)
    binding = read(PHASE / "c204-residual-model-001/checkpoint.json")
    parent, _ = load_model(binding)
    parent_state = state_sha256(parent.state_dict())
    contract = selection_contract()
    bindings = dict(read(PHASE / "c204-residual-model-001/qualification.json")["source_bindings"])
    for path in (Path(__file__).resolve(), ROOT / "research/direct/latency58_direct_sdr.py",
                 ROOT / "research/direct/latency58_direct_sdr_checkpoint.py",
                 ROOT / "research/direct/latency58_musdb_sdr_data.py", ROOT / "research/direct/latency58_sdr_context.py",
                 PRODUCTION / "train_production.py", PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json",
                 Path(binding["path"])):
        bindings[str(path)] = sha(path)
    plan = {"schema": "latency58-polarity-training-diagnostic-v1", "checkpoint": binding,
            "parent_model_state_sha256": parent_state, "training_selection": contract,
            "sample_indices": list(range(1_600_000, 1_600_032)), "data_seed": 60,
            "crop_samples": 176384, "warmup_samples": 88064, "vocal_active_probability": .85,
            "cpu_original_batch_size": 4, "variants": ["original", "negative_inverted", "fixed_half_average"],
            "coefficients_searched": False, "weights_fitted": False, "validation_or_test_audio_used": False,
            "scope": "Training-only diagnostic of a prospective polarity augmentation; no deployment or validation claim",
            "source_bindings": bindings, "output_directory": str(out)}
    verify_inputs(plan)
    out.mkdir()
    write(out / "plan.json", plan)
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=contract["source_manifest_sha256"], config=config)
    require(manifest_hash == contract["source_manifest_sha256"] and config["seed"] == plan["data_seed"], "Corpus differs")
    dataset = production.CounterAddressedCropDataset(select_tracks(tracks, contract), root_weights={"musdb18hq_train": 1.},
        seed=plan["data_seed"], crop_samples=plan["crop_samples"], vocal_active_probability=plan["vocal_active_probability"],
        final_sample_index=max(plan["sample_indices"]) + 1)
    accumulated = {name: {"sdr_sum": torch.zeros(4, dtype=torch.float64), "counts": torch.zeros(4, dtype=torch.int64)}
                   for name in plan["variants"]}
    began = time.monotonic()
    with (out / "progress.jsonl").open("x", buffering=1) as journal, torch.inference_mode():
        for first in range(0, len(plan["sample_indices"]), plan["cpu_original_batch_size"]):
            indices = plan["sample_indices"][first:first + plan["cpu_original_batch_size"]]
            examples = [dataset[index] for index in indices]
            mixture = torch.stack([x for x, y in examples])
            truth = torch.stack([y for x, y in examples])
            digest = hashlib.sha256()
            for value in (mixture, truth):
                digest.update(memoryview(value.contiguous().numpy()).cast("B"))
            output = render_scored_context(parent, torch.cat((mixture, -mixture)),
                warmup_samples=plan["warmup_samples"], carry_state=True)
            original, negative = output.deployed.chunk(2)
            physical = mixture[..., plan["warmup_samples"]:]
            targets = truth[..., plan["warmup_samples"]:]
            require(torch.equal(output.physical_mixture[:len(indices)], physical)
                    and torch.equal(output.physical_mixture[len(indices):], -physical), "Physical crop alignment differs")
            predictions = {"original": original, "negative_inverted": -negative, "fixed_half_average": .5 * (original - negative)}
            scores = {}
            for name, prediction in predictions.items():
                require(float((prediction.sum(1) - physical).abs().max()) < 1e-6, "Stem sum differs from physical mixture")
                terms = objective(prediction, prediction, targets, physical)
                counts = terms.active_window_counts
                sdr = -terms.per_stem_negative_sdr_db
                accumulated[name]["sdr_sum"] += sdr.double() * counts
                accumulated[name]["counts"] += counts
                scores[name] = {"per_stem_sdr_db": sdr.tolist(), "active_windows": counts.tolist()}
            row = {"completed_examples": first + len(indices), "sample_indices": indices,
                   "input_sha256": digest.hexdigest(), "scores": scores, "elapsed_seconds": time.monotonic() - began}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({k: row[k] for k in ("completed_examples", "elapsed_seconds")}), flush=True)
    summaries = {}
    base = accumulated["original"]["sdr_sum"] / accumulated["original"]["counts"].clamp_min(1)
    for name, values in accumulated.items():
        stems = values["sdr_sum"] / values["counts"].clamp_min(1)
        summaries[name] = {"per_stem_sdr_db": stems.tolist(), "full_sdr_db": float(stems.mean()),
                           "per_stem_delta_db": (stems - base).tolist(), "full_sdr_delta_db": float((stems - base).mean()),
                           "active_windows": values["counts"].tolist()}
    verify_inputs(plan)
    require(state_sha256(parent.state_dict()) == parent_state and not torch.cuda.is_initialized(), "Model or CPU scope changed")
    result = {"status": "complete", "scope": plan["scope"], "summaries": summaries,
              "model_unchanged": True, "source_bindings_unchanged": True, "gpu_used": False,
              "validation_or_test_audio_used": False, "weights_fitted": False,
              "inference_compute_note": "The fixed average uses two independent recurrent executions; it is only a diagnostic here.",
              "elapsed_seconds": time.monotonic() - began, "plan_sha256": sha(out / "plan.json")}
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
