"""Measure learned long-history contribution on fresh training-corpus crops.

This is a training diagnostic, separate from the unchanged full14 validation.
The zero-projection view is an intervention on the trained model, not a model
trained without the long-history features.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    import torch
    from research.direct.latency58_long_magnitude_checkpoint import load_model
    from research.direct.latency58_long_magnitude_context import render_scored_context
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_wave_spectral import augment, objective
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = PHASE / "long-magnitude-001"
    plan_path = root / "plan.json"
    source, audit = read(plan_path), read(root / "checkpoint-audit.json")
    verify_inputs(source)
    require(audit["status"] == "pass" and audit["step"] == 2000
            and read(root / "production-stage/execution.json")["actual_exit_code"] == 0,
            "Require a completed and audited saved checkpoint")
    parent, _ = load_model(source["parent_checkpoint"])
    candidate, _ = load_model(audit["checkpoint"])
    parent_hash, candidate_hash = state_sha256(parent.state_dict()), state_sha256(candidate.state_dict())
    require(parent_hash == source["parent_model_state_sha256"] and candidate_hash == audit["model_state_sha256"],
            "Model identities differ")
    projection = candidate.long_projection.weight.clone()
    require(torch.count_nonzero(projection) > 0, "Long projection did not learn")
    start, examples, microbatch = 2_600_000, 64, 4
    config = source["config"]
    require(start >= config["data_start"] + config["steps"] * config["batch_size"],
            "Diagnostic addresses overlap the latest training stage")
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=production_config)
    require(manifest_sha == source["manifest_sha256"], "Training corpus changed")
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=start + examples)
    out = PHASE / "long-magnitude-learning-001"
    require(not out.exists(), "Preserve existing diagnostics")
    require_space(source, 380_000_000)
    bindings = {**source["source_bindings"], str(plan_path): sha(plan_path),
                str(root / "checkpoint-audit.json"): sha(root / "checkpoint-audit.json"),
                audit["checkpoint"]["path"]: audit["checkpoint"]["sha256"],
                str(Path(__file__).resolve()): sha(__file__)}
    out.mkdir()
    plan = {"source_bindings": bindings, "data_start": start, "examples": examples, "microbatch_size": microbatch,
            "parent_model_state_sha256": parent_hash, "candidate_model_state_sha256": candidate_hash,
            "training_only": True, "validation_or_test_used": False, "precision": "CPU FP32",
            "views": ["parent", "trained", "trained_long_projection_zero"],
            "selection": source["training_selection"], "augmentation": source["config"]["augmentation"],
            "limitation": "Fresh counter addresses from the training corpus do not establish unseen-song generalization or exclude audio overlap with earlier crops. Disabling the trained projection is an intervention, not an independently trained ablation."}
    write(out / "plan.json", plan)
    began, rows = time.monotonic(), []
    with torch.inference_mode(), (out / "progress.jsonl").open("x", buffering=1) as progress:
        for first in range(start, start + examples, microbatch):
            inputs = [dataset[i] for i in range(first, first + microbatch)]
            mixture, truth = (torch.stack([item[i] for item in inputs]) for i in range(2))
            mixture, truth, changed, factors = augment(mixture, truth, seed=config["seed"], first_sample_index=first)
            digest = hashlib.sha256()
            for value in (mixture, truth, changed, factors):
                digest.update(value.contiguous().numpy().tobytes())
            scores = {}
            for name, model in (("parent", parent), ("trained", candidate), ("trained_long_projection_zero", candidate)):
                if name == "trained_long_projection_zero":
                    candidate.long_projection.weight.zero_()
                rendered = render_scored_context(model, mixture, warmup_samples=source["warmup_samples"], carry_state=True)
                terms = objective(rendered.raw, rendered.deployed, truth[..., source["warmup_samples"]:], rendered.physical_mixture)
                scores[name] = {"loss": float(terms.total), "sdr_db": -float(terms.negative_sdr_db),
                                "per_stem_sdr_db": [-v for v in terms.per_stem_negative_sdr_db.tolist()]}
                del rendered, terms
            candidate.long_projection.weight.copy_(projection)
            row = {"first_sample_index": first, "examples": microbatch, "augmented_inputs_sha256": digest.hexdigest(),
                   "scores": scores, "elapsed_seconds": time.monotonic() - began}
            rows.append(row)
            progress.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({"examples_complete": len(rows) * microbatch,
                  "trained_minus_parent_sdr_db": scores["trained"]["sdr_db"] - scores["parent"]["sdr_db"],
                  "trained_minus_zero_projection_sdr_db": scores["trained"]["sdr_db"] - scores["trained_long_projection_zero"]["sdr_db"],
                  "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
            del inputs, mixture, truth, changed, factors
    require(state_sha256(parent.state_dict()) == parent_hash and state_sha256(candidate.state_dict()) == candidate_hash
            and not torch.cuda.is_initialized(), "Models changed or CUDA was used")
    verify_inputs(plan)
    aggregates = {}
    for name in plan["views"]:
        aggregates[name] = {key: sum(row["scores"][name][key] for row in rows) / len(rows) for key in ("loss", "sdr_db")}
        aggregates[name]["per_stem_sdr_db"] = [sum(row["scores"][name]["per_stem_sdr_db"][i] for row in rows) / len(rows)
                                              for i in range(4)]
    result = {"status": "complete", "training_only": True, "validation_or_test_used": False,
              "examples": examples, "aggregates": aggregates, "source_bindings_unchanged": True,
              "model_tensors_restored_exactly": True, "gpu_used": False, "rows": rows,
              "elapsed_seconds": time.monotonic() - began, "plan_sha256": sha(out / "plan.json"),
              "limitation": plan["limitation"], "counted_bytes_after": require_space(source, 380_000_000)}
    write(out / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
