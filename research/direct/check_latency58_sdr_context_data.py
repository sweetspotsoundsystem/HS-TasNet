"""Check real four-second counter-addressed crops for a matched context trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import read, require, sha, write


def digest(values):
    result = hashlib.sha256()
    for name, value in values:
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Context data plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-context-data-v1" and plan["warmup_samples"] == 88064
            and plan["scored_samples"] == 88064 and plan["batch_size"] == 4
            and plan["sample_indices"] == list(range(920000, 920016))
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Context data scope differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CUDA-hidden CPU1")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve existing data proof")
    import torch
    from research import experiment
    from research.direct.train_latency58 import PRODUCTION
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260909)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=config)
    length = plan["warmup_samples"] + plan["scored_samples"]
    minimum = min(t.effective_frames for t in tracks)
    require(minimum >= length, "A corpus track is too short for a real four-second crop")
    options = dict(root_weights=config["sampling"]["root_weights"], seed=config["seed"], crop_samples=length,
                   vocal_active_probability=config["sampling"]["vocal_active_probability"], final_sample_index=920016)
    left, right = (production.CounterAddressedCropDataset(tracks, **options) for _ in range(2))
    rows = []
    for start in range(920000, 920016, 4):
        a, b = ([dataset[i] for i in range(start, start + 4)] for dataset in (left, right))
        require(all(torch.equal(x, y) for one, two in zip(a, b) for x, y in zip(one, two)),
                "Independent counter-addressed data reads differ")
        mixture, targets = (torch.stack([row[k] for row in a]) for k in (0, 1))
        require(mixture.shape == (4, 2, length) and targets.shape == (4, 4, 2, length), "Crop shape differs")
        input_sha = digest((("mixture", mixture), ("targets", targets)))
        rng = torch.get_rng_state().clone()
        full = experiment._augment_training_distribution(mixture=mixture, targets=targets)
        after = torch.get_rng_state().clone()
        pieces = []
        for region in (slice(0, 88064), slice(88064, None)):
            torch.set_rng_state(rng)
            value = experiment._augment_training_distribution(mixture=mixture[..., region], targets=targets[..., region])
            require(torch.equal(torch.get_rng_state(), after), "Augmentation draws depend on time-span length")
            pieces.append(value)
        torch.set_rng_state(after)
        require(torch.equal(torch.cat((pieces[0][0], pieces[1][0]), dim=-1), full[0])
                and torch.equal(torch.cat((pieces[0][1], pieces[1][1]), dim=-1), full[1])
                and torch.equal(pieces[0][2], full[2]) and torch.equal(pieces[1][2], full[2])
                and input_sha == digest((("mixture", mixture), ("targets", targets)))
                and all(bool(torch.isfinite(v).all()) for v in full),
                "Full-crop augmentation breaks the prefix/suffix relationship or changes inputs")
        rows.append({"first_sample_index": start, "next_sample_index": start + 4,
                     "unaugmented_batch_sha256": input_sha,
                     "augmented_batch_sha256_cpu": digest(zip(("mixture", "targets", "deranged"), full)),
                     "independent_data_reads_exact": True, "augmentation_commutes_with_physical_split": True,
                     "deranged_examples": int(full[2].sum())})
    require(not torch.cuda.is_initialized() and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Data fixture changed CPU scope or bound inputs")
    result = {"schema": "latency58-sdr-context-data-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "manifest_sha256": manifest_sha, "minimum_track_frames": minimum,
              "corpus_tracks": len(tracks), "full_crop_samples": length, "warmup_samples": 88064,
              "scored_samples": 88064, "sample_indices": plan["sample_indices"], "batches": rows,
              "source_bindings_unchanged": True, "model_loaded": False, "cuda_initialized": False,
              "scope": "Training-data preparation only; GPU augmentation hashes must still match during actual training",
              "elapsed_seconds": time.monotonic() - began}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "samples": 16, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
