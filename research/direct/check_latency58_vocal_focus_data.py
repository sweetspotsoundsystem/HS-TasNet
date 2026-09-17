"""Qualify focused augmentation on disjoint real training addresses, CPU only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs
from research.direct.check_latency58_long_context_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Focused data plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-data-plan-v1"
            and plan["sample_indices"] == list(range(972000, 972016))
            and plan["crop_samples"] == 176128 and plan["warmup_samples"] == 88064
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require the bounded, disjoint CPU1 data fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous data proof")
    import torch
    from research import experiment
    from research.direct.latency58_vocal_focus_augmentation import augment_vocal_focus, VERSION, VIEW_NAMES
    from research.direct.latency58_sdr_checkpoint import require_space
    before = require_space(plan, 2_000_000)
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260921)
    torch.use_deterministic_algorithms(True)
    config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=config)
    options = dict(root_weights=config["sampling"]["root_weights"], seed=config["seed"],
                   crop_samples=plan["crop_samples"],
                   vocal_active_probability=config["sampling"]["vocal_active_probability"], final_sample_index=972016)
    left, right = (production.CounterAddressedCropDataset(tracks, **options) for _ in range(2))
    rows, began = [], time.monotonic()
    for start in range(972000, 972016, 4):
        a, b = ([dataset[i] for i in range(start, start + 4)] for dataset in (left, right))
        require(all(torch.equal(x, y) for first, second in zip(a, b, strict=True)
                    for x, y in zip(first, second, strict=True)), "Independent source reads differ")
        mixture, targets = (torch.stack([r[k] for r in a]) for k in (0, 1))
        input_sha = digest((("mixture", mixture), ("targets", targets)))
        initial_rng = torch.get_rng_state().clone()
        original = experiment._augment_training_distribution(mixture=mixture, targets=targets)
        after_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        control = augment_vocal_focus(mixture, targets, first_sample_index=start, enabled=False)
        require(torch.equal(torch.get_rng_state(), after_rng)
                and all(torch.equal(x, y) for x, y in zip(original,
                    (control.mixture, control.targets, control.vocal_derangement), strict=True)),
                "Disabled focus must preserve original data and RNG exactly")
        torch.set_rng_state(initial_rng)
        focused = augment_vocal_focus(mixture, targets, first_sample_index=start, enabled=True)
        require(torch.equal(torch.get_rng_state(), after_rng)
                and all(torch.equal(x, y) for x, y in zip(original, focused.original_augmentation, strict=True))
                and focused.view_codes == (0, 1, 2, 3)
                and torch.equal(focused.mixture[2:], original[0][2:])
                and torch.equal(focused.targets[2:], original[1][2:])
                and torch.equal(focused.vocal_derangement[2:], original[2][2:])
                and not bool(focused.vocal_derangement[:2].any()),
                "Focused views changed ordinary examples, role allocation or RNG")
        for index, included in ((0, (0, 1, 3)), (1, (2,))):
            expected = torch.zeros_like(targets[index])
            expected[list(included)] = targets[index, list(included)]
            require(torch.equal(focused.targets[index], expected)
                    and torch.equal(focused.mixture[index], expected.sum(dim=0)),
                    "Focused view does not use exactly the pristine source subset")
        pieces = []
        for region in (slice(0, 88064), slice(88064, None)):
            torch.set_rng_state(initial_rng)
            piece = augment_vocal_focus(mixture[..., region], targets[..., region],
                                       first_sample_index=start, enabled=True)
            require(torch.equal(torch.get_rng_state(), after_rng), "Focused augmentation draws depend on crop length")
            pieces.append(piece)
        require(torch.equal(torch.cat([p.mixture for p in pieces], dim=-1), focused.mixture)
                and torch.equal(torch.cat([p.targets for p in pieces], dim=-1), focused.targets)
                and all(torch.equal(p.vocal_derangement, focused.vocal_derangement) for p in pieces)
                and input_sha == digest((("mixture", mixture), ("targets", targets))),
                "Focused augmentation changes source tensors or fails to commute with history/scored split")
        torch.set_rng_state(after_rng)
        # Absolute roles must also work for a non-aligned group of addresses.
        shifted_rng = torch.get_rng_state().clone()
        shifted = augment_vocal_focus(mixture, targets, first_sample_index=start + 1, enabled=True)
        require(shifted.view_codes == (1, 2, 3, 0), "View allocation depends on local batch position")
        torch.set_rng_state(shifted_rng)
        rows.append({"first_sample_index": start, "next_sample_index": start + 4,
            "unaugmented_batch_sha256": input_sha,
            "original_augmented_batch_sha256_cpu": digest(zip(("mixture", "targets", "deranged"), original)),
            "focused_augmented_batch_sha256_cpu": digest((("mixture", focused.mixture),
                ("targets", focused.targets), ("deranged", focused.vocal_derangement))),
            "view_names": [VIEW_NAMES[code] for code in focused.view_codes],
            "original_deranged_examples": int(original[2].sum()),
            "focused_deranged_examples": int(focused.vocal_derangement.sum()),
            "focused_vocal_power": focused.targets[:, 2].square().mean(dim=(1, 2)).tolist(),
            "all_input_data_unchanged": True, "ordinary_examples_and_rng_exact": True,
            "history_and_scored_views_exact": True, "independent_data_reads_exact": True})
    require(not torch.cuda.is_initialized(), "CPU-only data scope changed")
    verify_inputs(plan)
    result = {"schema": "latency58-vocal-focus-data-result-v1", "status": "pass", "version": VERSION,
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "manifest_sha256": manifest_sha,
              "crop_samples": 176128, "sample_indices": plan["sample_indices"], "batches": rows,
              "elapsed_seconds": time.monotonic() - began, "counted_bytes_before": before,
              "counted_bytes_after": require_space(plan, 0), "cuda_initialized": False,
              "model_loaded": False, "training_updates_executed": 0,
              "training_recipe_frozen": False, "quality_selected": False,
              "limitations": ["CPU data/RNG proof only; GPU augmentation and full-gradient behavior still require rehearsal.",
                              "No teacher weighting, parent or training recipe is selected by this fixture.",
                              "Source-isolated views may differ from full music; preserve a normal-training control and full-mixture evaluation."]}
    write(out / "result.json", result)
    print({"status": "pass", "examples": 16, "original_examples_and_rng_exact": True,
           "history_scored_alignment_exact": True}, flush=True)


if __name__ == "__main__":
    main()
