"""Verify every actual augmented batch and learning rate across teacher arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(2, 250, 500, 1000), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous match audit")
    from research.direct.latency58_sdr_checkpoint import read_generation
    plans, journals, receipts, bindings = {}, {}, {}, {str(Path(__file__).resolve()): sha(__file__)}
    for kind in ("c91", "cropped11"):
        path = PHASE / "sdr-teacher-prep-001" / (kind + "-plan.json")
        plan = read(path)
        require(plan["schema"] == "latency58-sdr-training-v1" and plan["teacher_kind"] == kind,
                "Unexpected training arm")
        generation = Path(plan["run_dir"]) / "checkpoints" / f"step-{args.step:06d}"
        receipt = read_generation(generation, expected_plan_sha=sha(path), require_optimizer=False)
        require(receipt["step"] == args.step, "Generation step differs")
        journal = [json.loads(line) for line in (generation / "metrics.jsonl").read_text().splitlines()]
        require(len(journal) == args.step and [r["step"] for r in journal] == list(range(1, args.step + 1))
                and all(r["teacher_kind"] == kind for r in journal), "Training journal scope differs")
        plans[kind], journals[kind], receipts[kind] = plan, journal, receipt
        bindings.update({str(p): sha(p) for p in (path, generation / "receipt.json", generation / "metrics.jsonl")})
    for key in ("config", "initial_model_state_sha256", "parent_checkpoint", "parent_provenance",
                "objective", "teacher_weight", "precision_policy", "environment", "manifest_sha256", "torch_version"):
        require(plans["c91"][key] == plans["cropped11"][key], "Planned matched condition differs: " + key)
    keys = ("step", "lr", "augmented_batch_sha256", "first_sample_index", "next_sample_index",
            "deranged_examples", "data_hops", "flush_hops")
    for left, right in zip(journals["c91"], journals["cropped11"], strict=True):
        require(all(left[k] == right[k] for k in keys), "Actual input or LR mismatch at step " + str(left["step"]))
    require(receipts["c91"]["model_state_sha256"] != receipts["cropped11"]["model_state_sha256"],
            "Different teachers unexpectedly produced identical trained models")
    require(all(sha(p) == s for p, s in bindings.items()), "Match audit inputs changed")
    report = {"schema": "latency58-sdr-matched-batches-v1", "status": "pass", "step": args.step,
              "matching_updates": args.step, "matching_augmented_examples": 4 * args.step,
              "matched_journal_fields": list(keys), "source_bindings": bindings,
              "source_bindings_unchanged": True, "configuration_identical": True,
              "initial_model_state_sha256": plans["c91"]["initial_model_state_sha256"],
              "different_trained_model_states": {k: r["model_state_sha256"] for k, r in receipts.items()},
              "quality_conclusion": None,
              "limitation": "One paired augmentation seed; no training-seed uncertainty estimate"}
    write(args.output, report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
