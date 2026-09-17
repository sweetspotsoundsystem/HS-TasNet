"""Verify every augmented microbatch and shared teacher target across history arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(250, 500), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use fresh output and CPU scope")
    from research.direct.latency58_sdr_history_checkpoint import read_generation, validate_journal
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plans, journals, receipts, rngs = {}, {}, {}, {}
    bindings = {str(Path(__file__).resolve()): sha(__file__)}
    for arm, warm in (("short", 88064), ("long", 352256)):
        path = PHASE / "sdr-history-prep-001" / (arm + "-training-plan.json")
        plan = read(path)
        verify_inputs(plan)
        require(plan["schema"] == "latency58-sdr-history-training-v1"
                and plan["warmup_samples"] == warm and not plan["resource_only"], "Unexpected history arm")
        generation = Path(plan["run_dir"]) / "checkpoints" / f"step-{args.step:06d}"
        receipt = read_generation(generation, expected_plan_sha=sha(path), require_optimizer=False)
        require(receipt["step"] == args.step, "Generation endpoint differs")
        helpers = load_source("history_match_helpers", plan["helper_source"])
        rows = validate_journal((generation / "metrics.jsonl").read_bytes(), args.step, plan, helpers)
        rngs[arm] = torch.load(generation / "rng.pt", map_location="cpu", weights_only=False)
        plans[arm], journals[arm], receipts[arm] = plan, rows, receipt
        bindings.update(plan["source_bindings"])
        bindings.update({str(p): sha(p) for p in (path, *(generation / name for name in
                                                           ("receipt.json", "model.pt", "metrics.jsonl", "rng.pt")))})
    shared = ("config", "parent", "teacher", "teacher_kind", "teacher_weight", "teacher_model_state_sha256",
              "teacher_history_samples", "scored_samples", "carry_state", "drum_weight", "objective_version",
              "accumulation_version", "history_version", "microbatch_size", "accumulation_steps",
              "optimizer_initialization", "precision_policy", "environment", "manifest_sha256", "torch_version",
              "matched_protocol", "matched_resource")
    require(all(plans["short"][k] == plans["long"][k] for k in shared), "A planned matched condition differs")
    keys = ("step", "lr", "augmented_batch_sha256", "teacher_targets_sha256", "first_sample_index",
            "next_sample_index", "deranged_examples", "data_hops", "flush_hops", "teacher_history_samples",
            "scored_samples", "gradient_clips_this_update", "adam_steps_this_update")
    micro_keys = ("micro_index", "batch_size", "first_sample_index", "next_sample_index", "deranged_examples",
                  "augmented_batch_sha256", "teacher_targets_sha256", "data_hops", "flush_hops")
    for left, right in zip(journals["short"], journals["long"], strict=True):
        require(all(left[k] == right[k] for k in keys), "Matched update differs: " + str(left["step"]))
        for a, b in zip(left["microbatches"], right["microbatches"], strict=True):
            require(all(a[k] == b[k] for k in micro_keys), "Matched microbatch or shared teacher target differs")
    left_rng, right_rng = rngs["short"], rngs["long"]
    require(left_rng["python"] == right_rng["python"]
            and pickle.dumps(left_rng["numpy"]) == pickle.dumps(right_rng["numpy"])
            and torch.equal(left_rng["torch_cpu"], right_rng["torch_cpu"])
            and len(left_rng["torch_cuda"]) == len(right_rng["torch_cuda"]) == 1
            and torch.equal(left_rng["torch_cuda"][0], right_rng["torch_cuda"][0])
            and not torch.cuda.is_initialized(), "Endpoint RNG streams differ or CUDA was initialized")
    require(all(sha(p) == s for p, s in bindings.items()), "Match evidence changed")
    result = {"schema": "latency58-history-matched-batches-v1", "status": "pass", "step": args.step,
              "source_bindings": bindings, "source_bindings_unchanged": True,
              "matching_updates": args.step, "matching_microbatches": 4 * args.step,
              "matching_augmented_examples": 16 * args.step, "matched_journal_fields": list(keys),
              "matched_microbatch_fields": list(micro_keys), "final_rng_states_exact": True,
              "student_history_samples": [88064, 352256], "teacher_history_samples": 352256,
              "all_augmentation_and_teacher_targets_exact": True,
              "initial_model_state_sha256": plans["short"]["parent"]["model_state_sha256"],
              "trained_model_states": {a: r["model_state_sha256"] for a, r in receipts.items()},
              "quality_selected": False, "cuda_initialized": False,
              "limitation": "One paired training seed; no training-seed uncertainty estimate."}
    write(args.output, result)
    print({"status": "pass", "matching_updates": args.step, "matching_microbatches": 4 * args.step}, flush=True)


if __name__ == "__main__":
    main()
