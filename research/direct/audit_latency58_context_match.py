"""Authenticate the actual data, teacher targets and RNG across context arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-plan", type=Path, required=True)
    parser.add_argument("--warm-plan", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(2, 25, 250, 500), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use fresh match output with CUDA hidden and CPU1")
    import numpy as np
    import torch
    from research.direct.latency58_context_checkpoint import read_generation, validate_journal
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plans, journals, receipts, rngs = {}, {}, {}, {}
    bindings = {str(Path(__file__).resolve()): sha(__file__)}
    for name, carry, path in (("reset", False, args.reset_plan), ("warm", True, args.warm_plan)):
        plan = read(path)
        require(plan["schema"] == "latency58-context-training-v1" and plan["carry_state"] is carry,
                "Context match arm differs")
        verify_inputs(plan)
        generation = Path(plan["run_dir"]) / "checkpoints" / f"step-{args.step:06d}"
        receipt = read_generation(generation, expected_plan_sha=sha(path), require_optimizer=False)
        helpers = load_source("latency58_context_match_helpers_" + name, plan["helper_source"])
        rows = validate_journal((generation / "metrics.jsonl").read_bytes(), args.step, plan, helpers)
        pointer = read(Path(plan["run_dir"]) / "audit-latest.json")
        audit_path, exec_path = (Path(pointer[k]["path"]) for k in ("audit", "execution"))
        audit, execution = read(audit_path), read(exec_path)
        require(receipt["step"] == args.step and receipt["carry_state"] is carry
                and audit["status"] == "pass" and audit["step"] == args.step and audit["carry_state"] is carry
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and audit["source_bindings_unchanged"] and execution["source_bindings_unchanged"],
                "Context endpoint lacks its independent saved-state audit")
        plans[name], journals[name], receipts[name] = plan, rows, receipt
        rngs[name] = torch.load(generation / "rng.pt", map_location="cpu", weights_only=False)
        bindings.update(plan["source_bindings"])
        bindings.update({str(p.resolve()): sha(p) for p in (path, audit_path, exec_path, generation / "receipt.json",
                                                          generation / "metrics.jsonl", generation / "rng.pt")})
    normalized = [{k: v for k, v in plan.items() if k not in ("carry_state", "run_dir")} for plan in plans.values()]
    require(normalized[0] == normalized[1], "Context plans differ beyond state carry and output path")
    keys = ("step", "lr", "augmented_batch_sha256", "teacher_targets_sha256", "first_sample_index", "next_sample_index",
            "deranged_examples", "data_hops", "flush_hops", "warmup_samples", "scored_samples", "initial_state_detached")
    for left, right in zip(journals["reset"], journals["warm"], strict=True):
        require(all(left[k] == right[k] for k in keys), "Actual matched condition differs at step " + str(left["step"]))
    left, right = rngs["reset"], rngs["warm"]
    require(left["python"] == right["python"]
            and all(np.array_equal(a, b) for a, b in zip(left["numpy"], right["numpy"], strict=True))
            and torch.equal(left["torch_cpu"], right["torch_cpu"])
            and len(left["torch_cuda"]) == len(right["torch_cuda"]) == 1
            and torch.equal(left["torch_cuda"][0], right["torch_cuda"][0]), "Saved matched RNG states differ")
    states = {name: receipt["model_state_sha256"] for name, receipt in receipts.items()}
    require(len(set(states.values())) == 2 and not torch.cuda.is_initialized()
            and all(sha(p) == s for p, s in bindings.items()), "Context match inputs changed or arms are identical")
    result = {"schema": "latency58-context-matched-batches-v1", "status": "pass", "step": args.step,
              "matching_updates": args.step, "matching_augmented_examples": 4 * args.step,
              "matched_journal_fields": list(keys), "source_bindings": bindings, "source_bindings_unchanged": True,
              "configuration_identical_except_student_state_carry": True, "saved_rng_states_exact": True,
              "initial_model_state_sha256": plans["reset"]["parent"]["model_state_sha256"],
              "different_trained_model_states": states, "quality_conclusion": None,
              "limitation": "One paired augmentation seed; no training-seed uncertainty estimate"}
    write(args.output, result)
    print(json.dumps({"status": "pass", "step": args.step, "matching_examples": 4 * args.step,
                      "teacher_targets_match": True, "saved_rng_states_exact": True}), flush=True)


if __name__ == "__main__":
    main()
