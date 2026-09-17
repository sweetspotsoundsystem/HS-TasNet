"""Audit all 250 updates and terminal RNG states across the three completed pilots."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs
from research.direct.audit_latency58_vocal_focus_match import ARMS, compare_journals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require frozen CPU1 matching audit")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-vocal-focus-training-match-plan-v1"
            and set(plan["arms"]) == set(ARMS), "Require all three pilot endpoints")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve final matching audit")
    import torch
    from research.direct.latency58_vocal_focus_checkpoint import read_generation, validate_journal, require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = require_space(plan, 3_000_000)
    plans, rows, receipts, rngs = {}, {}, {}, {}
    for arm in ARMS:
        spec = plan["arms"][arm]
        for item in spec.values():
            require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                    "Unbound final matching input")
        training = read(spec["training_plan"]["path"])
        verify_inputs(training)
        require(training["arm"] == arm and not training["resource_only"] and training["config"]["steps"] == 250,
                "Different completed pilot recipe")
        generation = Path(training["run_dir"]) / "checkpoints/step-000250"
        receipt = read_generation(generation, expected_plan_sha=spec["training_plan"]["sha256"], require_optimizer=False)
        audit, audit_execution, execution = (read(spec[k]["path"]) for k in ("audit", "audit_execution", "execution"))
        monitor_path = Path(execution["monitor_result"])
        monitor = read(monitor_path)
        require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
                and audit["arm"] == arm and audit["step"] == 250
                and audit["matched_input_journal_verified"] and audit["normalized_drum_objective_journal_verified"]
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and audit["model_state_sha256"] == receipt["model_state_sha256"]
                and audit["plan_sha256"] == audit_execution["plan_sha256"] == execution["plan_sha256"]
                == spec["training_plan"]["sha256"]
                and audit_execution["actual_exit_code"] == execution["actual_exit_code"] == 0
                and not audit_execution["timed_out"] and audit_execution["source_bindings_unchanged"]
                and execution["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                "Pilot lacks its original successful saved-state audit and clean monitored exit")
        for path in (monitor_path, *(generation / name for name in ("receipt.json", "model.pt", "rng.pt", "metrics.jsonl"))):
            require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound generation or monitor")
        helpers = load_source("vocal_focus_final_match_helpers", training["helper_source"])
        rows[arm] = validate_journal((generation / "metrics.jsonl").read_bytes(), 250, training, helpers)
        rngs[arm] = torch.load(generation / "rng.pt", map_location="cpu", weights_only=False)
        plans[arm], receipts[arm] = training, receipt
    compared = compare_journals(plans, rows)
    require(compared["updates_per_arm"] == 250 and compared["microbatches_per_arm"] == 1000
            and compared["examples_per_arm"] == 4000
            and all(plans[arm][key] == plans["original"][key] for arm in ARMS
                    for key in ("matched_protocol", "resource_pairing", "resource_pairing_execution")),
            "Full pilots differ from their common protocol or rehearsal pairing")
    initial_rng = rngs["original"]
    for arm in ARMS:
        current = rngs[arm]
        require(set(current) == {"python", "numpy", "torch_cpu", "torch_cuda"}
                and current["python"] == initial_rng["python"]
                and pickle.dumps(current["numpy"]) == pickle.dumps(initial_rng["numpy"])
                and torch.equal(current["torch_cpu"], initial_rng["torch_cpu"])
                and len(current["torch_cuda"]) == len(initial_rng["torch_cuda"]) == 1
                and torch.equal(current["torch_cuda"][0], initial_rng["torch_cuda"][0]), "Terminal RNG streams differ")
    require(not torch.cuda.is_initialized(), "Final matching initialized CUDA")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-vocal-focus-training-match-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "training_plans": {arm: plan["arms"][arm]["training_plan"] for arm in ARMS},
        "trained_model_states": {arm: receipt["model_state_sha256"] for arm, receipt in receipts.items()},
        **compared, "final_rng_states_exact": True, "training_updates_executed": 0,
        "optimizer_instances": 0, "cuda_initialized": False, "quality_selected": False,
        "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
        "limitations": ["One matched training seed; this audit verifies training conditions, not separation quality.",
                        "Original augmentation intentionally differs on focused examples; the two focused arms share final inputs and targets."]})
    print({"status": "pass", **compared, "final_rng_states_exact": True}, flush=True)


if __name__ == "__main__":
    main()
