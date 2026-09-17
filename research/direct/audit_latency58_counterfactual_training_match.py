"""Match every completed selective-training example and RNG stream to its control."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import pickle

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, load_source
from research.direct.latency58_counterfactual_checkpoint import SHARED_CONTROL_KEYS, IDENTITY_KEYS, require_space


def load_completed_match(binding, evidence):
    require(set(binding) == {"plan", "result", "execution"}, "Incomplete training match binding")
    for item in binding.values():
        require(sha(item["path"]) == item["sha256"], "Completed training match changed")
        evidence[item["path"]] = item["sha256"]
    plan, result, execution = (read(binding[k]["path"]) for k in ("plan", "result", "execution"))
    require(plan["schema"] == "latency58-counterfactual-training-match-plan-v1"
            and result["schema"] == "latency58-counterfactual-training-match-v1" and result["status"] == "pass"
            and result["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and result["plan_sha256"] == execution["plan_sha256"] == binding["plan"]["sha256"]
            and result["updates_per_mode"] == 250 and result["microbatches_per_mode"] == 1000
            and result["examples_per_mode"] == 4000 and result["all_inputs_and_teacher_targets_exact"]
            and result["final_rng_states_exact"] and result["teacher_mode"] == "ordinary_only"
            and result["training_plans"] == {k: v["training_plan"] for k, v in plan["models"].items()}
            and result["source_bindings"] == plan["source_bindings"], "Training match did not pass completely")
    verify_inputs(plan)
    evidence.update(plan["source_bindings"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use frozen CPU1 training authentication")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-counterfactual-training-match-plan-v1"
            and set(plan["models"]) == {"reference", "candidate"}, "Different matching scope")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve matching evidence")
    require_space(plan, 3_000_000)
    import torch
    from research.direct import latency58_vocal_focus_checkpoint as old
    from research.direct import latency58_counterfactual_checkpoint as new
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    recipes, rows, receipts, rngs = {}, {}, {}, {}
    for side, spec in plan["models"].items():
        for item in spec.values():
            require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                    "Unbound completed training evidence")
        training, audit, audited, execution = (read(spec[k]["path"]) for k in
                                               ("training_plan", "audit", "audit_execution", "execution"))
        family = new if side == "candidate" else old
        family.validate_recipe(training)
        verify_inputs(training)
        generation = Path(training["run_dir"]) / "checkpoints/step-000250"
        receipt = family.read_generation(generation, expected_plan_sha=spec["training_plan"]["sha256"], require_optimizer=False)
        monitor_path = Path(execution["monitor_result"])
        monitor, status = read(monitor_path), read(Path(training["run_dir"]) / "status.json")
        require(not training["resource_only"] and training["arm"] == "focused"
                and audit["status"] == "pass" and audit["source_bindings_unchanged"]
                and audit["matched_input_journal_verified"] and audit["normalized_drum_objective_journal_verified"]
                and audit["step"] == receipt["step"] == training["config"]["steps"] == status["step"] == 250
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and audit["model_state_sha256"] == receipt["model_state_sha256"]
                and audit["plan_sha256"] == audited["plan_sha256"] == execution["plan_sha256"] == spec["training_plan"]["sha256"]
                and audited["actual_exit_code"] == execution["actual_exit_code"] == 0
                and not audited["timed_out"] and audited["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
                and monitor["post_exit_quiet_completed"] and status["status"] == "complete",
                "Training or its original saved-state audit did not close successfully")
        if side == "candidate":
            require(training["teacher_mode"] == audit["teacher_mode"] == "ordinary_only"
                    and audit["selected_teacher_participation_and_batch_divisor_verified"], "Different selected teacher policy")
        for path in (monitor_path, Path(training["run_dir"]) / "status.json",
                     *(generation / name for name in ("receipt.json", "model.pt", "rng.pt", "metrics.jsonl"))):
            require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound retained state or clean exit")
        helpers = load_source("counterfactual_training_match_helpers", training["helper_source"])
        rows[side] = family.validate_journal((generation / "metrics.jsonl").read_bytes(), 250, training, helpers)
        rngs[side] = torch.load(generation / "rng.pt", map_location="cpu", weights_only=False)
        recipes[side], receipts[side] = training, receipt
    reference, candidate = recipes["reference"], recipes["candidate"]
    require(candidate["matched_control_training_plan"] == plan["models"]["reference"]["training_plan"]
            and all(reference[k] == candidate[k] for k in SHARED_CONTROL_KEYS), "Different matched recipes")
    keys = ("first_sample_index", "next_sample_index", "micro_index", "batch_size", "view_codes",
            "deranged_examples", "data_hops", "flush_hops", "initial_state_detached",
            "augmented_batch_sha256", "teacher_targets_sha256", *IDENTITY_KEYS)
    for a, b in zip(rows["reference"], rows["candidate"], strict=True):
        require(a["step"] == b["step"] and a["lr"] == b["lr"], "Different update or learning rate")
        for x, y in zip(a["microbatches"], b["microbatches"], strict=True):
            require(all(x[k] == y[k] for k in keys), "A production crop, teacher target or RNG draw differs")
    a, b = rngs["reference"], rngs["candidate"]
    require(set(a) == set(b) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and a["python"] == b["python"] and pickle.dumps(a["numpy"]) == pickle.dumps(b["numpy"])
            and torch.equal(a["torch_cpu"], b["torch_cpu"]) and len(a["torch_cuda"]) == len(b["torch_cuda"]) == 1
            and torch.equal(a["torch_cuda"][0], b["torch_cuda"][0]), "Terminal RNG states differ")
    saved_rng = {"python": pickle.dumps(b["python"]), "numpy": pickle.dumps(b["numpy"]),
                 "torch_cpu": b["torch_cpu"].numpy().tobytes(), "torch_cuda": b["torch_cuda"][0].numpy().tobytes()}
    require({k: hashlib.sha256(v).hexdigest() for k, v in saved_rng.items()}
                == receipts["candidate"]["final_rng_state_sha256"] and not torch.cuda.is_initialized(),
            "Candidate RNG receipt differs from saved state or CPU scope changed")
    require(receipts["reference"]["model_state_sha256"] != receipts["candidate"]["model_state_sha256"],
            "The isolated teacher policy produced the same final model")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-counterfactual-training-match-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "training_plans": {side: spec["training_plan"] for side, spec in plan["models"].items()},
          "model_states": {side: receipt["model_state_sha256"] for side, receipt in receipts.items()},
          "updates_per_mode": 250, "microbatches_per_mode": 1000, "examples_per_mode": 4000,
          "all_inputs_and_teacher_targets_exact": True, "final_rng_states_exact": True,
          "teacher_mode": "ordinary_only", "quality_selected": False, "training_updates_executed": 0,
          "limitations": ["One matched training seed; authentication does not establish a quality improvement."]})
    print({"status": "pass", "microbatches_per_mode": 1000, "final_rng_states_exact": True}, flush=True)


if __name__ == "__main__":
    main()
