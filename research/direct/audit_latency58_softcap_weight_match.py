"""Authenticate matched data and RNG for the two soft-cap coefficients at 250."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.parent.is_dir()
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a fresh match output with CUDA hidden and CPU1")
    import numpy as np
    import torch
    from research.direct import latency58_sdr_softcap_checkpoint as reference_loader
    from research.direct import latency58_sdr_softcap_strong_checkpoint as candidate_loader

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    step = 250
    plans, journals, receipts, rngs = {}, {}, {}, {}
    bindings = {str(Path(__file__).resolve()): sha(__file__)}
    arms = (("small", "sdr-softcap", args.reference_plan, reference_loader),
            ("strong", "sdr-softcap-strong", args.candidate_plan, candidate_loader))
    for name, schema_name, path, loader in arms:
        plan = read(path)
        require(plan["schema"] == f"latency58-{schema_name}-training-v1"
                and plan["carry_state"] is True and not plan["resource_only"],
                "Match arm or training scope differs")
        verify_inputs(plan)
        generation = Path(plan["run_dir"]) / "checkpoints" / f"step-{step:06d}"
        receipt = loader.read_generation(generation, expected_plan_sha=sha(path), require_optimizer=False)
        helpers = load_source("latency58_softcap_match_helpers_" + name, plan["helper_source"])
        rows = loader.validate_journal((generation / "metrics.jsonl").read_bytes(), step, plan, helpers)
        pointer = read(Path(plan["run_dir"]) / "audit-latest.json")
        audit_path, exec_path = (Path(pointer[k]["path"]) for k in ("audit", "execution"))
        audit, execution = read(audit_path), read(exec_path)
        root_exec_path = audit_path.parent / "execution.json"
        root_execution = read(root_exec_path)
        monitor_path = Path(root_execution["monitor_result"])
        monitor = read(monitor_path)
        require(receipt["step"] == audit["step"] == pointer["step"] == step
                and receipt["carry_state"] and audit["carry_state"]
                and audit["status"] == "pass"
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and audit["model_state_sha256"] == receipt["model_state_sha256"]
                and execution["actual_exit_code"] == root_execution["actual_exit_code"]
                == monitor["child_exit_code"] == 0 and not execution["timed_out"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["post_exit_quiet_completed"]
                and audit["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
                and root_execution["source_bindings_unchanged"]
                and execution["plan_sha256"] == root_execution["plan_sha256"]
                == audit["plan_sha256"] == sha(path),
                "Endpoint lacks its independent saved-state audit and healthy exit")
        plans[name], journals[name], receipts[name] = plan, rows, receipt
        rngs[name] = torch.load(generation / "rng.pt", map_location="cpu", weights_only=False)
        bindings.update(plan["source_bindings"])
        paths = [path, audit_path, exec_path, root_exec_path, monitor_path,
                 *[generation / n for n in ("model.pt", "receipt.json", "metrics.jsonl", "rng.pt")]]
        bindings.update({str(p.resolve()): sha(p) for p in paths})
    shared = ("parent", "parent_prefix", "config", "warmup_samples", "scored_samples", "carry_state",
              "teacher_kind", "teacher_weight", "teacher_model_state_sha256", "teacher",
              "precision_policy", "torch_version", "environment", "helper_source", "watchdog_source",
              "manifest_sha256", "geometry", "functional_proofs", "functional_proof", "functional_execution",
              "sdr_softcap_version", "sdr_softcap_error_ratio_floor")
    require(all(plans["small"][k] == plans["strong"][k] for k in shared),
            "Parent, data, optimizer schedule, teacher or geometry differs")
    require(plans["small"]["sdr_softcap_weight"] == 0.003
            and plans["strong"]["sdr_softcap_weight"] == 0.03
            and plans["strong"]["sdr_softcap_error_ratio_floor"] == 0.01,
            "The intended auxiliary recipes differ")
    keys = ("step", "lr", "augmented_batch_sha256", "teacher_targets_sha256", "first_sample_index",
            "next_sample_index", "deranged_examples", "data_hops", "flush_hops", "warmup_samples",
            "scored_samples", "initial_state_detached", "carry_state", "teacher_kind", "teacher_weight")
    for left, right in zip(journals["small"], journals["strong"], strict=True):
        require(all(left[k] == right[k] for k in keys)
                and left["sdr_softcap_active_windows"] == right["sdr_softcap_active_windows"]
                and left["sdr_softcap_active_examples_per_stem"]
                == right["sdr_softcap_active_examples_per_stem"],
                "Actual matched condition differs at step " + str(left["step"]))
    left, right = rngs["small"], rngs["strong"]
    require(left["python"] == right["python"]
            and all(np.array_equal(a, b) for a, b in zip(left["numpy"], right["numpy"], strict=True))
            and torch.equal(left["torch_cpu"], right["torch_cpu"])
            and len(left["torch_cuda"]) == len(right["torch_cuda"]) == 1
            and torch.equal(left["torch_cuda"][0], right["torch_cuda"][0]),
            "Saved matched RNG states differ")
    states = {name: receipt["model_state_sha256"] for name, receipt in receipts.items()}
    require(len(set(states.values())) == 2 and not torch.cuda.is_initialized()
            and all(sha(p) == s for p, s in bindings.items()),
            "Match inputs changed, CUDA was initialized or trained models are identical")
    write(args.output, {
        "schema": "latency58-softcap-weight-matched-batches-v1", "status": "pass", "step": step,
        "matching_updates": step, "matching_augmented_examples": 4 * step,
        "matched_journal_fields": list(keys), "source_bindings": bindings,
        "source_bindings_unchanged": True, "parent_data_teacher_and_optimizer_schedule_identical": True,
        "activity_masks_exact": True, "saved_rng_states_exact": True,
        "initial_model_state_sha256": plans["small"]["parent"]["model_state_sha256"],
        "different_trained_model_states": states, "quality_conclusion": None,
        "treatment": "Only the auxiliary coefficient changes from 0.003 to 0.03; formula and 20 dB soft cap are unchanged",
        "limitation": "One paired augmentation seed; no training-seed uncertainty estimate",
    })
    print("Matched 250 updates, 1000 augmented examples, teacher targets, activity masks and saved RNG states.",
          flush=True)


if __name__ == "__main__":
    main()
