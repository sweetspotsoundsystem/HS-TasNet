"""Correct inherited trial lineage metadata while preserving every model tensor.

Original completed checkpoints and their quality inputs remain immutable.
This writes a separate, auditable inference artifact; it does not train.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

from research.direct.train_latency58 import ROOT, disk_bytes, read, require, sha, state_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Canonicalization plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-teacher-lineage-canonicalization-v1", "Wrong correction plan")
    verify_inputs(plan)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1" for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    audit, execution = read(plan["audit"]["path"]), read(plan["audit_execution"]["path"])
    checkpoint, parent_checkpoint = plan["checkpoint"], plan["parent_checkpoint"]
    require(audit["status"] == "pass" and audit["checkpoint"] == checkpoint
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and audit["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and sha(checkpoint["path"]) == checkpoint["sha256"]
            and sha(parent_checkpoint["path"]) == parent_checkpoint["sha256"], "Original audit or checkpoint differs")
    destination = Path(plan["output_directory"])
    phase = ROOT / "research/direct/runs/latency58"
    pending = destination.with_name(destination.name + ".pending")
    require(destination.parent == phase and not destination.exists() and not pending.exists()
            and disk_bytes(phase) + 120_000_000 < plan["artifact_allowance_bytes"], "Fresh atomic output or allowance differs")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    began = time.monotonic()
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    parent = torch.load(parent_checkpoint["path"], map_location="cpu", weights_only=True)
    original = payload["provenance"]
    require(payload["schema"] == "latency58-teacher-inference-v1" and payload["step"] == 250
            and payload["model_state_sha256"] == audit["model_state_sha256"] == state_sha256(payload["model"])
            and payload["plan_sha256"] == audit["plan_sha256"]
            and parent["schema"] == "latency58-asymmetric-inference-v1" and parent["step"] == 500
            and parent_checkpoint["sha256"] == "889f24e482328601bd70d84e0fc16c7b94776ac91e604e2782539bec153108aa"
            and parent["model_state_sha256"] == state_sha256(parent["model"]) == original["parent_model_state_sha256"]
            and original["parent_checkpoint"] == parent_checkpoint
            and parent["provenance"]["training_updates"] == 4750
            and original["training_updates"] == 5000 and original["pilot_updates"] == 2750
            and original["asymmetric_training_updates"] == 750 and original["teacher_trial_updates"] == 250,
            "Authenticated weights, actual parent or completed update counts differ")
    stale = {"initialization": parent["provenance"]["initialization"],
             "parent_provenance": parent["provenance"]["parent_provenance"],
             "parent_training_updates": 4250,
             "initialized_model_state_sha256": parent["provenance"]["initialized_model_state_sha256"]}
    require(all(original[k] == v for k, v in stale.items()), "Checkpoint does not have the reviewed inherited metadata")
    corrected = copy.deepcopy(original)
    corrected.update(initialization="matched_trial_from_authenticated_asymmetric500",
                     parent_provenance=copy.deepcopy(parent["provenance"]), parent_training_updates=4750,
                     initialized_model_state_sha256=parent["model_state_sha256"])
    changes = {key: {"before": original[key], "after": corrected[key]} for key in stale}
    canonical = dict(payload, schema="latency58-teacher-inference-v2", provenance=corrected)
    del parent
    pending.mkdir()
    file = pending / "model.pt"
    with file.open("xb") as stream:
        torch.save(canonical, stream)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(file, map_location="cpu", weights_only=True)
    require(set(loaded) == set(canonical) and loaded["schema"] == canonical["schema"]
            and loaded["provenance"] == corrected and loaded["architecture"] == payload["architecture"]
            and loaded["step"] == payload["step"] and loaded["plan_sha256"] == payload["plan_sha256"]
            and state_sha256(loaded["model"]) == loaded["model_state_sha256"] == payload["model_state_sha256"]
            and all(torch.equal(loaded["model"][k], v) for k, v in payload["model"].items()),
            "Canonical round trip changed model tensors or unintended fields")
    verify_inputs(plan)
    require(not torch.cuda.is_initialized() and sha(args.plan) == args.plan_sha256, "CPU scope or plan changed")
    receipt = {"status": "pass", "schema": "latency58-teacher-lineage-canonicalization-result-v1",
               "input": checkpoint, "parent_checkpoint": parent_checkpoint,
               "output": {"kind": "inference", "path": str(destination / "model.pt"), "sha256": sha(file)},
               "model_state_sha256": loaded["model_state_sha256"], "every_model_tensor_bit_exact": True,
               "metadata_changes": changes, "source_bindings_unchanged": True,
               "plan_sha256": args.plan_sha256, "training_updates_executed": 0, "cuda_initialized": False,
               "elapsed_seconds": time.monotonic() - began}
    with (pending / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(pending, os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    os.rename(pending, destination)
    descriptor = os.open(phase, os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
