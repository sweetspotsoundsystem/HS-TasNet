"""Authenticated split checkpoints for the matched SDR recovery fine-tunes."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import random

from research.direct.train_latency58 import disk_bytes, read, require, sha, state_sha256


def require_space(plan, extra_bytes):
    counted = sum(disk_bytes(Path(root)) for root in plan["counted_roots"])
    require(counted + extra_bytes < plan["stop_counted_bytes"], "Insufficient reserved artifact space")
    return counted


def read_generation(directory, *, expected_plan_sha=None, require_optimizer=True):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    expected_files = {"model.pt", "optimizer.pt", "rng.pt", "metrics.jsonl"}
    require(receipt["schema"] == "latency58-sdr-generation-v1" and set(receipt["files"]) == expected_files
            and type(receipt["step"]) is int and receipt["step"] > 0,
            "Checkpoint receipt schema or inventory differs")
    if expected_plan_sha is not None:
        require(receipt["plan_sha256"] == expected_plan_sha, "Checkpoint belongs to another plan")
    for name, binding in receipt["files"].items():
        if name == "optimizer.pt" and not require_optimizer:
            continue
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == binding["bytes"]
                and sha(path) == binding["sha256"], "Checkpoint bytes changed: " + name)
    return receipt


def load_model(directory, plan, *, expected_plan_sha):
    """Read only inference tensors on CPU; no optimizer or RNG restoration."""
    import torch
    from research.direct.latency58_sdr_teacher import (
        STUDENT_SHA256, STUDENT_STATE_SHA256, load_initial_student)

    require(not torch.cuda.is_initialized(), "Load inference model before CUDA")
    receipt = read_generation(directory, expected_plan_sha=expected_plan_sha, require_optimizer=False)
    model = load_initial_student()
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    payload = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=True)
    step = receipt["step"]
    provenance = payload["provenance"]
    require(payload["schema"] == "latency58-sdr-inference-v1" and payload["step"] == step
            and payload["plan_sha256"] == expected_plan_sha and payload["architecture"] == model.architecture_metadata
            and payload["model_state_sha256"] == receipt["model_state_sha256"]
            and provenance["initialization"] == "sdr_recovery_from_user_accepted_teacher250"
            and provenance["parent_checkpoint"]["sha256"] == STUDENT_SHA256
            and provenance["parent_model_state_sha256"] == STUDENT_STATE_SHA256
            and provenance["parent_training_updates"] == 5000
            and provenance["training_updates"] == 5000 + step
            and provenance["asymmetric_training_updates"] == 750 + step
            and provenance["pilot_updates"] == 2750 + step
            and provenance["sdr_trial_updates"] == provenance["tail_updates"] == step
            and provenance["teacher_kind"] == plan["teacher_kind"]
            and provenance["teacher_weight"] == plan["teacher_weight"]
            and provenance["teacher_model_state_sha256"] == plan["teacher_model_state_sha256"]
            and provenance["training_plan_sha256"] == expected_plan_sha,
            "SDR checkpoint lineage or geometry differs")
    model.load_state_dict(payload["model"], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and all(torch.equal(v, frozen[name]) for name, v in model.named_buffers())
            and state_sha256(model.state_dict()) == receipt["model_state_sha256"], "Model tensors or fixed buffers differ")
    model.provenance = provenance
    return model, receipt


def save_generation(run, model, teacher, optimizer, step, plan, plan_sha, helpers):
    import numpy as np
    import torch
    from research.direct.latency58_sdr_teacher import PHASE, STUDENT_SHA256, STUDENT_STATE_SHA256

    require_space(plan, 350_000_000)
    require(state_sha256(teacher.state_dict()) == plan["teacher_model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Frozen teacher changed")
    journal = (run / "metrics.jsonl").read_bytes()
    helpers.validate_journal(journal, step, plan["config"])
    tensors = helpers.tensor_tree_cpu(model.state_dict(), torch)
    fingerprint = state_sha256(tensors)
    parent_provenance = copy.deepcopy(plan["parent_provenance"])
    provenance = copy.deepcopy(parent_provenance)
    provenance.update(initialization="sdr_recovery_from_user_accepted_teacher250",
                      parent_checkpoint={"kind": "inference", "path": str(PHASE / "teacher-half-canonical-001/model.pt"),
                                         "sha256": STUDENT_SHA256},
                      parent_model_state_sha256=STUDENT_STATE_SHA256,
                      initialized_model_state_sha256=STUDENT_STATE_SHA256,
                      parent_training_updates=5000, parent_provenance=parent_provenance,
                      training_updates=5000 + step, asymmetric_training_updates=750 + step,
                      pilot_updates=2750 + step, tail_updates=step, teacher_trial_updates=step,
                      sdr_trial_updates=step, teacher_kind=plan["teacher_kind"],
                      teacher_weight=plan["teacher_weight"], teacher_model_state_sha256=plan["teacher_model_state_sha256"],
                      training_objective="raw4_l1_plus_teacher_l1", training_plan_sha256=plan_sha,
                      training_precision=plan["precision_policy"])
    model_payload = {"schema": "latency58-sdr-inference-v1", "step": step, "model": tensors,
                     "model_state_sha256": fingerprint, "architecture": model.architecture_metadata,
                     "provenance": provenance, "plan_sha256": plan_sha}
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
    helpers.validate_rng(rng, torch)
    checkpoint_root = run / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    final = checkpoint_root / f"step-{step:06d}"
    pending = checkpoint_root / f"step-{step:06d}.pending"
    require(not final.exists() and not pending.exists(), "Preserve existing checkpoint generation")
    pending.mkdir()
    payloads = (("model.pt", model_payload), ("optimizer.pt", helpers.tensor_tree_cpu(optimizer.state_dict(), torch)),
                ("rng.pt", rng))
    for name, payload in payloads:
        with (pending / name).open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    with (pending / "metrics.jsonl").open("xb") as stream:
        stream.write(journal)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": "latency58-sdr-generation-v1", "step": step, "plan_sha256": plan_sha,
               "model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"],
               "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
               "optimizer_parameter_names": [name for name, _ in model.named_parameters()],
               "metrics_sha256": hashlib.sha256(journal).hexdigest(), "metrics_bytes": len(journal),
               "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pending.iterdir()}}
    helpers.atomic_json(pending / "receipt.json", receipt)
    helpers.fsync_dir(pending)
    pending.rename(final)
    helpers.fsync_dir(checkpoint_root)
    helpers.atomic_json(run / "latest.json", {"generation": str(final), "step": step,
                                              "receipt_sha256": sha(final / "receipt.json"), "plan_sha256": plan_sha})
    return receipt
