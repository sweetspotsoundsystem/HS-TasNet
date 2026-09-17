"""Authenticated parent and split generations for the student-context trial."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random

from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.train_latency58 import read, require, sha, state_sha256


def load_parent(plan):
    """Load a frozen accepted model or an audited SDR endpoint, on CPU only."""
    import torch
    from research.direct.latency58_sdr_teacher import load_initial_student

    require(not torch.cuda.is_initialized(), "Load parent before CUDA")
    parent = plan["parent"]
    require(sha(parent["checkpoint"]["path"]) == parent["checkpoint"]["sha256"], "Parent bytes changed")
    if parent["kind"] == "working_baseline":
        model = load_initial_student()
    else:
        from research.direct.latency58_sdr_checkpoint import load_model
        require(parent["kind"] == "sdr_candidate", "Unsupported context parent")
        binding = parent["training_plan"]
        require(sha(binding["path"]) == binding["sha256"], "Parent training plan changed")
        model, receipt = load_model(parent["generation"], read(binding["path"]), expected_plan_sha=binding["sha256"])
        require(parent["checkpoint"]["path"] == str(Path(parent["generation"]) / "model.pt")
                and receipt["files"]["model.pt"]["sha256"] == parent["checkpoint"]["sha256"], "Wrong parent generation")
    payload = torch.load(parent["checkpoint"]["path"], map_location="cpu", weights_only=True)
    require(payload["provenance"] == model.provenance == parent["provenance"]
            and state_sha256(payload["model"]) == state_sha256(model.state_dict()) == parent["model_state_sha256"]
            and payload["architecture"] == model.architecture_metadata
            and len(list(model.parameters())) == 21 and len(list(model.buffers())) == 6,
            "Context parent lineage or geometry differs")
    return model


def expected_provenance(plan, step, plan_sha):
    parent = plan["parent"]
    old = copy.deepcopy(parent["provenance"])
    value = copy.deepcopy(old)
    value.update(initialization="matched_student_context_trial", parent_provenance=old,
                 parent_checkpoint=copy.deepcopy(parent["checkpoint"]),
                 parent_model_state_sha256=parent["model_state_sha256"],
                 initialized_model_state_sha256=parent["model_state_sha256"],
                 parent_training_updates=old["training_updates"], tail_updates=step,
                 context_trial_updates=step, carry_state=plan["carry_state"],
                 warmup_samples=plan["warmup_samples"], scored_samples=plan["scored_samples"],
                 teacher_kind=plan["teacher_kind"], teacher_weight=plan["teacher_weight"],
                 teacher_model_state_sha256=plan["teacher_model_state_sha256"],
                 training_objective="raw4_l1_plus_context_teacher_l1",
                 training_plan_sha256=plan_sha, training_precision=plan["precision_policy"])
    for name in ("training_updates", "asymmetric_training_updates", "pilot_updates",
                 "teacher_trial_updates", "sdr_trial_updates"):
        value[name] = old.get(name, 0) + step
    return value


def read_generation(directory, *, expected_plan_sha=None, require_optimizer=True):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    require(receipt["schema"] == "latency58-context-generation-v1"
            and set(receipt["files"]) == {"model.pt", "optimizer.pt", "rng.pt", "metrics.jsonl"}
            and type(receipt["step"]) is int and receipt["step"] > 0
            and type(receipt["carry_state"]) is bool, "Malformed context generation")
    if expected_plan_sha is not None:
        require(receipt["plan_sha256"] == expected_plan_sha, "Generation belongs to another plan")
    for name, binding in receipt["files"].items():
        if name == "optimizer.pt" and not require_optimizer:
            continue
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == binding["bytes"]
                and sha(path) == binding["sha256"], "Generation bytes changed: " + name)
    return receipt


def load_model(directory, plan, *, expected_plan_sha):
    import torch

    receipt = read_generation(directory, expected_plan_sha=expected_plan_sha, require_optimizer=False)
    model = load_parent(plan)
    fixed = {name: value.clone() for name, value in model.named_buffers()}
    payload = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=True)
    step = receipt["step"]
    require(payload["schema"] == "latency58-context-inference-v1" and payload["step"] == step
            and 0 < step <= plan["config"]["steps"] and receipt["carry_state"] == plan["carry_state"]
            and receipt["teacher_kind"] == plan["teacher_kind"]
            and payload["plan_sha256"] == expected_plan_sha and payload["architecture"] == model.architecture_metadata
            and payload["model_state_sha256"] == receipt["model_state_sha256"]
            and payload["provenance"] == expected_provenance(plan, step, expected_plan_sha),
            "Context generation provenance differs")
    model.load_state_dict(payload["model"], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and all(torch.equal(v, fixed[name]) for name, v in model.named_buffers())
            and state_sha256(model.state_dict()) == receipt["model_state_sha256"], "Context tensors or fixed buffers differ")
    model.provenance = payload["provenance"]
    return model, receipt


def validate_journal(journal, step, plan, helpers):
    import math

    helpers.validate_journal(journal, step, plan["config"])
    rows = [json.loads(line) for line in journal.splitlines()]
    for row in rows:
        require(row["carry_state"] == plan["carry_state"] and row["initial_state_detached"]
                and row["teacher_kind"] == plan["teacher_kind"] and row["teacher_weight"] == plan["teacher_weight"]
                and row["warmup_samples"] == plan["warmup_samples"] and row["scored_samples"] == plan["scored_samples"]
                and row["data_hops"] == plan["scored_samples"] // 128 and row["flush_hops"] == 1
                and all(math.isfinite(row[k]) for k in ("loss", "supervised_loss", "teacher_l1", "grad_norm"))
                and abs(row["loss"] - row["supervised_loss"] - row["teacher_weight"] * row["teacher_l1"]) < 1e-7
                and all(len(row[k]) == 64 and all(c in "0123456789abcdef" for c in row[k])
                        for k in ("augmented_batch_sha256", "teacher_targets_sha256")), "Malformed context objective journal")
    return rows


def save_generation(run, model, teacher, optimizer, step, plan, plan_sha, helpers):
    import numpy as np
    import torch

    require_space(plan, 350_000_000)
    require(state_sha256(teacher.state_dict()) == plan["teacher_model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Frozen teacher changed")
    journal = (run / "metrics.jsonl").read_bytes()
    validate_journal(journal, step, plan, helpers)
    tensors = helpers.tensor_tree_cpu(model.state_dict(), torch)
    fingerprint = state_sha256(tensors)
    payload = {"schema": "latency58-context-inference-v1", "step": step, "model": tensors,
               "model_state_sha256": fingerprint, "architecture": model.architecture_metadata,
               "provenance": expected_provenance(plan, step, plan_sha), "plan_sha256": plan_sha}
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
    helpers.validate_rng(rng, torch)
    root = run / "checkpoints"
    root.mkdir(exist_ok=True)
    final, pending = (root / f"step-{step:06d}{suffix}" for suffix in ("", ".pending"))
    require(not final.exists() and not pending.exists(), "Preserve existing context generation")
    pending.mkdir()
    for name, value in (("model.pt", payload), ("optimizer.pt", helpers.tensor_tree_cpu(optimizer.state_dict(), torch)),
                        ("rng.pt", rng)):
        with (pending / name).open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    with (pending / "metrics.jsonl").open("xb") as stream:
        stream.write(journal)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": "latency58-context-generation-v1", "step": step, "plan_sha256": plan_sha,
               "model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"], "carry_state": plan["carry_state"],
               "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
               "optimizer_parameter_names": [name for name, _ in model.named_parameters()],
               "metrics_sha256": hashlib.sha256(journal).hexdigest(), "metrics_bytes": len(journal),
               "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pending.iterdir()}}
    helpers.atomic_json(pending / "receipt.json", receipt)
    helpers.fsync_dir(pending)
    pending.rename(final)
    helpers.fsync_dir(root)
    helpers.atomic_json(run / "latest.json", {"generation": str(final), "step": step,
                                              "receipt_sha256": sha(final / "receipt.json"), "plan_sha256": plan_sha})
    return receipt
