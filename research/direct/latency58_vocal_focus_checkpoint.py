"""Authenticated parent, matched data journal and generations for vocal-focus pilots."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random

from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.train_latency58 import read, require, sha, state_sha256


from research.direct.latency58_sdr_drum_accum_checkpoint import (
    validate_journal as validate_base_journal, LOSS_KEYS)
from research.direct.latency58_vocal_focus_augmentation import VERSION as AUGMENTATION_VERSION

IDENTITY_KEYS = ("pristine_batch_sha256", "original_augmentation_sha256",
                 "augmentation_rng_before_sha256", "augmentation_rng_after_sha256")


def validate_recipe(plan):
    config = plan["config"]
    require(plan["schema"] == "latency58-vocal-focus-training-v1"
            and plan["arm"] in ("original", "focused", "focused_mixer")
            and type(plan["focused_augmentation"]) is bool
            and plan["focused_augmentation"] == (plan["arm"] != "original")
            and type(plan["local_mask_mixer"]) is bool
            and plan["local_mask_mixer"] == (plan["arm"] == "focused_mixer")
            and plan["augmentation_version"] == AUGMENTATION_VERSION
            and type(plan["resource_only"]) is bool
            and config["steps"] == 250 and config["batch_size"] == 16
            and config["workers"] == 2 and config["precision"] == "bf16"
            and config["crop_samples"] == 176128
            and plan["warmup_samples"] == plan["scored_samples"] == 88064
            and plan["carry_state"] is True and plan["microbatch_size"] == 4
            and plan["accumulation_steps"] == 4 and config["data_start"] >= 972000
            and config["data_start"] % 16 == 0
            and config["lr"] == 3e-5 and config["min_lr"] == 3e-6 and config["warmup"] == 25
            and plan["teacher_kind"] == "c91" and plan["teacher_weight"] == .5
            and plan["drum_weight"] == 2 and plan["mixer_initialization_seed"] == 20260921
            and plan["parameter_tensors"] == (25 if plan["local_mask_mixer"] else 21)
            and plan["automatic_continuation"] is False,
            "Unsupported fixed 250-update vocal-focus recipe")


def load_parent(plan):
    import importlib
    import torch
    validate_recipe(plan)
    require(not torch.cuda.is_initialized(), "Load the common parent before CUDA")
    parent = plan["parent"]
    require(parent["kind"] in ("working", "drum_accum", "history"), "Unknown common parent family")
    if parent["kind"] == "working":
        from research.direct.latency58_sdr_teacher import load_initial_student
        model = load_initial_student()
    else:
        binding = parent["quality_plan"]
        require(sha(binding["path"]) == binding["sha256"]
                and plan["source_bindings"].get(binding["path"]) == binding["sha256"],
                "Common parent quality plan changed or is unbound")
        quality = read(binding["path"])
        for name in ("model.pt", "receipt.json", "rng.pt", "metrics.jsonl"):
            path = str(Path(quality["generation"]) / name)
            require(plan["source_bindings"].get(path) == sha(path), "Unbound common parent generation")
        module = importlib.import_module("research.direct.evaluate_latency58_sdr_" + parent["kind"])
        model, _ = module.load_evaluation_model(quality)
        require(quality["checkpoint"] == parent["checkpoint"], "Common parent checkpoint differs")
    require(state_sha256(model.state_dict()) == parent["model_state_sha256"]
            and model.provenance == parent["provenance"]
            and model.architecture_metadata == parent["architecture"]
            and len(list(model.parameters())) == 21 and len(list(model.buffers())) == 6,
            "Common parent identity or geometry differs")
    if plan["local_mask_mixer"]:
        from research.direct.latency58_vocal_focus_model import LocalMaskMixerModel
        model = LocalMaskMixerModel.from_parent(model, initialization_seed=plan["mixer_initialization_seed"])
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"]
            and model.architecture_metadata == plan["architecture"]
            and len(list(model.parameters())) == plan["parameter_tensors"]
            and not torch.cuda.is_initialized(), "Initialized arm identity differs")
    return model


def audit_live(model, optimizer, step, frozen, torch, plan, helpers):
    parameters = list(model.parameters())
    count = plan["parameter_tensors"]
    require(len(parameters) == count
            and all(p.requires_grad and p.dtype == torch.float32 and p.device.type == "cuda"
                    and bool(torch.isfinite(p).all()) for p in parameters), "Malformed trainable parameter inventory")
    buffers = dict(model.named_buffers())
    require(len(buffers) == len(frozen) == 6 and set(buffers) == set(frozen)
            and all(torch.equal(value, frozen[name]) for name, value in buffers.items()), "Fixed buffers changed")
    require(len(optimizer.param_groups) == 1, "Expected one Adam group")
    group = optimizer.param_groups[0]
    require(len(group["params"]) == count and all(a is b for a, b in zip(group["params"], parameters)),
            "Adam parameter identity or order differs")
    ids = list(range(count))
    helpers.validate_adam_group({**group, "params": ids}, ids, step, plan["config"])
    require(set(optimizer.state) == (set(parameters) if step else set()), "Unexpected Adam state inventory")
    for parameter, state in optimizer.state.items():
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                and state["step"].shape == () and state["step"].dtype == torch.float32
                and state["step"].device.type == "cpu" and float(state["step"]) == step, "Adam step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            require(value.dtype == torch.float32 and value.shape == parameter.shape
                    and value.device == parameter.device and bool(torch.isfinite(value).all()), "Malformed Adam moment")
        require(bool((state["exp_avg_sq"] >= 0).all()), "Negative second moment")


def expected_provenance(plan, step, plan_sha):
    parent = plan["parent"]
    old = copy.deepcopy(parent["provenance"])
    value = copy.deepcopy(old)
    value.update(initialization="matched_vocal_focus_fine_tune", parent_provenance=old,
                 parent_checkpoint=copy.deepcopy(parent["checkpoint"]),
                 parent_model_state_sha256=parent["model_state_sha256"],
                 initialized_model_state_sha256=plan["initialized_model_state_sha256"],
                 parent_training_updates=old["training_updates"], tail_updates=step,
                 vocal_focus_trial_updates=step, carry_state=plan["carry_state"],
                 vocal_focus_arm=plan["arm"], focused_augmentation=plan["focused_augmentation"],
                 augmentation_version=plan["augmentation_version"],
                 local_mask_mixer=plan["local_mask_mixer"],
                 optimizer_initialization="fresh_adam", automatic_continuation=False,
                 gradient_accumulation_steps=plan["accumulation_steps"],
                 training_microbatch_size=plan["microbatch_size"],
                 trial_augmented_examples=step * plan["config"]["batch_size"],
                 trial_microbatches=step * plan["accumulation_steps"],
                 warmup_samples=plan["warmup_samples"], scored_samples=plan["scored_samples"],
                 teacher_kind=plan["teacher_kind"], teacher_weight=plan["teacher_weight"],
                 teacher_model_state_sha256=plan["teacher_model_state_sha256"],
                 training_objective="mean_four_b4_normalized_drum_raw4_and_context_teacher_l1",
                 drum_weight=plan["drum_weight"], objective_version=plan["objective_version"],
                 projection_policy="original_unweighted_cap_per_b4_microbatch",
                 learning_rate_peak=plan["config"]["lr"], learning_rate_floor=plan["config"]["min_lr"],
                 training_plan_sha256=plan_sha, training_precision=plan["precision_policy"])
    if plan["local_mask_mixer"]:
        from research.direct.latency58_vocal_focus_model import VERSION as MODEL_VERSION
        value.update(model_extension=MODEL_VERSION,
                     extension_initialization_seed=plan["mixer_initialization_seed"],
                     extension_parent_model_state_sha256=parent["model_state_sha256"],
                     extension_training_updates=step)
    for name in ("training_updates", "asymmetric_training_updates", "pilot_updates",
                 "teacher_trial_updates", "sdr_trial_updates"):
        value[name] = old.get(name, 0) + step
    return value


def read_generation(directory, *, expected_plan_sha=None, require_optimizer=True):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    require(receipt["schema"] == "latency58-vocal-focus-generation-v1"
            and set(receipt["files"]) == {"model.pt", "optimizer.pt", "rng.pt", "metrics.jsonl"}
            and type(receipt["step"]) is int and receipt["step"] == 250
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
    require(payload["schema"] == "latency58-vocal-focus-inference-v1" and payload["step"] == step
            and 0 < step <= plan["config"]["steps"] and receipt["carry_state"] == plan["carry_state"]
            and receipt["teacher_kind"] == plan["teacher_kind"] and receipt["arm"] == plan["arm"]
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
    validate_recipe(plan)
    rows = validate_base_journal(journal, step, plan, helpers)
    for row in rows:
        require(row["arm"] == plan["arm"] and row["focused_augmentation"] == plan["focused_augmentation"]
                and row["local_mask_mixer"] == plan["local_mask_mixer"]
                and row["augmentation_version"] == AUGMENTATION_VERSION
                and row["parameter_tensors"] == plan["parameter_tensors"], "Vocal-focus update identity differs")
        gradients = row["first_update_inherited_gradient_sha256"]
        require(len(gradients) == (21 if row["step"] == 1 else 0)
                and all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                        for value in gradients.values()), "Inherited gradient identity differs")
        mixer_gradients = row["mixer_gradient_maxima_before_clip"]
        require(len(mixer_gradients) == (4 if plan["local_mask_mixer"] else 0)
                and all(math.isfinite(v) and v >= 0 for v in mixer_gradients.values()),
                "Malformed added-parameter gradients")
        if plan["local_mask_mixer"] and row["step"] <= 2:
            for name, maximum in mixer_gradients.items():
                require(maximum == 0 if row["step"] == 1 and ".local." in name else maximum > 0,
                        "Zero-initialized mixer does not enter its expected gradient stages")
        for micro in row["microbatches"]:
            codes = [(micro["first_sample_index"] + i) % 4 for i in range(4)] if plan["focused_augmentation"] else [2] * 4
            require(micro["view_codes"] == codes and micro["model_and_backward_rng_unchanged"]
                    and micro["forced_views_use_pristine_sources"], "Vocal-focus view or RNG contract differs")
            for key in IDENTITY_KEYS:
                require(len(micro[key]) == 64 and all(c in "0123456789abcdef" for c in micro[key]),
                        "Malformed matched microbatch digest")
            if not plan["focused_augmentation"]:
                require(micro["original_augmentation_sha256"] == micro["augmented_batch_sha256"],
                        "Original-augmentation control was changed")
        for key in IDENTITY_KEYS:
            require(row[key] == hashlib.sha256("".join(m[key] for m in row["microbatches"]).encode("ascii")).hexdigest(),
                    "Update digest does not bind all four microbatches")
    return rows


def save_generation(run, model, teacher, optimizer, step, plan, plan_sha, helpers):
    import numpy as np
    import torch

    require(step == plan["config"]["steps"] == 250 and not plan["resource_only"], "Only the complete pilot is saved")
    require_space(plan, 350_000_000)
    require(state_sha256(teacher.state_dict()) == plan["teacher_model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Frozen teacher changed")
    journal = (run / "metrics.jsonl").read_bytes()
    validate_journal(journal, step, plan, helpers)
    tensors = helpers.tensor_tree_cpu(model.state_dict(), torch)
    fingerprint = state_sha256(tensors)
    payload = {"schema": "latency58-vocal-focus-inference-v1", "step": step, "model": tensors,
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
    receipt = {"schema": "latency58-vocal-focus-generation-v1", "step": step, "plan_sha256": plan_sha,
               "model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"], "carry_state": plan["carry_state"],
               "arm": plan["arm"],
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
