"""Matched loss-only ablation: independent plan and journal checks, retained generation I/O."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from research.direct.train_latency58 import read, require, sha, state_sha256
from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.latency58_controlled_deployed_checkpoint import LOSS_KEYS, IDENTITY_KEYS, SHARED_CONTROL_KEYS
from research.direct.latency58_counterfactual_journal import rng_state_sha256


def bound(plan, item):
    require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
            "Unbound ablation prerequisite")
    return read(item["path"])


def validate_recipe(plan):
    from research.direct.latency58_leader_cleanup_checkpoint_v2 import validate_recipe as validate_positive
    reference = bound(plan, plan["matched_positive_training_plan"])
    validate_positive(reference)
    shared = tuple(k for k in SHARED_CONTROL_KEYS if k != "stop_counted_bytes") + (
        "teacher_mode", "counterfactual_version", "additional_loss_version")
    require(plan["schema"] == "hare-loss-ablation-training-v1" and type(plan["resource_only"]) is bool
            and plan["additional_loss_weight"] == 0 and reference["additional_loss_weight"] == .5
            and all(plan[k] == reference[k] for k in shared), "Ablation changes more than the added loss")
    require(plan["comparison_variable"] == "additional_loss_weight"
            and plan["stop_counted_bytes"] == 79_200_000_000
            and plan["outside_roots_reservation_bytes"] == 800_000_000
            and plan["checkpoint_and_quality_reserve_bytes"] == 400_000_000,
            "Different comparison or combined storage allowance")
    decision = bound(plan, plan["preparation_decision"])
    require(decision["comparison_variable"] == "additional_loss_weight"
            and decision["control_weight"] == 0 and decision["positive_weight"] == .5
            and decision["matched_positive_training_plan"] == plan["matched_positive_training_plan"]
            and decision["quality_endpoints"] == [250] and decision["maximum_production_updates"] == 250
            and not decision["automatic_continuation"] and not decision["automatic_model_replacement"],
            "Different prospective ablation")
    item = plan["matched_positive_journal"]
    require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
            "Unbound matched positive input journal")
    receipt = bound(plan, plan["matched_positive_receipt"])
    require(receipt["plan_sha256"] == plan["matched_positive_training_plan"]["sha256"]
            and receipt["metrics_sha256"] == item["sha256"] and receipt["step"] == 250,
            "Matched positive journal belongs to a different run")
    return reference


def load_parent(plan):
    from research.direct.latency58_leader_cleanup_checkpoint_v2 import load_parent as load_positive_parent
    return load_positive_parent(validate_recipe(plan))


def compare_inputs(actual, positive):
    keys = ("pristine_batch_sha256", "original_augmentation_sha256", "augmented_batch_sha256",
            "teacher_targets_sha256", "augmentation_rng_before_sha256", "augmentation_rng_after_sha256",
            "view_codes", "first_sample_index", "next_sample_index", "deranged_examples")
    require(actual["step"] == positive["step"] and actual["lr"] == positive["lr"],
            "Matched update or learning rate differs")
    require(len(actual["microbatches"]) == len(positive["microbatches"]) == 4,
            "Matched microbatch count differs")
    for a, b in zip(actual["microbatches"], positive["microbatches"], strict=True):
        require(all(a[k] == b[k] for k in keys), "Ablation inputs, teacher targets or RNG differ")
        require(a["model_and_backward_rng_unchanged"] and a["forced_views_use_pristine_sources"],
                "Ablation view or RNG contract failed")


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
    value.update(initialization="hare_matched_zero_deployed_truth_ablation",
                 training_objective="ordinary_only_teacher_zero_deployed_truth",
                 leader_cleanup_trial_updates=step,
                 counterfactual_version=plan["counterfactual_version"],
                 teacher_mode=plan["teacher_mode"], teacher_batch_divisor="all_four_examples",
                 additional_loss_version=plan["additional_loss_version"],
                 additional_loss_weight=plan["additional_loss_weight"],
                 deployed_truth_stem_weights=[2, 1, 1, 1], deployed_truth_divisor=20,
                 matched_positive_training_plan=plan["matched_positive_training_plan"],
                 comparison_variable="additional_loss_weight",
                 matched_loss_experiment=True, matched_quality_effect_not_yet_evaluated=True)
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
    require(receipt["schema"] == "hare-loss-ablation-generation-v1"
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
    require(payload["schema"] == "hare-loss-ablation-inference-v1" and payload["step"] == step
            and 0 < step <= plan["config"]["steps"] and receipt["carry_state"] == plan["carry_state"]
            and receipt["teacher_kind"] == plan["teacher_kind"] and receipt["arm"] == plan["arm"]
            and receipt["teacher_mode"] == plan["teacher_mode"]
            and receipt["counterfactual_version"] == plan["counterfactual_version"]
            and receipt["additional_loss_version"] == plan["additional_loss_version"]
            and receipt["additional_loss_weight"] == plan["additional_loss_weight"]
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
    from research.direct.latency58_sdr_drum_accum_checkpoint import validate_journal as validate_base
    from research.direct.latency58_controlled_deployed_journal import validate_microbatch
    validate_recipe(plan)
    rows = validate_base(journal, step, plan, helpers)
    reference = [json.loads(line) for line in Path(plan["matched_positive_journal"]["path"]).read_text().splitlines()]
    for row, positive in zip(rows, reference[:step], strict=True):
        compare_inputs(row, positive)
        require(row["additional_loss_weight"] == 0 and row["additional_loss_version"] == plan["additional_loss_version"]
                and row["teacher_mode"] == plan["teacher_mode"] and row["arm"] == plan["arm"]
                and row["parameter_tensors"] == 21 and row["focused_augmentation"]
                and not row["local_mask_mixer"], "Ablation journal identity differs")
        for micro in row["microbatches"]:
            validate_microbatch(micro, 0)
        for key in LOSS_KEYS:
            require(math.isfinite(row[key]) and row[key] >= 0
                    and row[key] == sum(m[key] for m in row["microbatches"]) / 4,
                    "Ablation loss does not average its four microbatches")
        for key in IDENTITY_KEYS:
            require(row[key] == hashlib.sha256("".join(m[key] for m in row["microbatches"]).encode("ascii")).hexdigest(),
                    "Ablation aggregate identity differs")
        require(row["controlled_examples"] == row["ordinary_examples"] == 8,
                "Ablation example participation differs")
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
    payload = {"schema": "hare-loss-ablation-inference-v1", "step": step, "model": tensors,
               "model_state_sha256": fingerprint, "architecture": model.architecture_metadata,
               "provenance": expected_provenance(plan, step, plan_sha), "plan_sha256": plan_sha}
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
    helpers.validate_rng(rng, torch)
    require(rng_state_sha256() == bound(plan, plan["matched_positive_receipt"])["final_rng_state_sha256"],
            "Final RNG state differs from the matched positive arm")
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
    receipt = {"schema": "hare-loss-ablation-generation-v1", "step": step, "plan_sha256": plan_sha,
               "model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"], "carry_state": plan["carry_state"],
               "arm": plan["arm"],
               "teacher_mode": plan["teacher_mode"], "counterfactual_version": plan["counterfactual_version"],
               "additional_loss_version": plan["additional_loss_version"],
               "additional_loss_weight": plan["additional_loss_weight"],
               "final_rng_state_sha256": rng_state_sha256(),
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
