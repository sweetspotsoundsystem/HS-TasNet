"""Frozen C204 fine-tuning with an independently checked reduced teacher loss."""
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
from research.direct.hare_loss_ablation_checkpoint import bound, compare_inputs, audit_live
from research.direct.latency58_reduced_teacher_loss import VERSION, validate_microbatch


def validate_recipe(plan):
    from research.direct.latency58_cleanup_rebound_checkpoint import validate_recipe as validate_reference
    reference = bound(plan, plan["matched_positive_training_plan"])
    validate_reference(reference)
    shared = tuple(k for k in SHARED_CONTROL_KEYS if k not in ("stop_counted_bytes", "teacher_weight")) + (
        "teacher_mode", "counterfactual_version", "additional_loss_version", "additional_loss_weight")
    require(plan["schema"] == "latency58-reduced-teacher-training-v1" and type(plan["resource_only"]) is bool
            and plan["teacher_weight"] == .25 and reference["teacher_weight"] == .5
            and plan["additional_loss_weight"] == .5 and plan["reduced_teacher_version"] == VERSION
            and all(plan[k] == reference[k] for k in shared), "Teacher trial changes another operational field")
    require(plan["comparison_variable"] == "teacher_weight" and plan["stop_counted_bytes"] == 79_200_000_000
            and plan["outside_roots_reservation_bytes"] == 800_000_000
            and plan["checkpoint_and_quality_reserve_bytes"] == 400_000_000,
            "Different teacher trial or combined storage allowance")
    decision = bound(plan, plan["preparation_decision"])
    require(decision["comparison_variable"] == "teacher_weight" and decision["candidate_weight"] == .25
            and decision["control_weight"] == .5 and decision["maximum_production_updates"] == 250
            and decision["quality_endpoints"] == [250] and not decision["automatic_continuation"]
            and not decision["automatic_model_replacement"]
            and decision["matched_positive_training_plan"] == plan["matched_positive_training_plan"],
            "Different fixed teacher decision")
    functional = bound(plan, plan["functional_check"])
    require(functional["status"] == "pass" and functional["version"] == VERSION
            and functional["half_weight_control_loss_and_gradients_exact"]
            and all(plan["source_bindings"].get(p) == h == sha(p) for p, h in functional["source_bindings"].items()),
            "Loss functional check is stale or incomplete")
    item = plan["matched_positive_journal"]
    require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
            "Unbound half-teacher input journal")
    receipt = bound(plan, plan["matched_positive_receipt"])
    require(receipt["plan_sha256"] == plan["matched_positive_training_plan"]["sha256"]
            and receipt["metrics_sha256"] == item["sha256"] and receipt["step"] == 250,
            "Half-teacher journal belongs to another trial")
    return reference


def load_parent(plan):
    from research.direct.latency58_cleanup_rebound_checkpoint import load_parent as load_reference_parent
    return load_reference_parent(validate_recipe(plan))


def expected_provenance(plan, step, plan_sha):
    from research.direct.latency58_cleanup_rebound_checkpoint import expected_provenance as previous_provenance
    value = previous_provenance(plan, step, plan_sha)
    value.pop("cleanup_rebound_trial_updates", None)
    value.update(initialization="c204_reduced_teacher_fine_tune",
                 training_objective="quarter_ordinary_teacher_plus_controlled_deployed_truth",
                 comparison_variable="teacher_weight", reduced_teacher_trial_updates=step,
                 reduced_teacher_version=VERSION, matched_positive_training_plan=plan["matched_positive_training_plan"])
    return value


def read_generation(directory, *, expected_plan_sha=None, require_optimizer=True):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    require(receipt["schema"] == "latency58-reduced-teacher-generation-v1"
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
    require(payload["schema"] == "latency58-reduced-teacher-inference-v1" and payload["step"] == step
            and 0 < step <= plan["config"]["steps"] and receipt["carry_state"] == plan["carry_state"]
            and receipt["teacher_weight"] == plan["teacher_weight"] and receipt["reduced_teacher_version"] == VERSION
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


def validate_base_journal(journal, step, plan, helpers):
    import math
    from research.direct.latency58_drum_emphasis import DRUM_WEIGHT, VERSION

    require(plan["drum_weight"] == DRUM_WEIGHT == 2 and plan["objective_version"] == VERSION
            and plan["teacher_weight"] == .25, "Unexpected accumulated drum objective")
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
        require(plan["microbatch_size"] == row["microbatch_size"] == 4
                and plan["accumulation_steps"] == row["accumulation_steps"] == 4
                and plan["config"]["batch_size"] == 16
                and row["gradient_clips_this_update"] == row["adam_steps_this_update"] == 1
                and len(row["microbatches"]) == 4
                and row["drum_weight"] == plan["drum_weight"] and row["objective_version"] == VERSION,
                "Malformed accumulated drum update")
        micros = row["microbatches"]
        for index, micro in enumerate(micros):
            require(micro["micro_index"] == index and micro["batch_size"] == 4
                    and micro["first_sample_index"] == row["first_sample_index"] + 4 * index
                    and micro["next_sample_index"] == micro["first_sample_index"] + 4
                    and micro["initial_state_detached"]
                    and micro["data_hops"] == row["data_hops"] and micro["flush_hops"] == 1
                    and type(micro["deranged_examples"]) is int and 0 <= micro["deranged_examples"] <= 4
                    and abs(micro["loss"] - micro["supervised_loss"]
                            - plan["teacher_weight"] * micro["teacher_l1"]) < 1e-7,
                    "Malformed accumulation microbatch")
            for key in LOSS_KEYS:
                require(math.isfinite(micro[key]) and micro[key] >= 0, "Nonfinite microbatch objective")
            require(abs(micro["waveform_l1"] - (4 * micro["unweighted_waveform_l1"] + micro["raw_drum_l1"]) / 5) < 1e-7
                    and abs(micro["teacher_l1"] - (4 * micro["unweighted_teacher_l1"] + micro["teacher_drum_l1"]) / 5) < 1e-7
                    and abs(micro["supervised_loss"] - micro["waveform_l1"] - micro["projection_contribution"]) < 1e-7,
                    "Microbatch does not implement the normalized drum objective")
            for key in ("augmented_batch_sha256", "teacher_targets_sha256"):
                require(len(micro[key]) == 64 and all(c in "0123456789abcdef" for c in micro[key]),
                        "Malformed microbatch identity")
        for key in LOSS_KEYS:
            require(row[key] == sum(m[key] for m in micros) / 4, "Update does not average four microbatch losses")
        for key in ("augmented_batch_sha256", "teacher_targets_sha256"):
            require(row[key] == hashlib.sha256("".join(m[key] for m in micros).encode("ascii")).hexdigest(),
                    "Accumulated identity does not bind every microbatch")
        require(row["deranged_examples"] == sum(m["deranged_examples"] for m in micros),
                "Accumulated activity count differs")
    return rows


def validate_journal(journal, step, plan, helpers):
    validate_recipe(plan)
    rows = [json.loads(line) for line in journal.splitlines()]
    projected = copy.deepcopy(rows)
    reference = [json.loads(line) for line in Path(plan["matched_positive_journal"]["path"]).read_text().splitlines()]
    require(len(rows) == step and len(reference) == 250, "Incomplete teacher trial or control journal")
    for row, base, positive in zip(rows, projected, reference[:step], strict=True):
        compare_inputs(row, positive)
        require(row["teacher_weight"] == .25 and row["reduced_teacher_version"] == VERSION
                and row["additional_loss_weight"] == .5
                and row["additional_loss_version"] == plan["additional_loss_version"]
                and row["teacher_mode"] == plan["teacher_mode"] and row["arm"] == plan["arm"]
                and row["parameter_tensors"] == 21 and row["focused_augmentation"]
                and row["counterfactual_version"] == plan["counterfactual_version"]
                and row["augmentation_version"] == plan["augmentation_version"]
                and not row["local_mask_mixer"], "Reduced-teacher journal identity differs")
        for micro in row["microbatches"]:
            validate_microbatch(micro, .25, .5)
        for key in LOSS_KEYS:
            require(math.isfinite(row[key]) and row[key] >= 0
                    and row[key] == sum(m[key] for m in row["microbatches"]) / 4,
                    "Teacher trial loss does not average four microbatches")
        for key in IDENTITY_KEYS:
            require(row[key] == hashlib.sha256("".join(m[key] for m in row["microbatches"]).encode("ascii")).hexdigest(),
                    "Teacher trial aggregate identity differs")
        gradients = row["first_update_inherited_gradient_sha256"]
        require(len(gradients) == (21 if row["step"] == 1 else 0)
                and all(len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in gradients.values())
                and row["mixer_gradient_maxima_before_clip"] == {}, "Gradient inventory differs")
        require(row["controlled_examples"] == row["ordinary_examples"] == 8, "View participation differs")
        base["loss"] = base["base_loss"]
        for micro in base["microbatches"]:
            micro["loss"] = micro["base_loss"]
    # This retained validator checks the raw and teacher base. The actual added
    # deployed term was separately reconstructed above, before this projection.
    validate_base_journal(("\n".join(json.dumps(row, allow_nan=False) for row in projected) + "\n").encode(),
                          step, plan, helpers)
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
    payload = {"schema": "latency58-reduced-teacher-inference-v1", "step": step, "model": tensors,
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
    receipt = {"schema": "latency58-reduced-teacher-generation-v1", "step": step, "plan_sha256": plan_sha,
               "model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"], "carry_state": plan["carry_state"],
               "arm": plan["arm"],
               "teacher_mode": plan["teacher_mode"], "counterfactual_version": plan["counterfactual_version"],
               "additional_loss_version": plan["additional_loss_version"],
               "additional_loss_weight": plan["additional_loss_weight"],
               "teacher_weight": plan["teacher_weight"], "reduced_teacher_version": VERSION,
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
