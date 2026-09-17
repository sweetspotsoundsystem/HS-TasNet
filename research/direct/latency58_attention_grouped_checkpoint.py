"""Two-group Adam snapshots for the unchanged saved-attention architecture.

The inference tensor schema remains compatible with the original loader.
Generation and optimizer metadata distinguish this training-only strategy.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_full_magnitude_checkpoint import cpu_tree
from research.direct.latency58_temporal_attention import Latency58TemporalAttentionModel, VERSION, ADAPTERS
from research.direct.latency58_temporal_attention_checkpoint import (
    SCHEMA, load_model as load_original_model, load_payload as load_original_payload,
)

GENERATION_SCHEMA = "latency58-temporal-attention-grouped-generation-v1"
OPTIMIZER_SCHEMA = "latency58-attention-backbone-two-group-adam-v1"
SCHEDULE_SCHEMA = "latency58-warmup-cosine-backbone-with-attention-multiplier-v1"
RESERVE_BYTES = 380_000_000
GROUP_NAMES = ("backbone", "attention")


def named_groups(model):
    require(type(model) is Latency58TemporalAttentionModel, "Require the unchanged attention architecture")
    named = list(model.named_parameters())
    core = [(name, parameter) for name, parameter in named if name not in ADAPTERS]
    attention = [(name, parameter) for name, parameter in named if name in ADAPTERS]
    require(len(core) == 26 and tuple(name for name, _ in attention) == ADAPTERS
            and [name for name, _ in core + attention] == [name for name, _ in named],
            "Two groups must preserve the original thirty-tensor order")
    return core, attention


def optimizer_configuration(multiplier):
    require(type(multiplier) in (float, int) and math.isfinite(multiplier) and multiplier > 1,
            "Require a finite attention learning-rate multiplier greater than one")
    return {"schema": OPTIMIZER_SCHEMA, "attention_lr_multiplier": float(multiplier),
            "group_names": list(GROUP_NAMES), "group_sizes": [26, 4]}


def make_optimizer(model, *, lr, attention_lr_multiplier):
    import torch
    configuration = optimizer_configuration(attention_lr_multiplier)
    require(type(lr) in (float, int) and math.isfinite(lr) and lr > 0, "Require a positive finite backbone rate")
    groups = [{"params": [parameter for _, parameter in group], "group_name": name,
               "lr_scale": scale, "lr": float(lr) * scale}
              for name, group, scale in zip(GROUP_NAMES, named_groups(model),
                  (1., configuration["attention_lr_multiplier"]), strict=True)]
    return torch.optim.Adam(groups, lr=float(lr), foreach=False)


def set_learning_rate(optimizer, backbone_lr):
    require(type(backbone_lr) in (float, int) and math.isfinite(backbone_lr) and backbone_lr > 0,
            "Require a positive finite scheduled backbone rate")
    require(len(optimizer.param_groups) == 2
            and [group["group_name"] for group in optimizer.param_groups] == list(GROUP_NAMES),
            "Require named backbone and attention groups")
    for group in optimizer.param_groups:
        group["lr"] = float(backbone_lr) * group["lr_scale"]


def scheduled_backbone_lr(config, update_index):
    require(type(update_index) is int and 0 <= update_index < config["steps"]
            and type(config["warmup"]) is int and 0 <= config["warmup"] < config["steps"]
            and math.isfinite(config["lr"]) and math.isfinite(config["min_lr"])
            and config["lr"] >= config["min_lr"] > 0, "Invalid grouped-optimizer schedule")
    if update_index < config["warmup"]:
        return config["lr"] * (update_index + 1) / config["warmup"]
    phase = (update_index - config["warmup"]) / max(1, config["steps"] - 1 - config["warmup"])
    return config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * phase))


def validate_groups(groups, expected_parameters, multiplier):
    configuration = optimizer_configuration(multiplier)
    require(len(groups) == 2 and len(expected_parameters) == 2, "Require two ordered Adam groups")
    for group, parameters, name, scale in zip(groups, expected_parameters, GROUP_NAMES,
            (1., configuration["attention_lr_multiplier"]), strict=True):
        require(group["params"] == parameters and group.get("group_name") == name
                and group.get("lr_scale") == scale
                and type(group["lr"]) in (float, int) and math.isfinite(group["lr"]) and group["lr"] > 0
                and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
                and group["weight_decay"] == 0 and group["amsgrad"] is False
                and group["foreach"] is False and group["maximize"] is False
                and group["capturable"] is False and group["differentiable"] is False
                and group.get("fused") is None and not group.get("decoupled_weight_decay", False),
                "Adam group membership, rate or hyperparameters changed")
    require(groups[1]["lr"] == groups[0]["lr"] * configuration["attention_lr_multiplier"],
            "Attention-to-backbone learning-rate ratio changed")


def load_payload(payload):
    model, recovered = load_original_payload(payload)
    configuration = recovered["provenance"].get("optimizer_configuration", {})
    require(configuration == optimizer_configuration(configuration.get("attention_lr_multiplier", 0)),
            "Grouped inference provenance changed")
    return model, recovered


def load_model(binding):
    model, payload = load_original_model(binding)
    configuration = payload["provenance"].get("optimizer_configuration", {})
    require(configuration == optimizer_configuration(configuration.get("attention_lr_multiplier", 0)),
            "Grouped inference provenance changed")
    return model, payload


def audit_live(model, optimizer, step, frozen, *, attention_lr_multiplier):
    import torch
    named = named_groups(model)
    parameters = list(model.parameters())
    require(type(optimizer) is torch.optim.Adam and type(step) is int and step >= 0
            and all(parameter.requires_grad for parameter in parameters)
            and set(frozen) == set(dict(model.named_buffers())), "Require all thirty learned tensors and fixed buffers")
    groups = [{**group, "params": [id(parameter) for parameter in group["params"]]}
              for group in optimizer.param_groups]
    validate_groups(groups, [[id(parameter) for _, parameter in group] for group in named], attention_lr_multiplier)
    require(all(parameter.dtype == torch.float32 and bool(torch.isfinite(parameter).all()) for parameter in parameters)
            and all(torch.equal(value, frozen[name]) for name, value in model.named_buffers()),
            "Invalid model values or changed fixed buffers")
    require(set(optimizer.state) == (set() if step == 0 else set(parameters)), "Wrong live Adam inventory")
    for parameter in parameters:
        if step:
            state = optimizer.state[parameter]
            require(state["step"].dtype == torch.float32 and state["step"].ndim == 0 and state["step"].item() == step
                    and all(state[key].shape == parameter.shape and state[key].dtype == torch.float32
                            and bool(torch.isfinite(state[key]).all()) for key in ("exp_avg", "exp_avg_sq"))
                    and bool((state["exp_avg_sq"] >= 0).all()), "Invalid live Adam state")


def make_payloads(model, optimizer, step, plan, plan_sha):
    import numpy as np
    import random
    import torch
    require(type(step) is int and step > 0 and step == plan["config"]["steps"]
            and state_sha256(dict(model.named_buffers())) == plan["fixed_buffers_sha256"]
            and model.provenance["temporal_attention_parent_model_state_sha256"] == plan["parent_model_state_sha256"],
            "Save only the declared temporal_attention endpoint with unchanged fixed buffers")
    require(plan["optimizer_schema"] == OPTIMIZER_SCHEMA and plan["optimizer_schedule"] == SCHEDULE_SCHEMA,
            "Wrong grouped-optimizer plan")
    configuration = optimizer_configuration(plan["attention_lr_multiplier"])
    audit_live(model, optimizer, step, dict(model.named_buffers()),
               attention_lr_multiplier=plan["attention_lr_multiplier"])
    require(optimizer.param_groups[0]["lr"] == scheduled_backbone_lr(plan["config"], step - 1),
            "Saved backbone rate differs from the declared schedule endpoint")
    tensors = cpu_tree(model.state_dict())
    names = [name for name, _ in model.named_parameters()]
    provenance = {**model.provenance, "temporal_attention_version": VERSION,
                  "temporal_attention_parent_checkpoint": plan["parent_checkpoint"],
                  "temporal_attention_parent_model_state_sha256": plan["parent_model_state_sha256"],
                  "temporal_attention_parent_updates": plan["parent_training_updates"],
                  "temporal_attention_updates": step, "training_updates": plan["parent_training_updates"] + step,
                  "temporal_attention_all_neural_parameters_trained": True,
                  "temporal_attention_training_plan_sha256": plan_sha,
                  "temporal_attention_objective_version": plan["objective_version"],
                  "training_objective": plan["objective_version"], "training_precision": plan["precision_policy"],
                  "optimizer_configuration": configuration, "quality_measured": False}
    payload = {"schema": SCHEMA, "step": step, "model": tensors,
               "model_state_sha256": state_sha256(tensors), "architecture": model.architecture_metadata,
               "fixed_buffers_sha256": plan["fixed_buffers_sha256"], "parameter_names": names,
               "provenance": provenance, "plan_sha256": plan_sha}
    numpy_state = np.random.get_state()
    resume = {"optimizer": cpu_tree(optimizer.state_dict()), "parameter_names": names,
              "optimizer_configuration": configuration,
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
              "python_rng": random.getstate(),
              "numpy_rng": [numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
              "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
              "step": step, "model_state_sha256": payload["model_state_sha256"], "plan_sha256": plan_sha}
    return payload, resume


def save_generation(model, optimizer, step, plan, plan_sha, run):
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, RESERVE_BYTES)
    payload, resume = make_payloads(model, optimizer, step, plan, plan_sha)
    pending, final = run / "checkpoint.pending", run / "checkpoint"
    require(not pending.exists() and not final.exists(), "Preserve existing temporal-attention checkpoints")
    pending.mkdir()
    names = payload["parameter_names"]
    for name, data in (("model.pt", payload), ("optimizer.pt", resume)):
        with (pending / name).open("xb") as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
    write(pending / "receipt.json", {"schema": GENERATION_SCHEMA, "step": step,
          "model_state_sha256": payload["model_state_sha256"], "plan_sha256": plan_sha,
          "parameter_names": names,
          "optimizer_configuration": payload["provenance"]["optimizer_configuration"],
          "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pending.iterdir()},
          "metrics_sha256": sha(run / "metrics.jsonl")})
    # Flush the receipt and directory entries before publishing the generation.
    with (pending / "receipt.json").open("rb") as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(pending, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    pending.rename(final)
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(final / "model.pt"), "sha256": sha(final / "model.pt")}


def audit_saved(binding, plan, plan_sha):
    import torch
    model, payload = load_model(binding)
    checkpoint = Path(binding["path"]).parent
    receipt = read(checkpoint / "receipt.json")
    require(receipt["schema"] == GENERATION_SCHEMA and set(receipt["files"]) == {"model.pt", "optimizer.pt"}
            and receipt["plan_sha256"] == payload["plan_sha256"] == plan_sha
            and receipt["step"] == payload["step"] == plan["config"]["steps"]
            and receipt["model_state_sha256"] == payload["model_state_sha256"]
            and receipt["parameter_names"] == payload["parameter_names"]
            and receipt["optimizer_configuration"] == payload["provenance"]["optimizer_configuration"]
            == optimizer_configuration(plan["attention_lr_multiplier"])
            and receipt["metrics_sha256"] == sha(checkpoint.parent / "metrics.jsonl")
            and payload["fixed_buffers_sha256"] == plan["fixed_buffers_sha256"]
            and payload["provenance"]["temporal_attention_parent_checkpoint"] == plan["parent_checkpoint"]
            and payload["provenance"]["temporal_attention_parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and payload["provenance"]["temporal_attention_parent_updates"] == plan["parent_training_updates"],
            "Saved temporal_attention endpoint or parent differs from its plan")
    for name, expected in receipt["files"].items():
        path = checkpoint / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"]
                and sha(path) == expected["sha256"], "Saved generation bytes changed: " + name)
    resume = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    audit_resume(model, payload, resume, plan, plan_sha)
    with torch.inference_mode():
        signal = torch.linspace(-.1, .1, 2 * 8 * 128).reshape(1, 2, 8 * 128)
        first, second = model.render(signal), model.render(signal)
        require(torch.equal(first.deployed.view(torch.int32), second.deployed.view(torch.int32))
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(first.state, second.state, strict=True)), "Saved reset replay differs")
        closure = float((first.deployed.sum(1) - first.delayed_mixture).abs().max())
        require(closure < 1e-6 and model.algorithmic_latency_samples == 256,
                "Saved temporal_attention closure or physical delay differs")
    return {"status": "pass", "step": payload["step"], "checkpoint": binding,
            "model_state_sha256": payload["model_state_sha256"], "parameter_names": payload["parameter_names"],
            "optimizer_sha256": receipt["files"]["optimizer.pt"]["sha256"], "saved_optimizer_tensor_count": 30,
            "optimizer_group_count": 2, "optimizer_configuration": receipt["optimizer_configuration"],
            "fixed_buffers_sha256": payload["fixed_buffers_sha256"], "exact_reset_replay": True,
            "closure_max_abs": closure, "algorithmic_latency_samples": 256, "quality_measured": False}


def audit_resume(model, payload, resume, plan, plan_sha):
    """Check optimizer endpoint, order and moments after RAM or file loading."""
    import torch
    require(resume["step"] == payload["step"] and resume["model_state_sha256"] == payload["model_state_sha256"]
            and resume["plan_sha256"] == plan_sha and resume["parameter_names"] == payload["parameter_names"]
            and resume["next_sample_index"] == plan["config"]["data_start"] + payload["step"] * plan["config"]["batch_size"],
            "Saved optimizer belongs to another endpoint")
    parameters = list(model.parameters())
    states, groups = resume["optimizer"]["state"], resume["optimizer"]["param_groups"]
    require(plan["optimizer_schema"] == OPTIMIZER_SCHEMA and plan["optimizer_schedule"] == SCHEDULE_SCHEMA
            and resume["optimizer_configuration"] == payload["provenance"]["optimizer_configuration"]
            == optimizer_configuration(plan["attention_lr_multiplier"])
            and set(states) == set(range(30)), "Saved grouped Adam inventory or provenance changed")
    named_groups(model)
    validate_groups(groups, [list(range(26)), list(range(26, 30))], plan["attention_lr_multiplier"])
    require(groups[0]["lr"] == scheduled_backbone_lr(plan["config"], payload["step"] - 1),
            "Saved backbone rate differs from the declared schedule endpoint")
    for index, state in states.items():
        require(state["step"].dtype == torch.float32 and state["step"].ndim == 0 and state["step"].item() == payload["step"]
                and all(state[k].shape == parameters[index].shape and state[k].dtype == torch.float32
                        and bool(torch.isfinite(state[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
                and bool((state["exp_avg_sq"] >= 0).all()), "Invalid saved Adam moments")
