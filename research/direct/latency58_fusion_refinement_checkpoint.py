"""Authenticated inference and optimizer snapshots for fusion_refinement training."""
from __future__ import annotations

import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_full_magnitude_checkpoint import cpu_tree
from research.direct.latency58_fusion_refinement import Latency58FusionRefinementModel, VERSION

SCHEMA = "latency58-fusion-refinement-inference-v1"
GENERATION_SCHEMA = "latency58-fusion-refinement-generation-v1"
RESERVE_BYTES = 380_000_000


def load_model(binding):
    import torch
    path = Path(binding["path"])
    require(path.is_file() and not path.is_symlink() and sha(path) == binding["sha256"],
            "Fusion-refinement checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_payload(payload)


def load_payload(payload):
    import torch
    require(payload["schema"] == SCHEMA and type(payload["step"]) is int and payload["step"] > 0,
            "Unknown fusion_refinement checkpoint or invalid step")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = Latency58FusionRefinementModel()
    names = [name for name, _ in model.named_parameters()]
    require(payload["architecture"] == model.architecture_metadata
            and payload["parameter_names"] == names and len(names) == 26,
            "Fusion-refinement geometry or learned tensor inventory changed")
    # Validate before load_state_dict, which could otherwise cast an invalid dtype.
    require(set(payload["model"]) == set(model.state_dict())
            and all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()) for value in payload["model"].values())
            and state_sha256(payload["model"]) == payload["model_state_sha256"],
            "Invalid fusion_refinement inference tensors")
    model.load_state_dict(payload["model"], strict=True)
    provenance = payload["provenance"]
    require(state_sha256(dict(model.named_buffers())) == payload["fixed_buffers_sha256"]
            and model.fixed_residual_share.item() == 1 / 16
            and provenance["fusion_refinement_version"] == VERSION
            and provenance["fusion_refinement_updates"] == payload["step"]
            and provenance["training_updates"] == provenance["fusion_refinement_parent_updates"] + payload["step"]
            and provenance["fusion_refinement_training_plan_sha256"] == payload["plan_sha256"]
            and provenance["fusion_refinement_all_neural_parameters_trained"] is True,
            "Fusion-refinement fixed buffers or lineage changed")
    model.provenance = provenance
    return model.eval().requires_grad_(False), payload


def audit_live(model, optimizer, step, frozen):
    import torch
    parameters = list(model.parameters())
    require(type(model) is Latency58FusionRefinementModel and type(step) is int and step >= 0
            and len(parameters) == 26 and all(p.requires_grad for p in parameters)
            and len(optimizer.param_groups) == 1
            and [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in parameters]
            and set(frozen) == set(dict(model.named_buffers())),
            "Require all 26 learned tensors and one ordered optimizer group")
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in parameters)
            and all(torch.equal(value, frozen[name]) for name, value in model.named_buffers()),
            "Invalid model values or changed fixed buffers")
    require(set(optimizer.state) == (set() if step == 0 else set(parameters)), "Wrong live Adam inventory")
    for parameter in parameters:
        if step:
            state = optimizer.state[parameter]
            require(state["step"].item() == step
                    and all(state[k].shape == parameter.shape and state[k].dtype == torch.float32
                            and bool(torch.isfinite(state[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
                    and bool((state["exp_avg_sq"] >= 0).all()), "Invalid live Adam state")


def make_payloads(model, optimizer, step, plan, plan_sha):
    import numpy as np
    import random
    import torch
    require(type(step) is int and step > 0 and step == plan["config"]["steps"]
            and state_sha256(dict(model.named_buffers())) == plan["fixed_buffers_sha256"]
            and model.provenance["fusion_refinement_parent_model_state_sha256"] == plan["parent_model_state_sha256"],
            "Save only the declared fusion_refinement endpoint with unchanged fixed buffers")
    audit_live(model, optimizer, step, dict(model.named_buffers()))
    tensors = cpu_tree(model.state_dict())
    names = [name for name, _ in model.named_parameters()]
    provenance = {**model.provenance, "fusion_refinement_version": VERSION,
                  "fusion_refinement_parent_checkpoint": plan["parent_checkpoint"],
                  "fusion_refinement_parent_model_state_sha256": plan["parent_model_state_sha256"],
                  "fusion_refinement_parent_updates": plan["parent_training_updates"],
                  "fusion_refinement_updates": step, "training_updates": plan["parent_training_updates"] + step,
                  "fusion_refinement_all_neural_parameters_trained": True,
                  "fusion_refinement_training_plan_sha256": plan_sha,
                  "fusion_refinement_objective_version": plan["objective_version"],
                  "training_objective": plan["objective_version"], "training_precision": plan["precision_policy"],
                  "quality_measured": False}
    payload = {"schema": SCHEMA, "step": step, "model": tensors,
               "model_state_sha256": state_sha256(tensors), "architecture": model.architecture_metadata,
               "fixed_buffers_sha256": plan["fixed_buffers_sha256"], "parameter_names": names,
               "provenance": provenance, "plan_sha256": plan_sha}
    numpy_state = np.random.get_state()
    resume = {"optimizer": cpu_tree(optimizer.state_dict()), "parameter_names": names,
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
    require(not pending.exists() and not final.exists(), "Preserve existing fusion-refinement checkpoints")
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
            and receipt["metrics_sha256"] == sha(checkpoint.parent / "metrics.jsonl")
            and payload["fixed_buffers_sha256"] == plan["fixed_buffers_sha256"]
            and payload["provenance"]["fusion_refinement_parent_checkpoint"] == plan["parent_checkpoint"]
            and payload["provenance"]["fusion_refinement_parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and payload["provenance"]["fusion_refinement_parent_updates"] == plan["parent_training_updates"],
            "Saved fusion_refinement endpoint or parent differs from its plan")
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
                "Saved fusion_refinement closure or physical delay differs")
    return {"status": "pass", "step": payload["step"], "checkpoint": binding,
            "model_state_sha256": payload["model_state_sha256"], "parameter_names": payload["parameter_names"],
            "optimizer_sha256": receipt["files"]["optimizer.pt"]["sha256"], "saved_optimizer_tensor_count": 26,
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
    require(set(states) == set(range(26)) and len(groups) == 1 and groups[0]["params"] == list(range(26)),
            "Saved Adam inventory or parameter ordering changed")
    for index, state in states.items():
        require(state["step"].item() == payload["step"]
                and all(state[k].shape == parameters[index].shape and state[k].dtype == torch.float32
                        and bool(torch.isfinite(state[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
                and bool((state["exp_avg_sq"] >= 0).all()), "Invalid saved Adam moments")
