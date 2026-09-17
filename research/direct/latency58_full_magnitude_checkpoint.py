"""Self-contained inference snapshots for full-model magnitude-feature training."""
from __future__ import annotations

import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import state_sha256

SCHEMA = "latency58-full-magnitude-inference-v1"


def load_model(binding):
    import torch
    from research.direct.latency58_residual_model import VERSION, load_checkpoint
    from research.direct.latency58_magnitude import Latency58MagnitudeModel
    require(sha(binding["path"]) == binding["sha256"], "Direct-SDR checkpoint file changed")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    if payload["schema"] == VERSION:
        parent, payload = load_checkpoint(binding["path"], binding["sha256"])
        return Latency58MagnitudeModel.from_parent(parent), payload
    require(payload["schema"] == SCHEMA, "Unknown direct-SDR inference schema")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = Latency58MagnitudeModel()
    require(payload["architecture"] == model.architecture_metadata, "Direct-SDR model geometry changed")
    model.load_state_dict(payload["model"], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and state_sha256(model.state_dict()) == payload["model_state_sha256"]
            and state_sha256(dict(model.named_buffers())) == payload["fixed_buffers_sha256"]
            and model.fixed_residual_share.item() == 1 / 16, "Direct-SDR model tensor identity differs")
    model.provenance = payload["provenance"]
    require(model.provenance["direct_sdr_updates"] == model.provenance["magnitude_updates"] == payload["step"]
            and model.provenance["all_neural_parameters_trained"] is True
            and model.provenance["training_updates"] == model.provenance["direct_sdr_parent_updates"] + payload["step"],
            "Direct-SDR update count differs")
    return model.eval().requires_grad_(False), payload


def audit_live(model, optimizer, step, frozen):
    import torch
    parameters = list(model.parameters())
    require(len(parameters) == 22 and all(p.requires_grad for p in parameters)
            and len(optimizer.param_groups) == 1
            and [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in parameters]
            and set(frozen) == set(dict(model.named_buffers())), "Require all neural parameters and only fixed buffers frozen")
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters())
            and all(torch.equal(v, frozen[name]) for name, v in model.named_buffers()), "Model or fixed buffers invalid")
    require(len(optimizer.state) == (0 if step == 0 else len(list(model.parameters()))), "Wrong Adam inventory")
    for parameter in model.parameters():
        if step:
            state = optimizer.state[parameter]
            require(state["step"].item() == step and state["exp_avg"].shape == parameter.shape
                    and state["exp_avg_sq"].shape == parameter.shape
                    and bool(torch.isfinite(state["exp_avg"]).all())
                    and bool(torch.isfinite(state["exp_avg_sq"]).all())
                    and bool((state["exp_avg_sq"] >= 0).all()), "Invalid Adam moments or step")


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_generation(model, optimizer, step, plan, plan_sha, run):
    import numpy as np
    import random
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 370_000_000)
    pending, final = run / "checkpoint.pending", run / "checkpoint"
    require(not pending.exists() and not final.exists(), "Preserve direct-SDR checkpoints")
    pending.mkdir()
    tensors = cpu_tree(model.state_dict())
    provenance = {**model.provenance, "direct_sdr_parent_checkpoint": plan["parent_checkpoint"],
                  "direct_sdr_parent_model_state_sha256": plan["parent_model_state_sha256"],
                  "direct_sdr_parent_updates": plan["parent_training_updates"], "direct_sdr_updates": step,
                  "training_updates": plan["parent_training_updates"] + step,
                  "magnitude_updates": step, "all_neural_parameters_trained": True,
                  "inherited_tensors_frozen_during_magnitude_training": False,
                  "direct_sdr_objective_version": plan["objective_version"], "direct_sdr_training_plan_sha256": plan_sha,
                  "additional_training_updates": step, "training_objective": plan["objective_version"],
                  "training_precision": plan["precision_policy"], "teacher_kind": "none", "teacher_weight": 0.0,
                  "teacher_model_state_sha256": None, "teacher_used_in_direct_sdr_training": False}
    payload = {"schema": SCHEMA, "step": step, "model": tensors,
               "model_state_sha256": state_sha256(tensors), "architecture": model.architecture_metadata,
               "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
               "provenance": provenance, "plan_sha256": plan_sha}
    numpy_state = np.random.get_state()
    resume = {"optimizer": cpu_tree(optimizer.state_dict()), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
              "numpy_rng": [numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
              "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
              "step": step, "model_state_sha256": payload["model_state_sha256"], "plan_sha256": plan_sha}
    for name, data in (("model.pt", payload), ("optimizer.pt", resume)):
        with (pending / name).open("xb") as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
    write(pending / "receipt.json", {"schema": "latency58-full-magnitude-generation-v1", "step": step,
          "model_state_sha256": payload["model_state_sha256"], "plan_sha256": plan_sha,
          "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pending.iterdir()},
          "metrics_sha256": sha(run / "metrics.jsonl")})
    pending.rename(final)
    return {"path": str(final / "model.pt"), "sha256": sha(final / "model.pt")}
