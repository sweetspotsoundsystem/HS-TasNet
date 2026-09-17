"""Self-contained, frozen-parent inference checkpoints for the magnitude adapter."""
from __future__ import annotations

import os
from pathlib import Path

from research.direct.run_latency58_quality import require, sha, write
from research.direct.train_latency58 import state_sha256

SCHEMA = "latency58-magnitude-sdr-inference-v1"


def load_model(binding):
    import torch
    from research.direct.latency58_magnitude import ADAPTER, Latency58MagnitudeModel
    from research.direct.latency58_residual_model import VERSION as PARENT_SCHEMA
    require(sha(binding["path"]) == binding["sha256"], "Magnitude checkpoint file changed")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    if payload["schema"] == PARENT_SCHEMA:
        from research.direct.latency58_direct_sdr_checkpoint import load_model as load_parent
        parent, payload = load_parent(binding)
        return Latency58MagnitudeModel.from_parent(parent), payload
    require(payload["schema"] == SCHEMA, "Unknown magnitude checkpoint schema")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = Latency58MagnitudeModel()
    require(payload["architecture"] == model.architecture_metadata, "Magnitude architecture differs")
    model.load_state_dict(payload["model"], strict=True)
    model.provenance = payload["provenance"]
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and state_sha256(model.state_dict()) == payload["model_state_sha256"]
            and state_sha256(dict(model.named_buffers())) == payload["fixed_buffers_sha256"]
            and state_sha256({k: v for k, v in model.state_dict().items() if k != ADAPTER})
                == model.provenance["magnitude_parent_state_sha256"], "Model or frozen inherited tensors differ")
    require(model.provenance["magnitude_updates"] == payload["step"]
            and model.provenance["training_updates"] == model.provenance["direct_sdr_parent_updates"] + payload["step"],
            "Magnitude update count differs")
    return model.eval().requires_grad_(False), payload


def audit_live(model, optimizer, step, frozen):
    import torch
    from research.direct.latency58_magnitude import ADAPTER
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    require([n for n, p in trainable] == [ADAPTER] and len(optimizer.param_groups) == 1
            and len(optimizer.param_groups[0]["params"]) == 1
            and optimizer.param_groups[0]["params"][0] is trainable[0][1], "Optimizer must update only the magnitude projection")
    require(set(frozen) == set(model.state_dict()) - {ADAPTER}
            and all(torch.equal(v, model.state_dict()[n]) for n, v in frozen.items())
            and all(p.grad is None for n, p in model.named_parameters() if n != ADAPTER)
            and all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Inherited tensors changed or gradients are nonfinite")
    require(len(optimizer.state) == (0 if step == 0 else 1), "Wrong Adam inventory")
    if step:
        parameter = trainable[0][1]
        state = optimizer.state[parameter]
        require(state["step"].item() == step and state["exp_avg"].shape == parameter.shape
                and state["exp_avg_sq"].shape == parameter.shape
                and bool(torch.isfinite(state["exp_avg"]).all()) and bool(torch.isfinite(state["exp_avg_sq"]).all())
                and bool((state["exp_avg_sq"] >= 0).all()), "Invalid magnitude Adam state")


def save_generation(model, optimizer, step, plan, plan_sha, run):
    import numpy as np
    import random
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import cpu_tree
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 140_000_000)
    pending, final = run / "checkpoint.pending", run / "checkpoint"
    require(not pending.exists() and not final.exists(), "Preserve magnitude checkpoints")
    pending.mkdir()
    tensors = cpu_tree(model.state_dict())
    provenance = {**model.provenance, "direct_sdr_parent_checkpoint": plan["parent_checkpoint"],
                  "direct_sdr_parent_model_state_sha256": plan["parent_model_state_sha256"],
                  "direct_sdr_parent_updates": plan["parent_training_updates"], "direct_sdr_updates": step,
                  "magnitude_updates": step, "training_updates": plan["parent_training_updates"] + step,
                  "direct_sdr_objective_version": plan["objective_version"], "direct_sdr_training_plan_sha256": plan_sha,
                  "additional_training_updates": step, "training_objective": plan["objective_version"],
                  "training_precision": plan["precision_policy"], "teacher_kind": "none", "teacher_weight": 0.,
                  "teacher_model_state_sha256": None, "teacher_used_in_direct_sdr_training": False,
                  "inherited_tensors_frozen_during_magnitude_training": True}
    payload = {"schema": SCHEMA, "step": step, "model": tensors, "model_state_sha256": state_sha256(tensors),
               "architecture": model.architecture_metadata, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
               "provenance": provenance, "plan_sha256": plan_sha}
    numpy_state = np.random.get_state()
    resume = {"optimizer": cpu_tree(optimizer.state_dict()), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
              "numpy_rng": [numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
              "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
              "step": step, "model_state_sha256": payload["model_state_sha256"], "plan_sha256": plan_sha,
              "trainable_parameter_names": plan["trainable_parameter_names"]}
    for name, value in (("model.pt", payload), ("optimizer.pt", resume)):
        with (pending / name).open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    write(pending / "receipt.json", {"schema": "latency58-magnitude-generation-v1", "step": step,
          "model_state_sha256": payload["model_state_sha256"], "plan_sha256": plan_sha,
          "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pending.iterdir()},
          "metrics_sha256": sha(run / "metrics.jsonl")})
    pending.rename(final)
    return {"path": str(final / "model.pt"), "sha256": sha(final / "model.pt")}
