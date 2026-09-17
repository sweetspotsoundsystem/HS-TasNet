"""Frozen accepted-model targets aligned to physical training samples."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.latency58_checkpoint import require, sha


def load_frozen_teacher(export_plan_binding, *, device="cpu"):
    """Authenticate the accepted snapshot with its existing strict loader."""
    import torch
    from research.direct.latency_cropped1024_onnx import load_checkpoint, load_source

    path = Path(export_plan_binding["path"])
    require(sha(path) == export_plan_binding["sha256"], "Teacher authority changed")
    plan = json.loads(path.read_text())
    require(plan["schema"] == "cropped1024-onnx-export-plan-v1"
            and plan["snapshot"]["checkpoint"]["sha256"] ==
            "ac46729e5e4d379b09914a6e40ae927e09089b43fd4eef219ae7e034f355da65"
            and plan["snapshot"]["training_updates"] == 2250, "Use the user-accepted teacher")
    bindings = [plan[k] for k in ("exporter_source", "reusable_exporter_source", "model_source",
                                   "base_model_source", "export_helpers_source")]
    bindings += [plan["snapshot"][k] for k in ("checkpoint", "training_plan")]
    require(all(sha(v["path"]) == v["sha256"] for v in bindings), "Teacher source/checkpoint differs")
    reusable = load_source(plan["reusable_exporter_source"], "latency58_frozen_teacher_support")
    family = load_source(plan["model_source"], "latency58_frozen_teacher_family")
    teacher, identity = load_checkpoint(plan, torch, reusable, family)
    teacher = teacher.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    require(all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Teacher must be frozen")
    return teacher, identity


def physical_teacher_targets(teacher, mixture):
    """Same augmented mixture, one teacher flush, FP32 native four-stem targets.

The teacher's 256-sample graph delay is removed exactly once. No host queue,
level normalization, source remapping or teacher gradients enter the loss.
"""
    import torch
    import torch.nn.functional as F

    require(mixture.ndim == 3 and mixture.shape[1] == 2 and mixture.shape[-1] > 0
            and mixture.dtype == torch.float32 and not teacher.training
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Invalid teacher target request")
    samples = mixture.shape[-1]
    with torch.no_grad(), torch.autocast(mixture.device.type, enabled=False):
        padded = F.pad(mixture.detach(), (0, (-samples) % 256 + 256))
        output = teacher.render(padded)
        require(torch.equal(output.delayed_mixture[..., 256:256 + samples], mixture.detach()),
                "Teacher target lost physical sample alignment")
        targets = output.deployed[..., 256:256 + samples].detach().contiguous()
    require(targets.shape == (mixture.shape[0], 4, 2, samples) and targets.dtype == torch.float32
            and targets.device == mixture.device and not targets.requires_grad and targets.grad_fn is None
            and bool(torch.isfinite(targets).all()), "Invalid detached native teacher targets")
    return targets


def deployed_teacher_l1(student_deployed, teacher_targets):
    import torch
    import torch.nn.functional as F

    require(student_deployed.shape == teacher_targets.shape and student_deployed.ndim == 4
            and student_deployed.shape[1:3] == (4, 2)
            and student_deployed.dtype == teacher_targets.dtype == torch.float32
            and student_deployed.device == teacher_targets.device
            and not teacher_targets.requires_grad and teacher_targets.grad_fn is None, "Unaligned or live teacher target")
    return F.l1_loss(student_deployed, teacher_targets)
