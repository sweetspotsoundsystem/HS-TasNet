"""Frozen, physically aligned teachers for the 5.8 ms SDR recovery trial.

The C91 path uses the same literal streaming transform as its recorded
validation score. No offline padding/window normalization is substituted.
These helpers affect training targets only; the student graph is unchanged.
"""
from __future__ import annotations

from pathlib import Path

from research.direct.latency58_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "research/direct/runs/latency58"
C91_SHA256 = "86ead5c164c3a49a5fb91f8d02a8db01e7c684109e485021620b33b51a51387a"
STUDENT_SHA256 = "05a973af6efa3482e759f63cb3d268646a80f547a3bdc13679f0e228efba6222"
STUDENT_STATE_SHA256 = "1daa6edb7be90eb641817b788b5540c89de57b125e454bdf463469ffcd3366ea"
HANN_PARENT_SHA256 = "6b0feba723b8252f59061ae0a93c5d365419444507cabd55f3c1748fcf3eed15"


def load_initial_student():
    """Load the accepted canonical checkpoint on CPU without changing RNG."""
    import torch
    from research.direct.latency58_asymmetric_checkpoint import make_model
    from research.direct.latency58_teacher_checkpoint_v2 import load_model_state
    from research.direct.train_latency58 import state_sha256

    require(not torch.cuda.is_initialized(), "Construct and authenticate student before CUDA")
    with torch.random.fork_rng(devices=[]):
        model = make_model({"kind": "inference",
                            "path": str(PHASE / "raw4-b4-lr3e-5-pilot/model-step-002000.pt"),
                            "sha256": HANN_PARENT_SHA256})
        load_model_state(model, {"kind": "inference",
                                "path": str(PHASE / "teacher-half-canonical-001/model.pt"),
                                "sha256": STUDENT_SHA256})
    require(state_sha256(model.state_dict()) == STUDENT_STATE_SHA256, "Initial student tensors differ")
    return model


def load_teacher(kind, binding):
    """Authenticate a named teacher; caller moves it to CUDA after CPU checks."""
    import torch
    from research.direct.train_latency58 import state_sha256

    require(kind in ("cropped11", "c91"), "Unknown teacher family")
    require(not torch.cuda.is_initialized(), "Authenticate teacher before initializing CUDA")
    with torch.random.fork_rng(devices=[]):
        if kind == "cropped11":
            from research.direct.latency58_teacher import load_frozen_teacher
            model, identity = load_frozen_teacher(binding)
        else:
            from research.direct.evaluate import load_checkpoint
            path = Path(binding["path"])
            require(binding["sha256"] == C91_SHA256 and sha(path) == C91_SHA256,
                    "Use the authenticated refined C91 reference")
            model, identity = load_checkpoint(path, torch.device("cpu"), allow_custom_gains=True)
            require((model.segment_len, model.overlap_len, model.audio_channels, model.num_sources)
                    == (1024, 512, 2, 4) and not getattr(model, "causal_current_chunk", False),
                    "C91 streaming geometry differs")
            expected = model.output_source_scales.new_tensor((0.5, 0.5, 0.45, 0.56))
            require(torch.equal(model.output_source_scales, expected) and model.conv_decode.hann_window_baked,
                    "C91 native gains or baked decoder differ")
            require(sha(path) == C91_SHA256, "C91 checkpoint changed while loading")
    model.eval().requires_grad_(False)
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), "Teacher is not frozen")
    identity = dict(identity, teacher_kind=kind, model_state_sha256=state_sha256(model.state_dict()),
                    graph_alignment_samples=256 if kind == "cropped11" else 512,
                    source_order=["drums", "bass", "vocals", "other"],
                    output_policy="native Drums/Bass/Vocals; Other = physical mixture minus DBV",
                    teacher_only_during_training=True)
    return model, identity


def physical_targets(teacher, mixture, *, kind):
    """Reset each independent crop and remove exactly the teacher graph delay."""
    import torch
    import torch.nn.functional as F

    require(kind in ("cropped11", "c91") and mixture.ndim == 3 and mixture.shape[0] > 0
            and mixture.shape[1] == 2 and mixture.shape[-1] > 0 and mixture.dtype == torch.float32
            and not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
            "Invalid frozen teacher request")
    if kind == "cropped11":
        from research.direct.latency58_teacher import physical_teacher_targets
        targets = physical_teacher_targets(teacher, mixture)
    else:
        from research.evaluate import _init_batched_stateful_transform

        require(teacher.overlap_len == 512 and next(teacher.parameters()).device == mixture.device,
                "C91 hop or device differs")
        samples = mixture.shape[-1]
        with torch.no_grad(), torch.autocast(mixture.device.type, enabled=False):
            # Include one final zero callback; the output of callback n belongs
            # to the preceding input callback. A partial last input is padded once.
            padded = F.pad(mixture.detach(), (0, (-samples) % 512 + 512))
            transform = _init_batched_stateful_transform(
                teacher, batch_size=mixture.shape[0], device=mixture.device)
            pieces = [transform(chunk) for chunk in padded.split(512, dim=-1)]
            raw = torch.cat(pieces, dim=-1)[..., 512:512 + samples]
            dbv = raw[:, :3]
            targets = torch.cat((dbv, mixture.detach()[:, None] - dbv.sum(dim=1, keepdim=True)), dim=1)
            # Ordinary detached tensors are safe as constants in a student loss.
            targets = targets.detach().contiguous()
    require(targets.shape == (mixture.shape[0], 4, 2, mixture.shape[-1])
            and targets.dtype == torch.float32 and targets.device == mixture.device
            and not targets.requires_grad and targets.grad_fn is None and not torch.is_inference(targets)
            and bool(torch.isfinite(targets).all()), "Invalid or live teacher targets")
    return targets
