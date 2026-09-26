"""Pinned CPU teacher targets for training; never part of streaming inference."""
from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import CROP_SAMPLES, WARMUP_SAMPLES
from .losses import _teacher_weight

SCORED_SAMPLES = CROP_SAMPLES - WARMUP_SAMPLES
CHECKPOINT_NAME = "7d865c68-3d5dd56b.th"
CHECKPOINT_SHA256 = "3d5dd56b5bc986f136dff98655ded22b2b033f465ccec7a28640a6b15fd71ed6"
CHECKPOINT_BYTES = 167918783
MODEL_SHA256 = "0f09cb3ecdb52332b2e38cff42c44a3078b1c926938486480429b568ca795da1"
BACKEND = Path(__file__).with_name("_teacher") / "demucs_sources.json"
PROVENANCE_KEY = "branch_memory_current_stage_teacher_supervision"
FLAGS = ("branch_memory_online_teacher_used", "branch_memory_teacher_generated_targets_in_current_stage")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def specification(coefficient):
    coefficient = _teacher_weight(coefficient)
    if not coefficient:
        return None
    return {"schema": "hs-tasnet-ordinary-teacher-v1", "coefficient": coefficient,
        "checkpoint": {"name": CHECKPOINT_NAME, "sha256": CHECKPOINT_SHA256, "bytes": CHECKPOINT_BYTES},
        "model_state_sha256": MODEL_SHA256, "backend_source_manifest_sha256": _sha(BACKEND),
        "sample_rate": 44100, "full_context_samples": CROP_SAMPLES,
        "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
        "input": "Exact final augmented ordinary mixture before student warmup removal",
        "normalization": "Official per-context mono mean/std plus 1e-8; undone after prediction",
        "random_shifts": 0, "split_inference": False, "teacher_device": "cpu",
        "teacher_precision": "fp32", "cpu_threads": 1,
        "output_policy": "Native Drums/Bass/Vocals; Other equals mixture minus DBV",
        "target_dtype": "normal detached FP32", "source_order": ["drums", "bass", "vocals", "other"],
        "auxiliary_teacher_weight": 0., "teacher_in_deployment_graph": False}


def supervision_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def attach(model, value):
    provenance = copy.deepcopy(getattr(model, "provenance", {}))
    if value is None:
        if PROVENANCE_KEY not in provenance and not any(provenance.get(key) for key in FLAGS):
            return
        provenance.pop(PROVENANCE_KEY, None)
        provenance.update({key: False for key in FLAGS})
    else:
        if value != specification(value.get("coefficient")):
            raise ValueError("Teacher specification differs from the supported training recipe")
        provenance.update({PROVENANCE_KEY: copy.deepcopy(value), **{key: True for key in FLAGS}})
    model.provenance = provenance


def validate_checkpoint(config, provenance):
    value = specification(config.get("teacher_coefficient", 0.))
    if value is None:
        if config.get("teacher_supervision") is not None or provenance.get(PROVENANCE_KEY) is not None or any(provenance.get(key) for key in FLAGS):
            raise ValueError("Checkpoint teacher provenance differs from its disabled training recipe")
    elif (config.get("teacher_supervision") != value or provenance.get(PROVENANCE_KEY) != value
          or not all(provenance.get(key) is True for key in FLAGS)):
        raise ValueError("Checkpoint teacher identity, coefficient or provenance differs")


def _model_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_model(checkpoint):
    try:
        import demucs
    except ImportError as error:
        raise ImportError("Install StemgenRT's teacher extra to enable teacher training") from error
    package = Path(demucs.__file__).resolve().parent
    source = json.loads(BACKEND.read_text())
    if (metadata.version("julius") != source["julius_version"]
            or any(_sha(package / name) != digest for name, digest in source["files"].items())):
        raise ValueError("Teacher backend differs from the pinned Demucs/Julius implementation")
    from demucs.hdemucs import HDemucs
    from demucs.apply import apply_model
    with torch.serialization.safe_globals([HDemucs]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["klass"] is not HDemucs:
        raise ValueError("Teacher checkpoint has a different model class")
    with torch.device("cpu"):
        model = HDemucs(*payload["args"], **payload["kwargs"])
    model.load_state_dict(payload["state"], strict=True)
    model.eval().requires_grad_(False)
    if (model.samplerate != 44100 or model.audio_channels != 2
            or model.sources != ["drums", "bass", "other", "vocals"]
            or any(value.device.type != "cpu" or value.dtype != torch.float32 for value in model.state_dict().values())
            or _model_sha(model) != MODEL_SHA256):
        raise ValueError("Teacher state, precision, sample rate or source order differs")
    return model, apply_model


def _predict(model, apply_model, mixture):
    initialized = torch.cuda.is_initialized()
    if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in model.state_dict().values()):
        raise ValueError("Teacher parameters must remain CPU FP32")
    wav = torch.from_numpy(np.array(mixture, dtype=np.float32, copy=True))
    ref = wav.mean(0)
    mean, scale = ref.mean(), ref.std() + 1e-8
    normalized = (wav - mean) / scale
    with torch.inference_mode():
        result = apply_model(model, normalized[None], shifts=0, split=False, device="cpu", num_workers=0)
        result = (result[0] * scale + mean)[[0, 1, 3, 2]]
    if (tuple(result.shape) != (4, 2, mixture.shape[-1]) or not bool(torch.isfinite(result).all())
            or result.device.type != "cpu" or result.dtype != torch.float32
            or torch.cuda.is_initialized() != initialized):
        raise ValueError("Teacher prediction changed its output or CPU execution contract")
    return result.numpy().copy()


class CPUTrainingTeacher:
    """Generate complete scored targets without changing the student's RNG streams."""
    def __init__(self, checkpoint, *, coefficient):
        self.coefficient = _teacher_weight(coefficient)
        self.specification = specification(self.coefficient)
        self.model = self.apply_model = None
        self.checkpoint = Path(checkpoint) if self.coefficient and checkpoint is not None else None
        if self.coefficient and (self.checkpoint is None or not self.checkpoint.is_file()
                or self.checkpoint.stat().st_size != CHECKPOINT_BYTES or _sha(self.checkpoint) != CHECKPOINT_SHA256):
            raise ValueError("Provide the pinned teacher checkpoint with its exact size and SHA-256")

    def _load(self):
        if self.model is not None:
            return
        if torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32:
            raise ValueError("Pinned teacher execution requires one CPU thread and default FP32")
        if _sha(self.checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Teacher checkpoint changed before loading")
        python_state, numpy_state = random.getstate(), np.random.get_state()
        initialized = torch.cuda.is_initialized()
        try:
            with torch.random.fork_rng(devices=[]):
                model, apply_model = _load_model(self.checkpoint)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        if torch.cuda.is_initialized() != initialized:
            raise ValueError("Teacher loading changed CUDA initialization")
        self.model, self.apply_model = model, apply_model

    def render(self, mixture):
        if not self.coefficient:
            return None
        if torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32:
            raise ValueError("Pinned teacher execution requires one CPU thread and default FP32")
        if (not isinstance(mixture, torch.Tensor) or mixture.ndim != 3
                or not 1 <= len(mixture) <= 16 or mixture.shape[1:] != (2, CROP_SAMPLES)
                or mixture.device.type != "cpu" or mixture.dtype != torch.float32
                or mixture.requires_grad or mixture.grad_fn is not None or torch.is_inference(mixture)
                or not bool(torch.isfinite(mixture).all())):
            raise ValueError("Require a fixed normal FP32 CPU batch of full six-second stereo crops")
        with torch.inference_mode(False), torch.no_grad(), torch.autocast("cpu", enabled=False):
            self._load()
            pieces = []
            for audio in mixture:
                native = _predict(self.model, self.apply_model, audio.numpy())
                full = np.asarray(native, dtype=np.float32).copy()
                full[3] = np.asarray(audio.numpy(), dtype=np.float32) - full[:3].sum(axis=0, dtype=np.float32)
                pieces.append(torch.from_numpy(full[..., WARMUP_SAMPLES:].copy()))
            targets = torch.stack(pieces)
        if (targets.shape != (len(mixture), 4, 2, SCORED_SAMPLES) or targets.dtype != torch.float32
                or targets.requires_grad or torch.is_inference(targets) or not bool(torch.isfinite(targets).all())):
            raise ValueError("Teacher returned invalid scored targets")
        return targets
