"""Bounded fine-tuning with detached history, accumulation and resumable Adam state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .streaming_checkpoint import (
    load_streaming_checkpoint, read_streaming_checkpoint, model_from_payload,
    save_streaming_checkpoint, state_sha256,
)
from .streaming_data import load_manifest, require_disjoint, CounterAddressedCropDataset, AbsoluteIndexSampler, worker_init
from .streaming_model import StreamingHSTasNet, require
from .streaming_training import augment_training_batch, render_scored_context, streaming_objective


@dataclass(frozen=True)
class StreamingTrainConfig:
    steps: int = 250
    batch_size: int = 16
    microbatch_size: int = 4
    crop_samples: int = 176128
    warmup_samples: int = 88064
    lr: float = 1e-5
    min_lr: float = 1e-6
    warmup: int = 25
    checkpoint_every: int = 250
    seed: int = 20260921
    data_seed: int = 60
    data_start: int = 0
    vocal_active_probability: float = .85
    controlled_views: bool = True
    teacher_weight: float = 0.
    deployed_truth_weight: float = .5
    precision: str = "bf16"
    workers: int = 2

    def validate(self):
        for name in ("steps", "batch_size", "microbatch_size", "crop_samples", "warmup_samples", "checkpoint_every"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, name + " must be a positive integer")
        for name in ("seed", "data_seed", "data_start", "warmup", "workers"):
            require(type(getattr(self, name)) is int and getattr(self, name) >= 0, name + " must be a nonnegative integer")
        require(self.microbatch_size == 4 and self.batch_size % 4 == 0, "Use B=4 microbatches and a divisible effective batch")
        require(self.crop_samples > self.warmup_samples and self.warmup_samples % 128 == 0
                and self.crop_samples % 128 == 0 and self.warmup < self.steps, "Invalid context or learning-rate warmup")
        for name in ("lr", "min_lr", "teacher_weight", "deployed_truth_weight", "vocal_active_probability"):
            value = getattr(self, name)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name + " must be finite and nonnegative")
        require(self.lr > 0 and self.min_lr <= self.lr and self.vocal_active_probability <= 1
                and self.precision in ("fp32", "bf16") and type(self.controlled_views) is bool, "Invalid training policy")
        return self


def learning_rate(step, config):
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    progress = (step - config.warmup) / max(1, config.steps - config.warmup - 1)
    return config.min_lr + (config.lr - config.min_lr) * .5 * (1 + math.cos(math.pi * progress))


def configure_determinism(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def capture_rng(device):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch_cpu"])
    require((state["torch_cuda"] is not None) == (device.type == "cuda"), "Resume device family differs")
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_streaming(config, manifest, output, *, checkpoint=None, resume=None, teacher=None,
                    validation_manifest=None, device="cuda", stop_after=None):
    """Run a finite training stage in a new output directory.

    ``teacher`` is an optional frozen ``nn.Module`` that maps each complete
    [B,2,T] crop to physically aligned [B,4,2,T] native-level outputs. It is
    required when teacher_weight is positive. Resume preserves the optimizer,
    random draws and absolute crop address; use a new output directory.
    """
    config.validate()
    device = torch.device(device)
    require(device.type in ("cpu", "cuda") and not (checkpoint and resume), "Choose a CPU/CUDA device and one initialization")
    if device.type == "cuda":
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
        require(torch.cuda.is_available(), "CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        require(config.precision != "bf16" or torch.cuda.is_bf16_supported(),
                "Requested CUDA precision is unavailable")
        torch.cuda.set_per_process_memory_fraction(.75, device)
    else:
        require(config.precision == "fp32", "CPU training requires precision=fp32")
    require((teacher is not None) == (config.teacher_weight > 0), "Supply a teacher exactly when its loss weight is positive")
    stop = config.steps if stop_after is None else stop_after
    require(type(stop) is int and 0 < stop <= config.steps, "Stop must lie within the configured horizon")
    tracks, root_weights, manifest_sha = load_manifest(manifest, crop_samples=config.crop_samples)
    if validation_manifest is not None:
        validation_tracks, _, _ = load_manifest(validation_manifest, crop_samples=1)
        require_disjoint(tracks, validation_tracks)
    configure_determinism(config.seed)
    payload = read_streaming_checkpoint(resume) if resume else None
    model = (model_from_payload(payload) if payload else load_streaming_checkpoint(checkpoint)
             if checkpoint else StreamingHSTasNet())
    model.to(device).train().requires_grad_(True)
    model.training_precision = config.precision
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, foreach=False)
    teacher_identity = None
    if teacher is not None:
        require(isinstance(teacher, torch.nn.Module), "Teacher must be an nn.Module")
        teacher.to(device).eval().requires_grad_(False)
        for parameter in teacher.parameters():
            parameter.grad = None
        require(set(map(id, model.parameters())).isdisjoint(map(id, teacher.parameters())), "Teacher shares trainable parameters")
        teacher_identity = {"class": type(teacher).__module__ + "." + type(teacher).__qualname__,
                            "model_state_sha256": state_sha256(teacher.state_dict())}
    step = 0
    initial = state_sha256(model.state_dict())
    if payload:
        saved = payload.get("training", {})
        require(saved.get("schema") == "hs-tasnet-streaming-adam-resume-v1"
                and saved["config"] == asdict(config) and saved["manifest_sha256"] == manifest_sha
                and saved["teacher"] == teacher_identity and saved["device_type"] == device.type,
                "Resume recipe, dataset, teacher or device differs")
        step = saved["step"]
        require(type(step) is int and 0 < step < stop and saved["next_sample_index"]
                == config.data_start + step * config.batch_size, "Resume endpoint differs")
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], device)
        initial = saved["initial_model_state_sha256"]
    start = step
    dataset = CounterAddressedCropDataset(tracks, root_weights=root_weights, seed=config.data_seed,
        crop_samples=config.crop_samples, vocal_active_probability=config.vocal_active_probability,
        final_sample_index=config.data_start + stop * config.batch_size)
    loader_options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if config.workers else {}
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.workers,
        sampler=AbsoluteIndexSampler(config.data_start + step * config.batch_size, config.data_start + stop * config.batch_size),
        pin_memory=device.type == "cuda", worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(config.seed + 1), **loader_options)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    fixed = {name: value.clone() for name, value in model.named_buffers()}
    began = time.monotonic()
    accumulation = config.batch_size // config.microbatch_size
    try:
        with (output / "metrics.jsonl").open("x") as journal:
            for mixture_cpu, targets_cpu in loader:
                require(mixture_cpu.shape[0] == config.batch_size, "Incomplete effective training batch")
                optimizer.param_groups[0]["lr"] = learning_rate(step, config)
                optimizer.zero_grad(set_to_none=True)
                first = config.data_start + step * config.batch_size
                micros = []
                for offset in range(0, config.batch_size, config.microbatch_size):
                    mixture = mixture_cpu[offset:offset + config.microbatch_size].to(device)
                    targets = targets_cpu[offset:offset + config.microbatch_size].to(device)
                    pristine_digest = tensor_digest((("mixture", mixture), ("targets", targets)))
                    batch = augment_training_batch(mixture, targets, first_sample_index=first + offset,
                                                   enabled=config.controlled_views)
                    teacher_targets = None
                    if teacher is not None:
                        with torch.no_grad(), torch.autocast(device.type, enabled=False):
                            teacher_output = teacher(batch.mixture)
                        require(isinstance(teacher_output, torch.Tensor) and teacher_output.shape == batch.targets.shape,
                                "Teacher must return all four physically aligned source tensors")
                        teacher_targets = teacher_output[..., config.warmup_samples:].detach().clone()
                        del teacher_output
                    rendered = render_scored_context(model, batch.mixture, warmup_samples=config.warmup_samples, carry_state=True)
                    terms = streaming_objective(rendered.raw, rendered.deployed, batch.targets[..., config.warmup_samples:],
                        batch.vocal_derangement, view_codes=batch.view_codes, teacher_targets=teacher_targets,
                        teacher_weight=config.teacher_weight, deployed_truth_weight=config.deployed_truth_weight)
                    (terms.total / accumulation).backward()
                    micros.append({"first_sample_index": first + offset, "view_codes": list(batch.view_codes),
                        **{key: float(getattr(terms, key).detach()) for key in StreamingLossKeys},
                        "pristine_batch_sha256": pristine_digest,
                        "original_augmentation_sha256": tensor_digest(zip(("mixture", "targets", "deranged"), batch.original_augmentation)),
                        "augmented_batch_sha256": tensor_digest((("mixture", batch.mixture), ("targets", batch.targets), ("deranged", batch.vocal_derangement)))})
                    del mixture, targets, batch, rendered, terms, teacher_targets
                require(all(parameter.grad is not None for parameter in model.parameters()), "Missing parameter gradients")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=False)
                optimizer.step()
                step += 1
                row = {"step": step, "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(norm),
                       "first_sample_index": first, "next_sample_index": first + config.batch_size,
                       "microbatches": micros, "elapsed_seconds": time.monotonic() - began,
                       **{key: sum(micro[key] for micro in micros) / accumulation for key in StreamingLossKeys}}
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({key: row[key] for key in ("step", "lr", "total", "grad_norm")}), flush=True)
                if step % config.checkpoint_every == 0 or step == stop:
                    require(all(torch.equal(value, fixed[name]) for name, value in model.named_buffers()), "Fixed buffers changed")
                    training = {"schema": "hs-tasnet-streaming-adam-resume-v1", "config": asdict(config), "step": step,
                                "next_sample_index": config.data_start + step * config.batch_size,
                                "manifest_sha256": manifest_sha, "initial_model_state_sha256": initial,
                                "teacher": teacher_identity, "device_type": device.type,
                                "optimizer": optimizer.state_dict(), "rng": capture_rng(device)}
                    save_streaming_checkpoint(model, output / f"step-{step:06d}.pt", training=training)
        require(step == stop, "Training ended before the requested step")
        _, _, final_manifest_sha = load_manifest(manifest, crop_samples=config.crop_samples)
        require(final_manifest_sha == manifest_sha, "Training manifest changed")
        if teacher is not None:
            require(state_sha256(teacher.state_dict()) == teacher_identity["model_state_sha256"]
                    and all(parameter.grad is None for parameter in teacher.parameters()), "Teacher changed")
        result = {"status": "pass", "start_step": start, "step": step, "configured_steps": config.steps,
                  "model_state_sha256": state_sha256(model.state_dict()), "manifest_sha256": manifest_sha,
                  "quality_evaluated": False, "elapsed_seconds": time.monotonic() - began}
    except BaseException as error:
        (output / "result.json").write_text(json.dumps({"status": "failed", "step": step, "error": repr(error)}) + "\n")
        raise
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


StreamingLossKeys = ("total", "raw_l1", "projection", "projection_contribution", "teacher_l1", "controlled_deployed_l1")
