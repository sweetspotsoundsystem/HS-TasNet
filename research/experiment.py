"""Editable HS-TasNet experiment definition.

Autoresearch candidates may change this file (and the model package) to alter
architecture, optimization, loss, precision, crop/batch sizing, clipping, and
training logging.  The frozen runner owns data access, timing, seeding, and the
single final-raw artifact rule.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio

from hs_tasnet import HSTasNet


SAMPLE_RATE = 44_100
STEM_SUBSET_AUGMENT_PROBABILITY = 0.25
VOCAL_DERANGEMENT_PROBABILITY = 0.125
VOCAL_SOURCE_INDEX = 2
OFF_TARGET_SOURCE_INDICES = (0, 1, 3)
DERANGED_PROJECTION_LOSS_WEIGHT = 0.01
DERANGED_PROJECTION_MAX_L1_FRACTION = 0.005
DERANGED_PROJECTION_RAMP_START_STEP = 750
DERANGED_PROJECTION_RAMP_END_STEP = 1_000
ACTIVITY_POWER_EPSILON = 1e-5
LEARNING_RATE_DECAY_START_STEP = 2_000
LEARNING_RATE_DECAY_END_STEP = 2_400
FINAL_SOURCE_CALIBRATION = (1.0, 1.0, 0.80, 1.12)


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    crop_seconds: float
    precision: str
    learning_rate: float
    grad_clip_norm: float
    log_every: int


def get_config(*, smoke: bool) -> ExperimentConfig:
    if smoke:
        return ExperimentConfig(
            batch_size=1,
            crop_seconds=2048 / SAMPLE_RATE,
            precision="fp32",
            learning_rate=3e-4,
            grad_clip_norm=5.0,
            log_every=1,
        )

    return ExperimentConfig(
        batch_size=16,
        crop_seconds=2.0,
        precision="bf16",
        learning_rate=3e-4,
        grad_clip_norm=5.0,
        log_every=25,
    )


def build_model(*, config: ExperimentConfig, smoke: bool) -> HSTasNet:
    del config
    if smoke:
        return HSTasNet(
            dim=16,
            small=True,
            stereo=True,
            num_basis=32,
            use_gru=True,
            use_branch_rnns=False,
            residual_source_softmax=True,
        )
    return HSTasNet(
        stereo=True,
        use_gru=True,
        use_branch_rnns=False,
        residual_source_softmax=True,
    )


def build_optimizer(
    *, model: torch.nn.Module, config: ExperimentConfig
) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer._autoresearch_base_lrs = tuple(  # type: ignore[attr-defined]
        group["lr"] for group in optimizer.param_groups
    )
    optimizer._autoresearch_step = 0  # type: ignore[attr-defined]
    return optimizer


def _apply_learning_rate_schedule(
    optimizer: torch.optim.Optimizer,
) -> None:
    step = int(optimizer._autoresearch_step)  # type: ignore[attr-defined]
    base_lrs = optimizer._autoresearch_base_lrs  # type: ignore[attr-defined]
    if step <= LEARNING_RATE_DECAY_START_STEP:
        scale = 1.0
    elif step >= LEARNING_RATE_DECAY_END_STEP:
        scale = 0.0
    else:
        progress = (
            (step - LEARNING_RATE_DECAY_START_STEP)
            / (LEARNING_RATE_DECAY_END_STEP - LEARNING_RATE_DECAY_START_STEP)
        )
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = float(base_lr) * scale


def _apply_final_source_calibration(model: torch.nn.Module) -> None:
    """Finalize source gains and the single-frame deployment decoder."""

    if getattr(model, "_autoresearch_source_calibrated", False):
        raise RuntimeError("final source calibration was applied more than once")
    if not isinstance(model, HSTasNet) or model.num_sources != 4:
        raise TypeError("final source calibration requires canonical four-stem HSTasNet")

    scales = torch.tensor(
        FINAL_SOURCE_CALIBRATION,
        device=model.device,
        dtype=torch.float32,
    )
    model.set_output_source_gains(scales)
    model.bake_decoder_hann_window_()
    model._autoresearch_source_calibrated = True


class DeterministicCropSampler:
    """Baseline uniform track/crop sampler; editable by candidate research."""

    def __init__(
        self,
        *,
        tracks: Sequence[Any],
        crop_samples: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.crop_samples = crop_samples
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.usable = tuple(track for track in tracks if track.frames >= crop_samples)
        if not self.usable:
            raise ValueError(f"no training track is at least {crop_samples} samples long")

    @staticmethod
    def _load(path: Path, offset: int, frames: int) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            audio, sample_rate = torchaudio.load(
                str(path),
                frame_offset=offset,
                num_frames=frames,
                normalize=True,
                channels_first=True,
                backend="soundfile",
            )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample rate changed while reading {path}")
        if tuple(audio.shape) != (2, frames):
            raise ValueError(f"short or malformed crop from {path}: {tuple(audio.shape)}")
        return audio

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        mixtures: list[torch.Tensor] = []
        all_stems: list[torch.Tensor] = []
        for _ in range(self.batch_size):
            track = self.usable[self.rng.randrange(len(self.usable))]
            offset = self.rng.randrange(track.frames - self.crop_samples + 1)
            mixtures.append(self._load(track.mixture, offset, self.crop_samples))
            all_stems.append(
                torch.stack(
                    [
                        self._load(path, offset, self.crop_samples)
                        for path in track.stems
                    ]
                )
            )
        return torch.stack(mixtures), torch.stack(all_stems)


def build_sampler(
    *,
    tracks: Sequence[Any],
    crop_samples: int,
    batch_size: int,
    seed: int,
) -> DeterministicCropSampler:
    return DeterministicCropSampler(
        tracks=tracks,
        crop_samples=crop_samples,
        batch_size=batch_size,
        seed=seed,
    )


def _augment_training_distribution(
    *, mixture: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix disjoint stem-subset and all-active vocal-derangement examples."""

    batch_size, source_count = targets.shape[:2]
    if source_count != 4:
        raise ValueError("training augmentation requires four ordered stems")

    augmentation_draw = torch.rand(batch_size, device=targets.device)
    subset_augment = augmentation_draw < STEM_SUBSET_AUGMENT_PROBABILITY
    vocal_derangement = (
        augmentation_draw >= STEM_SUBSET_AUGMENT_PROBABILITY
    ) & (
        augmentation_draw
        < STEM_SUBSET_AUGMENT_PROBABILITY + VOCAL_DERANGEMENT_PROBABILITY
    )
    if batch_size < 2:
        vocal_derangement = torch.zeros_like(vocal_derangement)
    keep_count = torch.randint(
        1,
        source_count,
        (batch_size,),
        device=targets.device,
    )
    random_order = torch.rand(
        batch_size, source_count, device=targets.device
    ).argsort(dim=1)
    ranks = random_order.argsort(dim=1)
    subset_mask = ranks < keep_count[:, None]
    keep_mask = torch.where(
        subset_augment[:, None],
        subset_mask,
        torch.ones_like(subset_mask),
    )

    augmented_targets = targets * keep_mask[:, :, None, None]
    donor_vocals = targets[:, VOCAL_SOURCE_INDEX].roll(shifts=1, dims=0)
    remixed_vocals = torch.where(
        vocal_derangement[:, None, None],
        donor_vocals,
        augmented_targets[:, VOCAL_SOURCE_INDEX],
    )
    augmented_targets[:, VOCAL_SOURCE_INDEX].copy_(remixed_vocals)

    augmented_mixture = augmented_targets.sum(dim=1)
    altered = subset_augment | vocal_derangement
    augmented_mixture = torch.where(
        altered[:, None, None], augmented_mixture, mixture
    )
    return augmented_mixture, augmented_targets, vocal_derangement


def _deranged_projection_loss_weight(step: int) -> float:
    if step < DERANGED_PROJECTION_RAMP_START_STEP:
        return 0.0
    if step >= LEARNING_RATE_DECAY_END_STEP:
        return 0.0
    if step >= DERANGED_PROJECTION_RAMP_END_STEP:
        return DERANGED_PROJECTION_LOSS_WEIGHT

    progress = (
        (step - DERANGED_PROJECTION_RAMP_START_STEP)
        / (
            DERANGED_PROJECTION_RAMP_END_STEP
            - DERANGED_PROJECTION_RAMP_START_STEP
        )
    )
    return DERANGED_PROJECTION_LOSS_WEIGHT * 0.5 * (
        1.0 - math.cos(math.pi * progress)
    )


def _deranged_vocal_projection_loss(
    *,
    estimates: torch.Tensor,
    targets: torch.Tensor,
    vocal_derangement: torch.Tensor,
) -> torch.Tensor:
    """Discourage accompaniment projection only when vocal identity was shuffled."""

    sample_count = min(SAMPLE_RATE, estimates.shape[-1])
    vocal = estimates[:, VOCAL_SOURCE_INDEX, :, -sample_count:].float()
    desired_vocal = targets[
        :, VOCAL_SOURCE_INDEX, :, -sample_count:
    ].float()
    references = targets[
        :, OFF_TARGET_SOURCE_INDICES, :, -sample_count:
    ].float()

    desired_vocal_energy = desired_vocal.square().sum(dim=(-2, -1))
    reference_energy = references.square().sum(dim=(-2, -1))
    vocal_energy = vocal.square().sum(dim=(-2, -1))
    correlations = (references * vocal[:, None]).sum(dim=(-2, -1))
    squared_cosine = correlations.square() / (
        reference_energy * vocal_energy[:, None]
    ).clamp_min(1e-12)

    channel_sample_count = vocal.shape[-2] * sample_count
    active_references = (
        reference_energy / float(channel_sample_count)
        > ACTIVITY_POWER_EPSILON
    )
    active_desired_vocal = (
        desired_vocal_energy / float(channel_sample_count)
        > ACTIVITY_POWER_EPSILON
    )
    weights = (
        vocal_derangement[:, None]
        & active_desired_vocal[:, None]
        & active_references
    ).to(dtype=squared_cosine.dtype)
    return (squared_cosine * weights).sum() / weights.sum().clamp_min(1.0)


def train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    mixture: torch.Tensor,
    targets: torch.Tensor,
    config: ExperimentConfig,
    amp_dtype: torch.dtype | None,
) -> float:
    _apply_learning_rate_schedule(optimizer)
    optimizer.zero_grad(set_to_none=True)
    mixture, targets, vocal_derangement = _augment_training_distribution(
        mixture=mixture,
        targets=targets,
    )
    with torch.autocast(
        device_type=mixture.device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        waveform_loss, estimates = model(
            mixture,
            targets=targets,
            return_targets_with_loss=True,
        )

    step = int(optimizer._autoresearch_step)  # type: ignore[attr-defined]
    loss_weight = _deranged_projection_loss_weight(step)
    waveform_loss = waveform_loss.float()
    if loss_weight > 0.0:
        projection_loss = _deranged_vocal_projection_loss(
            estimates=estimates,
            targets=targets[..., : estimates.shape[-1]],
            vocal_derangement=vocal_derangement,
        )
        contribution_cap = (
            DERANGED_PROJECTION_MAX_L1_FRACTION
            * waveform_loss.detach()
            / projection_loss.detach().clamp_min(1e-8)
        )
        effective_weight = torch.minimum(
            projection_loss.new_tensor(loss_weight), contribution_cap
        )
        loss = waveform_loss + effective_weight * projection_loss
    else:
        loss = waveform_loss

    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss: {loss.item()}")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.grad_clip_norm,
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    if optimizer._autoresearch_step == LEARNING_RATE_DECAY_END_STEP:  # type: ignore[attr-defined]
        _apply_final_source_calibration(model)
    optimizer._autoresearch_step += 1  # type: ignore[attr-defined]
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite reported training loss: {value}")
    return value


def log_step(
    *, config: ExperimentConfig, step: int, loss: float, training_seconds: float
) -> None:
    if step % config.log_every == 0:
        print(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "training_seconds": training_seconds,
                }
            ),
            flush=True,
        )


def describe(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "name": "single_frame_decoder_matrix_hs_tasnet",
        "config": asdict(config),
        "model": (
            "HSTasNet(stereo=True, use_gru=True, use_branch_rnns=False, "
            "residual_source_softmax=True)"
        ),
        "optimizer": "torch.optim.Adam",
        "learning_rate_schedule": (
            "constant through step 2000 then cosine decay to zero at step 2400"
        ),
        "loss": (
            "waveform L1 plus a capped deranged-row vocal projection penalty"
        ),
        "deranged_vocal_projection_loss": {
            "source_order": ["drums", "bass", "vocals", "other"],
            "references": ["drums", "bass", "other"],
            "metric": "mean squared cosine projection over active references",
            "formula": "dot(vocal reference)^2 / (vocal_energy reference_energy)",
            "precision": "FP32 outside autocast",
            "examples": "vocal-derangement rows only",
            "activity": "desired vocal and off-target reference above power floor",
            "maximum_weight": DERANGED_PROJECTION_LOSS_WEIGHT,
            "maximum_l1_fraction": DERANGED_PROJECTION_MAX_L1_FRACTION,
            "ramp_steps": [
                DERANGED_PROJECTION_RAMP_START_STEP,
                DERANGED_PROJECTION_RAMP_END_STEP,
            ],
            "stop_step": LEARNING_RATE_DECAY_END_STEP,
            "window_samples": SAMPLE_RATE,
        },
        "final_source_calibration": {
            "step": LEARNING_RATE_DECAY_END_STEP,
            "drums": FINAL_SOURCE_CALIBRATION[0],
            "bass": FINAL_SOURCE_CALIBRATION[1],
            "vocals": FINAL_SOURCE_CALIBRATION[2],
            "other": FINAL_SOURCE_CALIBRATION[3],
            "mechanism": "saved output-source scale after branch summation",
        },
        "waveform_decoder": {
            "finalize_step": LEARNING_RATE_DECAY_END_STEP,
            "baked_weight_shape": [1500, 2, 1024],
            "single_frame_matrix_shape": [1500, 2048],
            "single_frame_path": "bias-free FP32 eager linear",
            "fallback": "ConvTranspose1d for training and noncanonical inputs",
            "parameter_count_change": 0,
        },
        "mask_competition": {
            "sources": ["drums", "bass", "vocals", "other"],
            "branches": ["spectrogram", "waveform"],
            "formula": "0.5 * raw + 0.5 * (4 * source softmax(raw))",
            "raw_signed_path_retained": True,
        },
        "training_augmentation": {
            "mutually_exclusive_probabilities": {
                "random_nonempty_proper_stem_subset": (
                    STEM_SUBSET_AUGMENT_PROBABILITY
                ),
                "all_active_vocal_derangement": VOCAL_DERANGEMENT_PROBABILITY,
                "native_mixture": (
                    1.0
                    - STEM_SUBSET_AUGMENT_PROBABILITY
                    - VOCAL_DERANGEMENT_PROBABILITY
                ),
            },
            "stem_subset": "uniformly retain one two or three stems",
            "vocal_derangement": (
                "cyclically roll only vocals across the batch and rebuild mixture"
            ),
        },
        "artifact_selection": "not editable; frozen runner saves final raw weights",
    }
