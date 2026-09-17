"""Relative waveform and multi-resolution complex spectral reconstruction loss."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from research.direct.latency58_direct_sdr import objective as sdr_diagnostic

VERSION = "latency58-relative-wave-l1-plus-quarter-complex-stft-and-raw-l1-v1"
AUGMENTATION = "half_original_half_same_song_source_gain_3db_and_shared_polarity"
FFT_SIZES = (512, 1024, 2048)


@dataclass(frozen=True)
class ReconstructionLoss:
    total: torch.Tensor
    waveform: torch.Tensor
    spectral: torch.Tensor
    raw_anchor: torch.Tensor
    negative_sdr_db: torch.Tensor
    per_stem_negative_sdr_db: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor


def augment(mixture, targets, *, seed, first_sample_index):
    """Address augmentation by crop index; keep half the original recordings."""
    if mixture.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError("Address augmentation on CPU before transferring the batch")
    if mixture.shape != (targets.shape[0], 2, targets.shape[-1]) or targets.shape[1:3] != (4, 2):
        raise ValueError("Require aligned mixture and four stereo stems")
    rows, choices = [], []
    for index in range(first_sample_index, first_sample_index + targets.shape[0]):
        generator = torch.Generator().manual_seed(seed * 1_000_003 + index)
        values = torch.rand(6, generator=generator).tolist()
        changed = values[0] < .5
        sign = -1. if values[5] < .5 else 1.
        # Scalar FP64 exponentiation and one FP32 rounding avoid the CPU pow
        # kernel's vector-width-dependent final bit when batch size changes.
        rows.append([sign * math.pow(10., (6 * value - 3) / 20) if changed else 1.
                     for value in values[1:5]])
        choices.append(changed)
    changed = torch.tensor(choices, dtype=torch.bool)
    factors = torch.tensor(rows, dtype=torch.float32)
    transformed = targets * factors[:, :, None, None]
    rendered = torch.where(changed[:, None, None], transformed.sum(1), mixture)
    return rendered, transformed, changed, factors


def objective(raw, deployed, targets, mixture):
    if raw.shape != deployed.shape or raw.shape != targets.shape or raw.ndim != 4 or raw.shape[1:3] != (4, 2):
        raise ValueError("Require aligned four-stem stereo estimates and references")
    if mixture.shape != (raw.shape[0], 2, raw.shape[-1]) or raw.shape[-1] < 44100:
        raise ValueError("Require an aligned physical mixture and at least one scored second")
    if any(t.dtype != torch.float32 or t.device != raw.device for t in (raw, deployed, targets, mixture)):
        raise ValueError("Use aligned FP32 reconstruction tensors")
    if targets.requires_grad or mixture.requires_grad:
        raise ValueError("Training references must remain fixed")
    with torch.autocast(raw.device.type, enabled=False):
        reference_rms = targets.square().mean((2, 3)).sqrt()
        mixture_rms = mixture.square().mean((1, 2)).sqrt()
        scale = torch.maximum(reference_rms, .1 * mixture_rms[:, None]).clamp_min(1e-3)
        waveform = ((deployed - targets).abs().mean((2, 3)) / scale).mean()
        raw_anchor = ((raw - targets).abs().mean((2, 3)) / scale).mean()
        components = []
        for size in FFT_SIZES:
            window = torch.hann_window(size, device=raw.device, dtype=torch.float32)
            def spectrum(value):
                return torch.stft(value.reshape(-1, value.shape[-1]), n_fft=size, hop_length=size // 4,
                    window=window, center=False, return_complex=True).unflatten(0, value.shape[:-1])
            estimate = spectrum(deployed)
            reference = spectrum(targets)
            physical = spectrum(mixture)
            # Complex distance retains phase sensitivity; normalization averages
            # channels, bins and frames independently for each sample and stem.
            denominator = torch.maximum(reference.abs().mean((2, 3, 4)),
                .1 * physical.abs().mean((1, 2, 3))[:, None]).clamp_min(1e-3)
            components.append(((estimate - reference).abs().mean((2, 3, 4)) / denominator).mean())
        spectral = torch.stack(components).mean()
        total = waveform + .25 * spectral + .25 * raw_anchor
        with torch.no_grad():
            metric = sdr_diagnostic(raw, deployed, targets, mixture)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite waveform/spectral loss")
    return ReconstructionLoss(total, waveform, spectral, raw_anchor, metric.negative_sdr_db,
        metric.per_stem_negative_sdr_db, metric.active_window_counts, metric.absent_window_counts)


def check():
    """Check restoring gradients, exact references, silence, and addressed views."""
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261004)
    truth = torch.randn((2, 4, 2, 44100 + 128), generator=generator) * .03
    truth[0, 1] = 0
    mixture = truth.sum(1)
    noise = torch.randn(truth.shape, generator=generator) * .008
    estimate = (truth + noise).requires_grad_()
    terms = objective(estimate, estimate, truth, mixture)
    terms.total.backward()
    assert bool(torch.isfinite(estimate.grad).all()) and float((estimate.grad * noise).sum()) > 0
    assert torch.count_nonzero(estimate.grad[0, 1]) > 0
    exact = objective(truth, truth, truth, mixture)
    assert exact.total.item() == 0
    improved = objective(truth + .5 * noise, truth + .5 * noise, truth, mixture)
    assert improved.total < terms.total
    silence = torch.zeros_like(truth)
    assert objective(silence, silence, silence, torch.zeros_like(mixture)).total.item() == 0
    # Augmentation depends on absolute indices, so CPU batching cannot change it.
    expanded = truth.repeat(4, 1, 1, 1)
    physical = expanded.sum(1) + .0001
    together = augment(physical, expanded, seed=20261004, first_sample_index=1_700_000)
    pieces = [augment(x, y, seed=20261004, first_sample_index=1_700_000 + 2 * i)
              for i, (x, y) in enumerate(zip(physical.split(2), expanded.split(2), strict=True))]
    assert all(torch.equal(value, torch.cat([row[i] for row in pieces])) for i, value in enumerate(together))
    audio, stems, changed, factors = together
    assert changed.any() and (~changed).any()
    assert torch.equal(audio[~changed], physical[~changed]) and torch.equal(stems[~changed], expanded[~changed])
    assert torch.equal(audio[changed], stems[changed].sum(1))
    assert bool((factors[changed].abs() >= 10 ** (-3 / 20)).all())
    assert bool((factors[changed].abs() <= 10 ** (3 / 20)).all())
    assert not torch.cuda.is_initialized()
    return {"status": "pass", "objective_version": VERSION, "augmentation": AUGMENTATION,
            "restoring_gradient": True, "absent_stem_has_gradient": True, "exact_reference_loss_zero": True,
            "all_silence_loss_zero": True, "augmentation_batch_partition_exact": True,
            "ordinary_mixtures_exact": True, "augmented_mixture_equals_augmented_stem_sum": True,
            "source_gain_bounds_db": [-3, 3], "inference_architecture_changed": False}


if __name__ == "__main__":
    import json
    print(json.dumps(check()))
