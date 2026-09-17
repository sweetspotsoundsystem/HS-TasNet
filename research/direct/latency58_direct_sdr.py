"""Direct, scale-dependent windowed SDR supervision for deployed stem audio."""
from __future__ import annotations

from dataclasses import dataclass

import torch

VERSION = "latency58-deployed-windowed-sdr-absence-half-anchor-five-percent-v1"
WINDOW = 44100
ACTIVITY_POWER = 1e-5
ABSENCE_WEIGHT = .5
ANCHOR_WEIGHT = .05


@dataclass(frozen=True)
class DirectSDRLoss:
    total: torch.Tensor
    negative_sdr_db: torch.Tensor
    per_stem_negative_sdr_db: torch.Tensor
    absence_db: torch.Tensor
    relative_l1_anchor: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor


def objective(raw, deployed, targets, mixture):
    """Use full one-second windows and the evaluation's fixed activity rule.

The main loss is negative scale-dependent SDR in dB, with equal stem weight.
Inactive windows receive a mixture-relative leakage penalty. A small relative
L1 term also anchors the pre-residual heads, including the learned Other head.
All coefficients affect training only. No teacher or fitted gain is used.
"""
    if (raw.ndim != 4 or raw.shape != deployed.shape or raw.shape != targets.shape
            or raw.shape[1:3] != (4, 2) or raw.shape[-1] < WINDOW
            or mixture.shape != (raw.shape[0], 2, raw.shape[-1])
            or any(v.dtype != torch.float32 or v.device != raw.device
                   for v in (raw, deployed, targets, mixture))
            or targets.requires_grad or mixture.requires_grad):
        raise ValueError("Require aligned FP32 raw/deployed/truth [B,4,2,T] and fixed physical mixture")
    windows = raw.shape[-1] // WINDOW
    with torch.autocast(raw.device.type, enabled=False):
        reference = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        estimate = deployed[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        raw_windows = raw[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        physical = mixture[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        signal = reference.square().sum(dim=(2, 4))
        error = (estimate - reference).square().sum(dim=(2, 4))
        active = signal / (2 * WINDOW) > ACTIVITY_POWER
        values = (10 * torch.log10((error + 1e-12) / (signal + 1e-12))).clamp(-60, 60)
        active_counts = active.sum(dim=(0, 2))
        per_stem = torch.where(active, values, 0).sum(dim=(0, 2)) / active_counts.clamp_min(1)
        primary = per_stem.sum() / (active_counts > 0).sum().clamp_min(1)
        mixture_power = physical.square().mean(dim=(1, 3))
        leakage = estimate.square().mean(dim=(2, 4))
        absent_counts = (~active).sum(dim=(0, 2))
        absence_values = 10 * torch.log10(1 + leakage / mixture_power[:, None].clamp_min(ACTIVITY_POWER))
        absence_per_stem = torch.where(~active, absence_values, 0).sum(dim=(0, 2)) / absent_counts.clamp_min(1)
        absence = absence_per_stem.sum() / (absent_counts > 0).sum().clamp_min(1)
        scale = torch.maximum(signal / (2 * WINDOW), .01 * mixture_power[:, None]).clamp_min(ACTIVITY_POWER).sqrt()
        anchor = ((raw_windows - reference).abs().mean(dim=(2, 4)) / scale).mean()
        total = primary + ABSENCE_WEIGHT * absence + ANCHOR_WEIGHT * anchor
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite direct SDR objective")
    return DirectSDRLoss(total, primary, per_stem, absence, anchor, active_counts, absent_counts)


def check():
    """Compare the primary loss to the independent unchanged NumPy scorer."""
    import numpy as np
    from research.metrics import MetricConfig, windowed_sdr
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260925)
    target = rng.normal(0, .05, (3, 4, 2, 2 * WINDOW + 128)).astype(np.float32)
    target[0, 1] = 0
    target[1, 2, :, :WINDOW] = 0
    estimates = target + rng.normal(0, .02, target.shape).astype(np.float32)
    truth = torch.from_numpy(target)
    raw = torch.from_numpy(estimates.copy()).requires_grad_()
    mixed = truth.sum(dim=1)
    terms = objective(raw, raw, truth, mixed)
    expected = []
    for stem in range(4):
        rows = [windowed_sdr(target[b, stem], estimates[b, stem], MetricConfig()) for b in range(3)]
        count = sum(row["active_windows"] for row in rows)
        expected.append(-sum(row["db"] * row["active_windows"] for row in rows if row["db"] is not None) / count)
    error = float(np.max(np.abs(np.asarray(expected) - terms.per_stem_negative_sdr_db.detach().numpy())))
    assert error < 2e-5, error
    gradient, = torch.autograd.grad(terms.negative_sdr_db, raw, retain_graph=True)
    assert bool(torch.isfinite(gradient).all()) and float((gradient * (raw.detach() - truth)).sum()) > 0
    assert torch.count_nonzero(gradient[0, 1]) == 0
    assert torch.count_nonzero(gradient[..., 2 * WINDOW:]) == 0
    terms.total.backward()
    assert bool(torch.isfinite(raw.grad).all()) and torch.count_nonzero(raw.grad[0, 1]) > 0
    with torch.no_grad():
        corrected = truth + .5 * (raw - truth)
        improved = objective(corrected, corrected, truth, mixed)
        assert improved.negative_sdr_db < terms.negative_sdr_db
        silence = torch.zeros_like(truth)
        zeros = objective(silence, silence, silence, torch.zeros_like(mixed))
        assert zeros.total.item() == 0 and not bool(zeros.active_window_counts.any())
    return {"status": "pass", "version": VERSION, "numpy_primary_max_abs_db": error,
            "gradient_points_toward_truth": True, "absence_has_finite_restoring_gradient": True,
            "trailing_partial_window_excluded_from_primary": True, "silence_loss_zero": True,
            "coefficient_search_performed": False}


if __name__ == "__main__":
    import json
    print(json.dumps(check()))
