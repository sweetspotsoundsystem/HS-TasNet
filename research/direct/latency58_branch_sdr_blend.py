"""Retain reconstruction supervision and add a fixed direct-SDR auxiliary loss."""
from dataclasses import dataclass

import torch

from research.direct.latency58_wave_spectral import objective as reconstruction
from research.direct.latency58_direct_sdr import objective as direct_sdr

VERSION = "latency58-wave-spectral-plus-fifth-direct-sdr-v1"
SDR_WEIGHT = .2


@dataclass(frozen=True)
class BlendedLoss:
    total: torch.Tensor
    waveform: torch.Tensor
    spectral: torch.Tensor
    raw_anchor: torch.Tensor
    negative_sdr_db: torch.Tensor
    per_stem_negative_sdr_db: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor
    reconstruction_loss: torch.Tensor
    direct_sdr_loss: torch.Tensor
    absence_db: torch.Tensor
    direct_raw_anchor: torch.Tensor


def objective(raw, deployed, targets, mixture):
    """Both summands carry gradients; the coefficient is fixed before training.

    Reconstruction is wave L1 + .25 complex-STFT + .25 raw-head L1.
    The direct term is negative scale-dependent SDR + .5 absence + .05
    window-normalized raw-head L1. Inference and evaluation are unchanged.
    """
    base = reconstruction(raw, deployed, targets, mixture)
    direct = direct_sdr(raw, deployed, targets, mixture)
    total = base.total + SDR_WEIGHT * direct.total
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite blended reconstruction/SDR loss")
    return BlendedLoss(total, base.waveform, base.spectral, base.raw_anchor,
                       direct.negative_sdr_db, direct.per_stem_negative_sdr_db,
                       direct.active_window_counts, direct.absent_window_counts,
                       base.total, direct.total, direct.absence_db,
                       direct.relative_l1_anchor)


def check():
    """Check metric agreement and that SDR supplies a restoring gradient."""
    from research.direct.latency58_direct_sdr import check as check_metric
    metric = check_metric()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261022)
    truth = .03 * torch.randn(2, 4, 2, 44160, generator=generator)
    truth[0, 1] = 0
    mixture = truth.sum(1)
    noise = .008 * torch.randn(truth.shape, generator=generator)
    estimate = (truth + noise).requires_grad_()
    blend = objective(estimate, estimate, truth, mixture)
    base = reconstruction(estimate, estimate, truth, mixture)
    combined_grad, = torch.autograd.grad(blend.total, estimate, retain_graph=True)
    base_grad, = torch.autograd.grad(base.total, estimate)
    sdr_grad, = torch.autograd.grad(blend.direct_sdr_loss, estimate)
    assert torch.isfinite(combined_grad).all() and torch.isfinite(sdr_grad).all()
    assert torch.allclose(combined_grad - base_grad, SDR_WEIGHT * sdr_grad, atol=2e-10, rtol=2e-4)
    assert float((sdr_grad * noise).sum()) > 0 and torch.count_nonzero(sdr_grad[0, 1]) > 0
    assert float((combined_grad * noise).sum()) > 0
    with torch.no_grad():
        improved = objective(truth + .5 * noise, truth + .5 * noise, truth, mixture)
        assert improved.total < blend.total
        assert improved.negative_sdr_db < blend.negative_sdr_db
        assert improved.absence_db < blend.absence_db
        zeros = torch.zeros_like(truth)
        assert objective(zeros, zeros, zeros, zeros.sum(1)).total.item() == 0
    assert not torch.cuda.is_initialized()
    return {"status": "pass", "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
            "numpy_primary_metric_check": metric,
            "direct_sdr_has_independent_restoring_gradient": True,
            "absent_stem_has_restoring_gradient": True,
            "combined_gradient_contains_direct_sdr_contribution": True,
            "reduced_error_improves_loss_primary_and_absence": True,
            "all_silence_loss_zero": True, "coefficient_search_performed": False,
            "validation_audio_used": False, "inference_architecture_changed": False}


if __name__ == "__main__":
    import json
    print(json.dumps(check()))
