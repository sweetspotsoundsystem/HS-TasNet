"""A recorded-training-only source calibration with no additional audio state."""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from research.direct.latency58_residual_model import Latency58ResidualModel
from research.direct.run_latency58_quality import require, sha
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-recorded-training-source-mixer-v1"
BUFFER = "trained_source_mixer"
WINDOW = 44100


def mix_sources(estimates, mixture, coefficients):
    retained = torch.einsum("os,bsct->boct", coefficients, estimates)
    other = mixture - ((retained[:, 0] + retained[:, 1]) + retained[:, 2])
    return torch.cat((retained, other.unsqueeze(1)), 1)


class Latency58SourceMixer(Latency58ResidualModel):
    def __init__(self):
        super().__init__()
        self.register_buffer(BUFFER, torch.eye(4, dtype=torch.float32)[:3].clone())

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "version": VERSION,
                "output_policy_version": VERSION,
                "source_mixer_inputs": "complete four deployed residual-parent stems",
                "source_mixer_shape": [3, 4], "source_mixer_other": "physical mixture minus calibrated DBV",
                "component_semantics": "native spectral and waveform components precede source calibration",
                "extra_audio_buffering_samples": 0, "extra_stream_state_tensors": 0,
                "additional_learned_coefficients": 12, "native_host_qualified": False}

    def _render_fp32(self, audio, state):
        native = super()._render_fp32(audio, state)
        estimates = mix_sources(native.deployed, native.delayed_mixture, self.trained_source_mixer)
        return replace(native, raw=estimates, deployed=estimates)

    @classmethod
    def from_parent(cls, parent, coefficients):
        require(type(parent) is Latency58ResidualModel and coefficients.shape == (3, 4)
                and coefficients.dtype == torch.float32 and coefficients.device.type == "cpu"
                and bool(torch.isfinite(coefficients).all()), "Require a CPU residual parent and finite 3x4 coefficients")
        identity = torch.eye(4)[:3]
        lower, upper = identity - .25, identity + .25
        diagonal = torch.arange(3)
        lower[diagonal, diagonal], upper[diagonal, diagonal] = .5, 1.5
        require(bool((coefficients >= lower).all()) and bool((coefficients <= upper).all()), "Calibration exceeds its training bounds")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        model.load_state_dict({**parent.state_dict(), BUFFER: coefficients}, strict=True)
        model.provenance = {**copy.deepcopy(parent.provenance), "source_mixer_parent_state_sha256": state_sha256(parent.state_dict()),
                            "source_mixer_version": VERSION}
        return model.eval().requires_grad_(False)


def load_model(binding):
    from research.direct.latency58_direct_sdr_checkpoint import load_model as load_parent
    require(sha(binding["path"]) == binding["sha256"], "Source calibration checkpoint changed")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    require(payload["schema"] == VERSION and not payload["fit_used_validation_or_test_audio"], "Invalid source calibration provenance")
    parent, _ = load_parent(payload["parent_checkpoint"])
    model = Latency58SourceMixer.from_parent(parent, payload["coefficients"])
    require(state_sha256(model.state_dict()) == payload["model_state_sha256"]
            and payload["architecture"] == model.architecture_metadata, "Restored calibration differs")
    model.provenance = payload["provenance"]
    return model, payload


def covariances(estimates, targets, mixture):
    require(estimates.shape == targets.shape and estimates.ndim == 4 and estimates.shape[1:3] == (4, 2)
            and mixture.shape == (estimates.shape[0], 2, estimates.shape[-1]), "Require aligned four-stem stereo audio")
    windows = estimates.shape[-1] // WINDOW
    def frame(x):
        return x[..., :windows * WINDOW].reshape(x.shape[0], 4, 2, windows, WINDOW).permute(0, 3, 1, 2, 4).reshape(-1, 4, 2 * WINDOW).double()
    y, t = frame(estimates), frame(targets)
    m = mixture[..., :windows * WINDOW].reshape(mixture.shape[0], 2, windows, WINDOW).permute(0, 2, 1, 3).reshape(-1, 2 * WINDOW).double()
    return {"gram": y @ y.transpose(1, 2), "cross": y @ t.transpose(1, 2),
            "truth_energy": t.square().sum(-1), "mixture_energy": m.square().sum(-1)}


def loss_from_covariances(coefficients, statistics):
    """Exact windowed SD-SDR and relative absence energy from fixed audio moments."""
    matrix = torch.cat((coefficients, 1 - coefficients.sum(0, keepdim=True)), 0)
    prediction = torch.einsum("oi,nij,oj->no", matrix, statistics["gram"], matrix)
    correlation = torch.einsum("oi,nio->no", matrix, statistics["cross"])
    signal = statistics["truth_energy"]
    error = (prediction - 2 * correlation + signal).clamp_min(0)
    active = signal / (2 * WINDOW) > 1e-5
    values = (10 * torch.log10((error + 1e-12) / (signal + 1e-12))).clamp(-60, 60)
    count = active.sum(0)
    per_stem = torch.where(active, values, 0).sum(0) / count.clamp_min(1)
    primary = per_stem.sum() / (count > 0).sum().clamp_min(1)
    leakage = 10 * torch.log10(1 + prediction.clamp_min(0) / statistics["mixture_energy"][:, None].clamp_min(2 * WINDOW * 1e-5))
    absent_count = (~active).sum(0)
    absence = (torch.where(~active, leakage, 0).sum(0) / absent_count.clamp_min(1)).sum() / (absent_count > 0).sum().clamp_min(1)
    identity = torch.eye(4, dtype=coefficients.dtype, device=coefficients.device)[:3]
    return primary + .5 * absence + .05 * (coefficients - identity).square().mean(), -per_stem, absence


def fit(statistics, *, steps=2000):
    import math
    coefficients = torch.eye(4, dtype=torch.float64)[:3].clone().requires_grad_()
    lower, upper = coefficients.detach().clone() - .25, coefficients.detach().clone() + .25
    diagonal = torch.arange(3)
    lower[diagonal, diagonal], upper[diagonal, diagonal] = .5, 1.5
    optimizer = torch.optim.Adam([coefficients], lr=.01, foreach=False)
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = .0001 + .5 * (.01 - .0001) * (1 + math.cos(math.pi * step / (steps - 1)))
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = loss_from_covariances(coefficients, statistics)
        loss.backward()
        require(bool(torch.isfinite(loss)) and bool(torch.isfinite(coefficients.grad).all()), "Nonfinite source calibration fit")
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(lower, upper)
    return coefficients.detach().float()


def check():
    torch.manual_seed(20261001)
    truth = torch.randn(2, 4, 2, 2 * WINDOW) * .04
    truth[0, 1, :, :WINDOW] = 0
    mixture = truth.sum(1)
    estimates = truth + torch.randn_like(truth) * .015
    estimates[:, 3] = mixture - estimates[:, :3].sum(1)
    coefficients = torch.eye(4, dtype=torch.float64)[:3] + torch.randn(3, 4, dtype=torch.float64) * .02
    statistics = covariances(estimates, truth, mixture)
    _, calculated, _ = loss_from_covariances(coefficients, statistics)
    actual = mix_sources(estimates.double(), mixture.double(), coefficients)
    reference = truth.double().reshape(2, 4, 2, 2, WINDOW)
    error = (actual - truth.double()).reshape(2, 4, 2, 2, WINDOW).square().sum((2, 4))
    signal = reference.square().sum((2, 4))
    active = signal / (2 * WINDOW) > 1e-5
    expected = (10 * torch.log10((signal + 1e-12) / (error + 1e-12))).clamp(-60, 60)
    expected = torch.where(active, expected, 0).sum((0, 2)) / active.sum((0, 2)).clamp_min(1)
    discrepancy = float((calculated - expected).abs().max())
    require(discrepancy < 1e-5, "Covariance objective disagrees with direct waveform errors")
    return {"status": "pass", "covariance_vs_waveform_sdr_max_abs_db": discrepancy,
            "validation_or_test_audio_used": False, "quality_claimed": False}
