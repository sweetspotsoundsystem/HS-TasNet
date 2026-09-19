"""Current model geometry, analysis/synthesis, and spectral feature helpers."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HOP = 128
FEATURE_SAMPLES = 1024
FEATURE_HISTORY = FEATURE_SAMPLES - HOP
SYNTHESIS_SAMPLES = 2 * HOP
CROP_START = FEATURE_SAMPLES - SYNTHESIS_SAMPLES
MASK_BINS = 513
CHANNELS, SOURCES, BASIS, EMBED = 2, 4, 1500, 500
PUBLIC_FUSION_SCALE = 2.0**-18
SOURCE_ORDER = ("drums", "bass", "vocals", "other")
WINDOW, KEY, VALUE = 32, 64, 128

def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)

def overlap_frames(frames: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
    require(frames.ndim >= 2 and frames.shape[-1] == SYNTHESIS_SAMPLES
            and frames.shape[-2] >= 1
            and previous_tail.shape == (*frames.shape[:-2], HOP)
            and previous_tail.dtype == frames.dtype
            and previous_tail.device == frames.device,
            "Expected [...,F,256] frames and matching [...,128] tail")
    left, right = frames[..., :HOP], frames[..., HOP:]
    prior = torch.cat((previous_tail.unsqueeze(-2), right[..., :-1, :]), dim=-2)
    emitted = (left + prior).reshape(*frames.shape[:-2], frames.shape[-2] * HOP)
    return emitted, right[..., -1, :].clone()

def asymmetric_windows(*, dtype=torch.float32):
    """K=1024, M=128, d=0 instance of Wang et al. arXiv:2106.11794.

    The spectral synthesis window is separate from the unchanged waveform
    decoder's Hann256. Their analysis/synthesis product is Hann256.
    """
    require(dtype in (torch.float32, torch.float64), "Use FP32 or FP64 windows")
    n = torch.arange(1024, dtype=torch.float64, device="cpu")
    analysis = torch.where(n < 896, torch.sin(torch.pi * n / 1792),
                           torch.cos(torch.pi * (n - 896) / 256))
    prototype = 0.5 * (1 - torch.cos(torch.pi * torch.arange(256, dtype=torch.float64) / 128))
    synthesis = torch.cat((prototype[:128] / analysis[768:896], analysis[896:]))
    return analysis.to(dtype), synthesis.to(dtype)

class AsymmetricSynthesis(nn.Module):
    def __init__(self, analysis_window, spectral_window):
        torch.nn.Module.__init__(self)
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and analysis_window.dtype in (torch.float32, torch.float64), "Expected FP32/FP64 analysis")
        waveform = torch.hann_window(SYNTHESIS_SAMPLES, periodic=True,
                                     dtype=analysis_window.dtype, device=analysis_window.device)
        self.register_buffer("window", waveform)
        self.register_buffer("spectral_window", spectral_window.clone())
        self.register_buffer("spectral_denominator", torch.zeros_like(waveform[:HOP]))
        self.register_buffer("waveform_window_sum", waveform[:HOP] + waveform[HOP:])
        self.rebuild_denominator(analysis_window)

    def rebuild_denominator(self, analysis_window):
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and self.spectral_window.shape == (SYNTHESIS_SAMPLES,)
                and all(t.dtype == self.window.dtype and t.device == self.window.device
                        and bool(torch.isfinite(t).all()) for t in (analysis_window, self.spectral_window)),
                "Analysis and spectral synthesis window contracts differ")
        product = analysis_window[CROP_START:] * self.spectral_window
        denominator = product[:HOP] + product[HOP:]
        require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
                "Spectral overlap divisor must be finite and positive")
        with torch.no_grad():
            self.spectral_denominator.copy_(denominator)

    def spectral(self, spectrum, previous_tail):
        require(spectrum.ndim >= 2 and spectrum.is_complex() and spectrum.shape[-1] == MASK_BINS
                and spectrum.real.dtype == self.window.dtype and spectrum.device == self.window.device,
                "Expected matching complex [...,F,513] spectrum")
        frames = torch.fft.irfft(spectrum, n=FEATURE_SAMPLES, dim=-1)[..., CROP_START:]
        numerator, tail = overlap_frames(frames * self.spectral_window, previous_tail)
        return numerator / self.spectral_denominator.repeat(spectrum.shape[-2]), tail

    def waveform(self, frames: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
        require(frames.dtype == self.window.dtype and frames.device == self.window.device,
                "Waveform frames and synthesis window must share dtype/device")
        return overlap_frames(frames * self.window, previous_tail)

class MagnitudeEncoder(nn.Linear):
    """Retain the phase-sensitive linear map and add a learned magnitude map.

    The new map starts at zero. Features use only the already available FFT;
    normalization pools channels and frequency within the current frame.
    """

    def __init__(self, in_features=2052, out_features=500):
        require(in_features == 2052 and out_features == 500, "Unexpected inherited encoder")
        super().__init__(in_features, out_features)
        self.magnitude_projection = nn.Linear(1026, out_features, bias=False)
        nn.init.zeros_(self.magnitude_projection.weight)

    @staticmethod
    def magnitude_features(packed):
        with torch.autocast(packed.device.type, enabled=False):
            pairs = packed.float().unflatten(-1, (1026, 2))
            power = pairs.square().sum(-1)
            magnitude = (power + 1e-12).sqrt()
            scale = power.mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
            return torch.log1p(magnitude / scale)

    def forward(self, packed):
        original = F.linear(packed, self.weight, self.bias)
        additional = self.magnitude_projection(self.magnitude_features(packed))
        return original + additional

def cross_component_correction(carrier_ri, coefficients):
    """Return i*carrier times each real coefficient in explicit RI coordinates."""
    require(carrier_ri.shape[:-1] == coefficients.shape[:-1] and carrier_ri.shape[-1] == 2
            and coefficients.shape[-1] == SOURCES and carrier_ri.dtype == coefficients.dtype == torch.float32,
            "Require matching FP32 carrier and four-source coefficients")
    quadrature = torch.stack((-carrier_ri[..., 1], carrier_ri[..., 0]), -1)
    return quadrature.unsqueeze(-1) * coefficients.unsqueeze(-2)

def corrected_estimates(raw, delayed_mixture, share):
    # Match the reference NumPy source-axis reduction order explicitly.
    discrepancy = delayed_mixture - (((raw[:, 0] + raw[:, 1]) + raw[:, 2]) + raw[:, 3])
    retained = raw[:, :3] + share * discrepancy.unsqueeze(1)
    corrected_raw = torch.cat((retained, raw[:, 3:4]), dim=1)
    other = delayed_mixture - ((retained[:, 0] + retained[:, 1]) + retained[:, 2])
    return corrected_raw, torch.cat((retained, other.unsqueeze(1)), dim=1)
