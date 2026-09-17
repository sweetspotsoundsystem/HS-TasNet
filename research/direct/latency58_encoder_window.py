"""Prospective window pair and exact-arithmetic linear encoder conversion.

These helpers neither change a model nor run an experiment on import. The
conversion preserves spectral features in exact arithmetic, not separated
audio or BF16 rounding. A different carrier still needs quality evaluation.
"""
from __future__ import annotations

import torch

from research.direct.latency58_checkpoint import require


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


def convert_encoder_weight(weight, old_window, new_window):
    """Return CPU FP32 weights for the same two-channel linear feature map.

    Packing is [output, channel, bin, real/imag]. Bias is kept by the caller.
    Work in FP64; only the returned matrix is rounded to the original FP32.
    DC and Nyquist imaginary coefficients are unobservable for real input
    and are copied unchanged. No input tensor or random state is modified.
    """
    require(weight.ndim == 2 and weight.shape[1] == 2 * 513 * 2
            and weight.dtype == torch.float32 and weight.device.type == "cpu",
            "Use the real CPU FP32 stereo spectral encoder weight")
    require(all(w.shape == (1024,) and w.device.type == "cpu"
                and w.dtype in (torch.float32, torch.float64) for w in (old_window, new_window)),
            "Use CPU analysis windows of length 1024")
    require(all(bool(torch.isfinite(t).all()) for t in (weight, old_window, new_window)),
            "Encoder and windows must be finite")
    with torch.no_grad():
        old, new = old_window.double(), new_window.double()
        supported = new != 0
        require(bool((old[~supported] == 0).all()), "The new window discards samples used by the old encoder")
        ratio = torch.zeros_like(old)
        ratio[supported] = old[supported] / new[supported]
        packed = weight.double().reshape(weight.shape[0], 2, 513, 2)
        coefficients = torch.view_as_complex(packed.contiguous()).clone()
        coefficients[..., 1:-1] *= 0.5
        # irfft includes a factor 1/N and a factor 2 on interior bins.
        kernels = torch.fft.irfft(coefficients, n=1024, dim=-1) * 1024
        transformed = torch.fft.rfft(kernels * ratio, n=1024, dim=-1) / 1024
        transformed[..., 1:-1] *= 2
        converted = torch.view_as_real(transformed).clone()
        converted[..., 0, 1] = packed[..., 0, 1]
        converted[..., -1, 1] = packed[..., -1, 1]
        converted = converted.reshape_as(weight).float()
        require(bool(torch.isfinite(converted).all()), "Converted encoder is non-finite")
        return converted
