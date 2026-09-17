"""
Export HS-TasNet PyTorch checkpoint to ONNX format.

Usage:
  python export_onnx.py checkpoints/hs-tasnet.ckpt.1673.pt --output model.onnx
  python export_onnx.py checkpoints/hs-tasnet.ckpt.1673.pt --output model.onnx --opset 18
  python export_onnx.py checkpoint.pt --mode streaming --residual-source-index 3 \
      --no-external-data --output model-streaming-mixture-consistent.onnx
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from hs_tasnet import HSTasNet

logger = logging.getLogger("hs_tasnet.export_onnx")
DEFAULT_EPS = 1e-11
DEFAULT_RESIDUAL_SOURCE_INDEX: int | None = None


# -----------------------------------------------------------------------------
# Convolutional STFT/iSTFT (ONNX-friendly)
# -----------------------------------------------------------------------------


def _arange_like(length: int, *, like: torch.Tensor) -> torch.Tensor:
    return torch.arange(length, device=like.device, dtype=like.dtype)


def create_dft_filters(
    n_fft: int, win_length: int, window: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create DFT basis filters for convolution-based STFT.

    Returns:
      cos_filters: (n_fft//2 + 1, 1, win_length)
      sin_filters: (n_fft//2 + 1, 1, win_length) (negative sign for forward DFT)
    """
    if window.ndim != 1 or window.shape[0] != win_length:
        raise ValueError(f"window must be shape ({win_length},), got {tuple(window.shape)}")

    n_freqs = n_fft // 2 + 1

    n = _arange_like(win_length, like=window)  # (win_length,)
    k = _arange_like(n_freqs, like=window)  # (n_freqs,)

    phase = (2.0 * math.pi) * k.unsqueeze(1) * n.unsqueeze(0) / float(n_fft)  # (n_freqs, win_length)

    cos_basis = torch.cos(phase)
    sin_basis = -torch.sin(phase)

    cos_filters = (cos_basis * window.unsqueeze(0)).unsqueeze(1)  # (n_freqs, 1, win_length)
    sin_filters = (sin_basis * window.unsqueeze(0)).unsqueeze(1)

    return cos_filters, sin_filters


def create_idft_filters(
    n_fft: int, win_length: int, window: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create inverse DFT synthesis matrices for convolution-based iSTFT.

    Returns:
      cos_synth: (win_length, n_fft//2 + 1)
      sin_synth: (win_length, n_fft//2 + 1)
    """
    if window.ndim != 1 or window.shape[0] != win_length:
        raise ValueError(f"window must be shape ({win_length},), got {tuple(window.shape)}")

    n_freqs = n_fft // 2 + 1

    n = _arange_like(win_length, like=window)  # (win_length,)
    k = _arange_like(n_freqs, like=window)  # (n_freqs,)

    phase = (2.0 * math.pi) * n.unsqueeze(1) * k.unsqueeze(0) / float(n_fft)  # (win_length, n_freqs)

    cos_basis = torch.cos(phase)
    sin_basis = torch.sin(phase)

    # One-sided spectrum scaling for real signals.
    # Even n_fft has a Nyquist bin that should not be doubled; odd n_fft does not.
    scale = torch.ones(n_freqs, device=window.device, dtype=window.dtype)
    if n_fft % 2 == 0:
        if n_freqs > 2:
            scale[1:-1] = 2.0
    else:
        if n_freqs > 1:
            scale[1:] = 2.0

    cos_basis = cos_basis * scale.unsqueeze(0) / float(n_fft)
    sin_basis = sin_basis * scale.unsqueeze(0) / float(n_fft)

    cos_basis = cos_basis * window.unsqueeze(1)
    sin_basis = sin_basis * window.unsqueeze(1)

    return cos_basis, sin_basis


class FakeComplexTensor:
    """
    Represent complex tensors with a real tensor of shape (..., 2) for ONNX export.
      [..., 0] = real, [..., 1] = imag
    """

    def __init__(self, real_imag_tensor: torch.Tensor, *, eps: float = DEFAULT_EPS):
        if not isinstance(real_imag_tensor, torch.Tensor):
            raise TypeError("real_imag_tensor must be a torch.Tensor")
        if real_imag_tensor.ndim < 1 or real_imag_tensor.shape[-1] != 2:
            raise ValueError(f"expected trailing complex dim=2, got {tuple(real_imag_tensor.shape)}")
        self._tensor = real_imag_tensor
        self._eps = float(eps)

    @classmethod
    def from_real_imag(
        cls, real: torch.Tensor, imag: torch.Tensor, *, eps: float = DEFAULT_EPS
    ) -> "FakeComplexTensor":
        return cls(torch.stack([real, imag], dim=-1), eps=eps)

    @property
    def real(self) -> torch.Tensor:
        return self._tensor[..., 0]

    @property
    def imag(self) -> torch.Tensor:
        return self._tensor[..., 1]

    @property
    def shape(self):
        return self._tensor.shape[:-1]

    def abs(self) -> torch.Tensor:
        # Numerical stability for magnitude computation (keep in sync with STFT eps default).
        return torch.sqrt(self.real.square() + self.imag.square() + self._eps)

    def angle(self) -> torch.Tensor:
        return torch.atan2(self.imag, self.real)

    def view_as_real(self) -> torch.Tensor:
        return self._tensor

    def __getitem__(self, key):
        out = self._tensor[key]
        if isinstance(out, torch.Tensor) and out.ndim >= 1 and out.shape[-1] == 2:
            return FakeComplexTensor(out, eps=self._eps)
        return out


class ConvSTFTForONNX(nn.Module):
    """
    ONNX-compatible STFT/iSTFT using conv1d + matmul + fold, avoiding complex dtypes.
    """

    def __init__(self, original_stft: nn.Module):
        super().__init__()
        self.n_fft = int(original_stft.n_fft)
        self.hop_length = int(original_stft.hop_length)
        self.win_length = int(original_stft.win_length)
        self.eps = float(getattr(original_stft, "eps", DEFAULT_EPS))
        self.fixed_inverse_frames: int | None = None

        window = original_stft.window.clone()
        self.register_buffer("window", window)
        streaming_envelope = getattr(original_stft, "streaming_envelope", None)
        if not isinstance(streaming_envelope, torch.Tensor):
            raise ValueError("STFT module is missing its streaming envelope")
        self.register_buffer("streaming_envelope", streaming_envelope.clone())

        cos_filters, sin_filters = create_dft_filters(self.n_fft, self.win_length, window)
        self.register_buffer("cos_filters", cos_filters)
        self.register_buffer("sin_filters", sin_filters)

        cos_synth, sin_synth = create_idft_filters(self.n_fft, self.win_length, window)
        self.register_buffer("cos_synth", cos_synth)
        self.register_buffer("sin_synth", sin_synth)

    def forward(self, audio: torch.Tensor) -> tuple[FakeComplexTensor, torch.Tensor]:
        # audio: (batch, samples)
        x = audio.unsqueeze(1)  # (batch, 1, samples)
        real = F.conv1d(x, self.cos_filters, stride=self.hop_length)
        imag = F.conv1d(x, self.sin_filters, stride=self.hop_length)
        spec = FakeComplexTensor.from_real_imag(real, imag, eps=self.eps)
        mag = torch.sqrt(real.square() + imag.square() + self.eps)
        return spec, mag

    def inverse(self, spec: Any, is_streaming: bool = False) -> torch.Tensor:
        # spec may be FakeComplexTensor, complex tensor, or (..., 2) real tensor.
        if isinstance(spec, FakeComplexTensor):
            spec_real = spec.real
            spec_imag = spec.imag
        elif isinstance(spec, torch.Tensor) and torch.is_complex(spec):
            spec_real = spec.real
            spec_imag = spec.imag
        else:
            if not isinstance(spec, torch.Tensor) or spec.ndim < 1 or spec.shape[-1] != 2:
                raise TypeError("spec must be FakeComplexTensor, complex tensor, or real tensor (..., 2)")
            spec_real = spec[..., 0]
            spec_imag = spec[..., 1]

        # (batch, n_freqs, frames) -> (batch, frames, n_freqs)
        spec_real_t = spec_real.transpose(1, 2)
        spec_imag_t = spec_imag.transpose(1, 2)

        # (batch, frames, win_length)
        time_frames = torch.matmul(spec_real_t, self.cos_synth.t()) - torch.matmul(
            spec_imag_t, self.sin_synth.t()
        )

        # (batch, win_length, frames) for fold
        time_frames = time_frames.transpose(1, 2)

        frames = (
            time_frames.shape[-1]
            if self.fixed_inverse_frames is None
            else self.fixed_inverse_frames
        )
        output_size = (frames - 1) * self.hop_length + self.win_length

        y = F.fold(
            time_frames,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0]

        if is_streaming:
            return y / self.streaming_envelope.clamp(min=self.eps)

        window_sq = self.window.square().view(1, self.win_length, 1).expand(1, self.win_length, frames)
        env = F.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).view(-1)

        return y / env.clamp(min=self.eps)


class ConvTranspose1DWithHannWindowForONNX(nn.Module):
    """
    ONNX-friendly ConvTranspose1d with a Hann window applied to the filters.
    """

    def __init__(self, original_conv: nn.Module):
        super().__init__()
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.out_channels = int(original_conv.out_channels)
        self.kernel_size = tuple(int(value) for value in original_conv.kernel_size)
        self.hann_window_baked = bool(getattr(original_conv, "hann_window_baked", False))
        self.register_buffer("weight", original_conv.weight.clone())
        self.register_buffer("window", original_conv.window.clone())

    def forward(self, x: torch.Tensor, is_streaming: bool = False) -> torch.Tensor:
        filters = self.weight if self.hann_window_baked else self.weight * self.window
        if is_streaming and self.hann_window_baked and x.shape[-1] == 1:
            decoded = F.linear(x[..., 0], filters.flatten(1).t(), bias=None)
            return decoded.view(x.shape[0], self.out_channels, self.kernel_size[0])

        # Match the original implementation (no bias passed).
        return F.conv_transpose1d(
            x, filters, stride=self.stride, padding=self.padding
        )


# -----------------------------------------------------------------------------
# ONNX-friendly Rearrange (einops.layers.torch.Rearrange replacement)
# -----------------------------------------------------------------------------


class RearrangeForONNX(nn.Module):
    def __init__(self, pattern: str, **axes_lengths: int):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pattern
        a = self.axes_lengths

        if p == "(b s) f n ... -> b n (s f ...)":
            s = int(a.get("s", 2))
            # (b*s, f, n, ...) -> (b, n, s*f*...)
            x = x.unflatten(0, (-1, s))  # (b, s, f, n, ...)
            perm = [0, 3, 1, 2] + list(range(4, x.ndim))
            x = x.permute(*perm)  # (b, n, s, f, ...)
            return x.flatten(2)

        if p == "b n (s f c t) -> (b s) f n c t":
            s = int(a.get("s", 2))
            t = int(a.get("t", 4))
            c = int(a.get("c", 2))
            b, n, flat = x.shape
            f = flat // (s * c * t)
            # (b, n, s*f*c*t) -> (b*s, f, n, c, t)
            x = x.unflatten(-1, (s, f, c, t))  # (b, n, s, f, c, t)
            x = x.permute(0, 2, 3, 1, 4, 5)  # (b, s, f, n, c, t)
            return x.flatten(0, 1)

        if p == "... (t basis) -> ... basis t":
            t = int(a.get("t", 4))
            flat = x.shape[-1]
            basis = flat // t
            # (..., t*basis) -> (..., basis, t)
            x = x.unflatten(-1, (t, basis))
            return x.transpose(-1, -2)

        if p == "b c l -> b l c":
            return x.permute(0, 2, 1)

        if p == "b s 1 n -> b s n":
            return x.squeeze(2)

        raise NotImplementedError(
            f"Unsupported Rearrange pattern for ONNX export: {p}. "
            "Add support in RearrangeForONNX.forward()."
        )

    def extra_repr(self) -> str:
        return f"pattern={self.pattern!r}, axes_lengths={self.axes_lengths!r}"


class RMSNormForONNX(nn.Module):
    """Small ONNX-friendly equivalent of ``torch.nn.RMSNorm``."""

    def __init__(self, original: nn.RMSNorm):
        super().__init__()
        if not bool(original.elementwise_affine) or original.weight is None:
            raise ValueError("ONNX RMSNorm replacement requires affine weights")
        eps = original.eps
        if eps is None:
            eps = torch.finfo(original.weight.dtype).eps
        self.eps = float(eps)
        self.register_buffer("weight", original.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class OneFrameGRUForONNX(nn.Module):
    """Exact, exportable GRU recurrence for the one-frame c91 streaming graph."""

    def __init__(self, original: nn.GRU):
        super().__init__()
        if not original.batch_first:
            raise ValueError("c91 streaming export requires a batch-first GRU")
        if original.bidirectional:
            raise ValueError("c91 streaming export requires a unidirectional GRU")
        if not original.bias:
            raise ValueError("c91 streaming export requires GRU bias tensors")
        if float(original.dropout) != 0.0:
            raise ValueError("c91 streaming export does not support GRU dropout")

        self.input_size = int(original.input_size)
        self.hidden_size = int(original.hidden_size)
        self.num_layers = int(original.num_layers)
        for layer in range(self.num_layers):
            for prefix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                value = getattr(original, f"{prefix}_l{layer}")
                self.register_buffer(f"{prefix}_l{layer}", value.detach().clone())

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_input = x[:, 0]
        next_hidden: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            previous = hidden[layer]
            input_gates = F.linear(
                layer_input,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"bias_ih_l{layer}"),
            )
            hidden_gates = F.linear(
                previous,
                getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_hh_l{layer}"),
            )
            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            current = candidate + update * (previous - candidate)
            next_hidden.append(current)
            layer_input = current

        stacked_hidden = torch.stack(next_hidden, dim=0)
        return layer_input.unsqueeze(1), stacked_hidden


def _get_rearrange_pattern_and_axes(rearrange_layer: nn.Module) -> tuple[str, dict]:
    pattern = getattr(rearrange_layer, "pattern", None)
    if not isinstance(pattern, str):
        pattern = getattr(rearrange_layer, "_pattern", None)
    if not isinstance(pattern, str):
        # Fallback to repr parsing: Rearrange('...', ...)
        import re

        try:
            m = re.search(r"Rearrange\('([^']+)'", repr(rearrange_layer))
            if not m:
                m = re.search(r'Rearrange\("([^"]+)"', repr(rearrange_layer))
        except re.error:
            m = None
        if m:
            pattern = m.group(1)

    if not isinstance(pattern, str):
        raise ValueError(f"Could not extract einops pattern from {rearrange_layer!r}")

    axes = getattr(rearrange_layer, "axes_lengths", None)
    if axes is None:
        axes = getattr(rearrange_layer, "_axes_lengths", None)
    if axes is None:
        axes = {}

    return pattern, dict(axes)


def replace_rearrange_layers(module: nn.Module) -> nn.Module:
    from einops.layers.torch import Rearrange

    for name, child in list(module.named_children()):
        if isinstance(child, Rearrange):
            pattern, axes = _get_rearrange_pattern_and_axes(child)
            setattr(module, name, RearrangeForONNX(pattern, **axes))
            logger.debug("Replaced Rearrange layer %s (%s)", name, pattern)
            continue

        replace_rearrange_layers(child)

    return module


def replace_rmsnorm_layers(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, RMSNormForONNX(child))
            logger.debug("Replaced RMSNorm layer %s", name)
            continue
        replace_rmsnorm_layers(child)
    return module


# -----------------------------------------------------------------------------
# Temporary patching (restored) for ONNX export
# -----------------------------------------------------------------------------


def _onnx_multiply(pattern: str, *args, **kwargs) -> torch.Tensor:
    if pattern == "b w f, w":
        a, w = args
        return a * w.view(1, -1, 1)

    if pattern == "o i k, k":
        filters, window = args
        return filters * window

    if pattern == "b ..., b ... t -> (b t) ...":
        a, mask = args
        t = mask.shape[-1]
        b = mask.shape[0]
        out = a.unsqueeze(-1) * mask  # (b, ..., t)
        ndim = out.ndim
        perm = [0, ndim - 1] + list(range(1, ndim - 1))  # (b, t, ...)
        out = out.permute(*perm)
        # Merge (b, t) without materializing dynamic shape tuples.
        return out.flatten(0, 1)

    if pattern == "b basis n, b n basis t -> (b t) basis n":
        basis, mask = args
        b = basis.shape[0]
        basis_size = basis.shape[1]
        n = basis.shape[2]
        t = mask.shape[-1]
        mask_bt = mask.permute(0, 2, 1, 3)  # (b, basis, n, t)
        out = basis.unsqueeze(-1) * mask_bt
        out = out.permute(0, 3, 1, 2)  # (b, t, basis, n)
        return out.reshape(b * t, basis_size, n)

    raise NotImplementedError(f"Unsupported einx.multiply pattern for ONNX export: {pattern}")


def _onnx_divide(pattern: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Be strict here: a silent wrong export is worse than a loud failure.
    p = " ".join(str(pattern).split())
    if p in ("b n, n", "b n, n -> b n"):
        return a / b
    raise NotImplementedError(f"Unsupported einx.divide pattern for ONNX export: {pattern}")


def _onnx_repeat(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    if pattern == "b ... -> (b t) ...":
        t = int(axes_lengths.get("t", 1))
        # Expand only the new t dimension; keep all others as-is (-1).
        expanded = tensor.unsqueeze(1).expand(-1, t, *([-1] * (tensor.ndim - 1)))
        return expanded.flatten(0, 1)

    if pattern == "w -> 1 w t":
        t = int(axes_lengths.get("t", 1))
        return tensor.view(1, -1, 1).expand(1, tensor.shape[0], t)

    if pattern == "n -> s n":
        s = int(axes_lengths.get("s", 2))
        return tensor.unsqueeze(0).expand(s, -1)

    if pattern == "1 d -> s d":
        s = int(axes_lengths.get("s", 2))
        return tensor.expand(s, -1)

    if pattern == "1 1 n -> 1 s n":
        s = int(axes_lengths.get("s", 2))
        return tensor.expand(1, s, -1)

    raise NotImplementedError(f"Unsupported einops.repeat pattern for ONNX export: {pattern}")


def _onnx_rearrange(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    if pattern in ("(b s t) ... -> b t s ...", "(b s t) n -> b t s n"):
        b = int(axes_lengths.get("b", 1))
        s = int(axes_lengths.get("s", 2))
        # Split dim-0 into (b, s, t) while preserving the remaining dims.
        reshaped = tensor.unflatten(0, (b, s, -1))
        perm = [0, 2, 1] + list(range(3, reshaped.ndim))
        return reshaped.permute(*perm)

    if pattern in ("(b t) ... -> b t ...", "(b t) c n -> b t c n"):
        b = int(axes_lengths.get("b", 1))
        return tensor.unflatten(0, (b, -1))

    if pattern == "b s ... -> (b s) ...":
        return tensor.flatten(0, 1)

    if pattern == "b l -> b 1 l":
        return tensor.unsqueeze(1)

    if pattern == "... -> 1 ...":
        return tensor.unsqueeze(0)

    if pattern == "1 ... -> ...":
        return tensor.squeeze(0)

    if pattern == "... 1 t -> ... t":
        return tensor.squeeze(-2)

    if pattern == "b s 1 n -> b s n":
        return tensor.squeeze(2)

    if pattern == "1 1 1 n -> n":
        return tensor.reshape(-1)

    raise NotImplementedError(f"Unsupported einops.rearrange pattern for ONNX export: {pattern}")


def _onnx_view_as_real(tensor: Any) -> torch.Tensor:
    if isinstance(tensor, FakeComplexTensor):
        return tensor.view_as_real()
    if isinstance(tensor, torch.Tensor) and torch.is_complex(tensor):
        return torch.stack([tensor.real, tensor.imag], dim=-1)
    if isinstance(tensor, torch.Tensor):
        return tensor
    raise TypeError("view_as_real expects a tensor-like input")


def _onnx_view_as_complex(tensor: Any) -> FakeComplexTensor:
    if isinstance(tensor, FakeComplexTensor):
        return tensor
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[-1] != 2:
        raise TypeError("view_as_complex expects a tensor with trailing dim=2")
    return FakeComplexTensor(tensor)


def _onnx_polar(abs_val: torch.Tensor, angle: torch.Tensor) -> FakeComplexTensor:
    real = abs_val * torch.cos(angle)
    imag = abs_val * torch.sin(angle)
    return FakeComplexTensor.from_real_imag(real, imag)


@contextmanager
def onnx_export_patches() -> Iterator[None]:
    """
    Temporarily patch hs_tasnet's imported einops/einx helpers and torch complex helpers.
    Restores everything on exit (even on exception).
    """
    import hs_tasnet.hs_tasnet as hs_mod

    original: list[tuple[Any, str, Any]] = []

    def _patch(obj: Any, name: str, value: Any) -> None:
        original.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    _patch(hs_mod, "multiply", _onnx_multiply)
    _patch(hs_mod, "divide", _onnx_divide)
    _patch(hs_mod, "repeat", _onnx_repeat)
    _patch(hs_mod, "rearrange", _onnx_rearrange)

    _patch(torch, "view_as_real", _onnx_view_as_real)
    _patch(torch, "view_as_complex", _onnx_view_as_complex)
    _patch(torch, "polar", _onnx_polar)

    try:
        yield
    finally:
        for obj, name, old in reversed(original):
            setattr(obj, name, old)


# -----------------------------------------------------------------------------
# Model patching + wrappers
# -----------------------------------------------------------------------------


def patch_model_for_onnx(model: HSTasNet, *, streaming: bool = False) -> HSTasNet:
    """
    Replace submodules that are problematic for ONNX export.
    """
    model.stft = ConvSTFTForONNX(model.stft)
    model.conv_decode = ConvTranspose1DWithHannWindowForONNX(model.conv_decode)
    replace_rmsnorm_layers(model)
    replace_rearrange_layers(model)
    if streaming:
        if bool(model.use_branch_rnns):
            raise ValueError(
                "streaming ONNX export currently supports c91 branchless recurrence only"
            )
        if not isinstance(model.fusion_branch, nn.GRU):
            raise ValueError("streaming ONNX export currently supports a fusion GRU only")
        model.fusion_branch = OneFrameGRUForONNX(model.fusion_branch)
    return model


def override_overlap_len_for_export(model: HSTasNet, overlap_len: int) -> None:
    """
    Override overlap/hop length used by the model for export.

    This changes:
    - waveform branch: conv encoder/decoder stride
    - spec branch: STFT hop_length
    - model.overlap_len / model.hop_length attributes

    NOTE: This does *not* change kernel sizes / segment_len. Using a different hop than
    training is experimental and may degrade separation quality. We also require that
    the hop cleanly divides the window and causal pad for predictable shapes.
    """

    overlap_len = int(overlap_len)
    if overlap_len <= 0:
        raise ValueError(f"overlap_len must be > 0, got {overlap_len}")
    if overlap_len >= int(model.segment_len):
        raise ValueError(
            f"overlap_len must be < segment_len ({int(model.segment_len)}), got {overlap_len}"
        )

    segment_len = int(model.segment_len)
    causal_pad = int(getattr(model, "causal_pad", segment_len // 2))

    if segment_len % overlap_len != 0:
        raise ValueError(
            f"segment_len ({segment_len}) must be divisible by overlap_len ({overlap_len})"
        )
    if causal_pad % overlap_len != 0:
        raise ValueError(
            f"causal_pad ({causal_pad}) must be divisible by overlap_len ({overlap_len})"
        )

    model.overlap_len = overlap_len
    model.hop_length = overlap_len

    if hasattr(model, "stft") and hasattr(model.stft, "hop_length"):
        model.stft.hop_length = overlap_len

    if hasattr(model, "conv_encode") and hasattr(model.conv_encode, "stride"):
        model.conv_encode.stride = (overlap_len,)

    if hasattr(model, "conv_decode") and hasattr(model.conv_decode, "stride"):
        model.conv_decode.stride = (overlap_len,)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_onnx_metadata(path: Path, props: dict[str, str]) -> None:
    """
    Persist metadata into the ONNX model file.

    Notes:
    - We load with load_external_data=False to avoid reading large .onnx.data blobs.
    - We overwrite keys if they already exist.
    """
    try:
        import onnx
    except ImportError:
        logger.warning("onnx not installed; skipping ONNX metadata embedding")
        return

    try:
        model = onnx.load_model(str(path), load_external_data=False)
        existing = {p.key: p.value for p in model.metadata_props}
        existing.update({str(k): str(v) for k, v in props.items()})

        del model.metadata_props[:]
        for k in sorted(existing.keys()):
            entry = model.metadata_props.add()
            entry.key = k
            entry.value = existing[k]

        # Save in-place. We intentionally do not load external tensor data to avoid
        # reading large .onnx.data blobs; this should preserve existing external-data
        # references, but behavior may vary across onnx versions.
        onnx.save_model(model, str(path))
    except Exception as e:
        logger.warning("Failed to embed ONNX metadata into %s: %s", path, e)


def resolve_residual_source_index(
    num_sources: int, residual_source_index: int | None
) -> int | None:
    if residual_source_index is None:
        return None
    index = int(residual_source_index)
    if index < 0:
        index += int(num_sources)
    if not 0 <= index < int(num_sources):
        raise ValueError(
            f"residual source index {residual_source_index} is outside "
            f"the {num_sources}-source output"
        )
    return index


def route_mixture_residual(
    separated: torch.Tensor,
    mixture: torch.Tensor,
    *,
    residual_source_index: int,
) -> torch.Tensor:
    """Route all reconstruction residual to one stem without changing the others."""
    index = int(residual_source_index)
    before = separated[:, :index]
    after = separated[:, index + 1 :]
    non_residual_sources = torch.cat((before, after), dim=1)
    residual_source = mixture - non_residual_sources.sum(dim=1)
    return torch.cat(
        (before, residual_source.unsqueeze(1), after),
        dim=1,
    )


class HSTasNetONNXWrapper(nn.Module):
    """
    Stateless wrapper for non-streaming ONNX export.

    Input:  audio (b, channels, samples)
    Output: separated (b, sources, channels, samples)
    """

    def __init__(
        self,
        model: HSTasNet,
        *,
        residual_source_index: int | None = DEFAULT_RESIDUAL_SOURCE_INDEX,
    ):
        super().__init__()
        self.model = model
        self.residual_source_index = resolve_residual_source_index(
            model.num_sources, residual_source_index
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # The underlying model is designed around segment-based processing.
        # For best results and stable output shapes, feed inputs padded to a
        # multiple of `model.segment_len` (e.g. 1024 for the default checkpoints).
        separated, _ = self.model(
            audio,
            hiddens=None,
            auto_causal_pad=True,
            auto_curtail_length_to_multiple=False,
        )
        if self.residual_source_index is not None:
            separated = route_mixture_residual(
                separated,
                audio,
                residual_source_index=self.residual_source_index,
            )
        return separated


class HSTasNetStreamingONNXWrapper(nn.Module):
    """Explicit-state, one-hop c91 streaming contract for ONNX Runtime.

    The emitted chunk is aligned with ``past_audio``. Initialize every state input
    to zero, then feed one final zero audio chunk to flush the last real chunk.
    """

    def __init__(
        self,
        model: HSTasNet,
        *,
        residual_source_index: int | None = DEFAULT_RESIDUAL_SOURCE_INDEX,
    ):
        super().__init__()
        if bool(model.use_branch_rnns):
            raise ValueError("streaming wrapper requires c91 branch RNNs to be disabled")
        if not isinstance(model.fusion_branch, OneFrameGRUForONNX):
            raise ValueError("streaming wrapper requires the exportable one-frame GRU")
        self.model = model
        self.residual_source_index = resolve_residual_source_index(
            model.num_sources, residual_source_index
        )

    def forward(
        self,
        audio_chunk: torch.Tensor,
        past_audio: torch.Tensor,
        overlap_add_buffer: torch.Tensor,
        fusion_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full_chunk = torch.cat((past_audio, audio_chunk), dim=-1)
        window_output, next_hiddens = self.model(
            full_chunk,
            hiddens=(None, None, fusion_hidden, None, None),
            auto_causal_pad=False,
            auto_curtail_length_to_multiple=False,
            is_streaming=True,
        )
        overlap = overlap_add_buffer + window_output
        chunk_samples = int(self.model.overlap_len)
        separated_chunk = overlap[..., :chunk_samples]
        next_overlap_add_buffer = torch.cat(
            (
                overlap[..., chunk_samples:],
                torch.zeros_like(separated_chunk),
            ),
            dim=-1,
        )
        if self.residual_source_index is not None:
            separated_chunk = route_mixture_residual(
                separated_chunk,
                past_audio,
                residual_source_index=self.residual_source_index,
            )

        next_fusion_hidden = next_hiddens[2]
        if next_fusion_hidden is None:
            raise RuntimeError("c91 fusion GRU did not return its next hidden state")
        # Clone prevents torch.export from alias-renaming the audio input as an output.
        next_past_audio = audio_chunk.clone()
        return (
            separated_chunk,
            next_past_audio,
            next_overlap_add_buffer,
            next_fusion_hidden,
        )


# -----------------------------------------------------------------------------
# Load + export
# -----------------------------------------------------------------------------


def load_model(ckpt_path: Path, device: torch.device) -> HSTasNet:
    """
    Load model from a checkpoint.

    Prefer checkpoints saved via `HSTasNet.save()` (they include config). If config is
    missing, fall back to a best-effort heuristic.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    pkg = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(pkg, dict) and "config" in pkg and "model" in pkg:
        # Standard saved checkpoint format.
        import pickle

        config = pickle.loads(pkg["config"])
        model = HSTasNet(**config)
        model.load_state_dict(pkg["model"], strict=True)
        return model.to(device)

    # Heuristic fallback.
    logger.warning(
        "Checkpoint is missing embedded config; using heuristic config inference. "
        "Only run this export on trusted checkpoints."
    )
    state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format (expected a state_dict-like object)")

    conv_w = state.get("conv_encode.weight", None)
    if conv_w is None:
        raise ValueError("Could not infer config: missing conv_encode.weight in state dict")

    audio_channels = int(conv_w.shape[1])
    segment_len = int(conv_w.shape[-1])
    num_basis = int(conv_w.shape[0] // 2)
    stereo = audio_channels == 2

    # Defaults consistent with the original codebase assumptions.
    overlap_len = segment_len // 2
    n_fft = segment_len

    # Infer small (1-layer vs 2-layer) by presence of layer-1 weights.
    small = True
    for k in state.keys():
        if ".weight_ih_l1" in k:
            small = False
            break

    # Infer spec_branch_use_phase from spec encoder input dim.
    spec_branch_use_phase = True
    spec_w = state.get("spec_encode.1.weight", None)
    if isinstance(spec_w, torch.Tensor) and spec_w.ndim == 2:
        spec_in_dim = int(spec_w.shape[1])
        base = (n_fft // 2 + 1) * (2 if stereo else 1)
        if spec_in_dim == base:
            spec_branch_use_phase = False
        elif spec_in_dim == base * 2:
            spec_branch_use_phase = True

    # Infer dim.
    dim = int(spec_w.shape[0]) if isinstance(spec_w, torch.Tensor) else 500

    # Infer num_sources from waveform masks linear if available.
    num_sources = 4
    wf_w = state.get("to_waveform_masks.1.weight", None)
    if isinstance(wf_w, torch.Tensor) and wf_w.ndim == 2 and num_basis > 0:
        out_features = int(wf_w.shape[0])
        if out_features % num_basis == 0:
            num_sources = out_features // num_basis

    model = HSTasNet(
        dim=dim,
        small=small,
        stereo=stereo,
        num_basis=num_basis,
        segment_len=segment_len,
        overlap_len=overlap_len,
        n_fft=n_fft,
        num_sources=num_sources,
        spec_branch_use_phase=spec_branch_use_phase,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = getattr(incompatible, "missing_keys", None)
    unexpected = getattr(incompatible, "unexpected_keys", None)
    if missing:
        logger.warning("Missing keys when loading checkpoint (strict=False): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint (strict=False): %s", unexpected)
    return model.to(device)


def export_onnx(
    ckpt_path: Path,
    output_path: Path,
    *,
    opset_version: int = 18,
    device: torch.device = torch.device("cpu"),
    external_data: bool = True,
    overlap_len: int | None = None,
    mode: str = "offline",
    residual_source_index: int | None = DEFAULT_RESIDUAL_SOURCE_INDEX,
    offline_samples: int | None = None,
) -> Path:
    if mode not in {"offline", "streaming"}:
        raise ValueError(f"unsupported ONNX export mode: {mode!r}")
    model = load_model(ckpt_path, device)
    model.eval()

    if overlap_len is not None and int(overlap_len) != int(model.overlap_len):
        logger.warning(
            "Overriding overlap_len for export: %s -> %s (segment_len=%s). This is experimental.",
            int(model.overlap_len),
            int(overlap_len),
            int(model.segment_len),
        )
        override_overlap_len_for_export(model, int(overlap_len))

    streaming = mode == "streaming"
    model = patch_model_for_onnx(model, streaming=streaming)

    channels = 2 if model.stereo else 1
    if streaming:
        if int(model.segment_len) != 2 * int(model.overlap_len):
            raise ValueError(
                "streaming export requires segment_len == 2 * overlap_len"
            )
        wrapped: nn.Module = HSTasNetStreamingONNXWrapper(
            model,
            residual_source_index=residual_source_index,
        ).eval()
        fusion = model.fusion_branch
        if not isinstance(fusion, OneFrameGRUForONNX):
            raise AssertionError("streaming fusion GRU replacement was not installed")
        chunk_samples = int(model.overlap_len)
        dummy_audio = torch.randn(1, channels, chunk_samples, device=device)
        dummy_past_audio = torch.zeros_like(dummy_audio)
        dummy_overlap = torch.zeros(
            1,
            int(model.num_sources),
            channels,
            int(model.segment_len),
            device=device,
        )
        dummy_hidden = torch.zeros(
            fusion.num_layers,
            1,
            fusion.hidden_size,
            device=device,
        )
        export_args = (
            dummy_audio,
            dummy_past_audio,
            dummy_overlap,
            dummy_hidden,
        )
        input_names = [
            "audio_chunk",
            "past_audio",
            "overlap_add_buffer",
            "fusion_hidden",
        ]
        output_names = [
            "separated_chunk",
            "next_past_audio",
            "next_overlap_add_buffer",
            "next_fusion_hidden",
        ]
        dynamic_axes = None
        dynamo = True
    else:
        wrapped = HSTasNetONNXWrapper(
            model,
            residual_source_index=residual_source_index,
        ).eval()
        dummy_len = int(model.segment_len) if offline_samples is None else int(offline_samples)
        if dummy_len <= 0 or dummy_len % int(model.segment_len) != 0:
            raise ValueError(
                "offline sample count must be a positive multiple of segment_len"
            )
        dummy_audio = torch.randn(1, channels, dummy_len, device=device)
        if not isinstance(model.stft, ConvSTFTForONNX):
            raise AssertionError("offline STFT replacement was not installed")
        padded_samples = dummy_len + 2 * int(model.causal_pad)
        model.stft.fixed_inverse_frames = (
            padded_samples - int(model.stft.win_length)
        ) // int(model.stft.hop_length) + 1
        export_args = (dummy_audio,)
        input_names = ["audio"]
        output_names = ["separated"]
        # Keep the offline graph fixed-length. ONNX Col2Im cannot represent this
        # model's dynamically computed overlap-add output size at opset 18.
        dynamic_axes = None
        dynamo = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with onnx_export_patches():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*dynamic_axes.*dynamo=True.*",
                category=UserWarning,
            )
            torch.onnx.export(
                wrapped,
                export_args,
                str(output_path),
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                external_data=external_data,
                dynamo=dynamo,
            )

    # Embed provenance metadata for traceability.
    props: dict[str, str] = {
        "hs_tasnet.checkpoint_name": Path(ckpt_path).name,
        "hs_tasnet.checkpoint_sha256": _sha256_file(Path(ckpt_path)),
        "hs_tasnet.exported_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hs_tasnet.export_script": Path(__file__).name,
        "hs_tasnet.opset_version": str(opset_version),
        "hs_tasnet.external_data": "true" if external_data else "false",
        "hs_tasnet.export.mode": mode,
        "hs_tasnet.export.dynamo": "true" if dynamo else "false",
        "hs_tasnet.torch_version": str(getattr(torch, "__version__", "")),
        "hs_tasnet.model.segment_len": str(int(getattr(model, "segment_len", 0))),
        "hs_tasnet.model.overlap_len": str(int(getattr(model, "overlap_len", 0))),
        "hs_tasnet.model.n_fft": str(int(getattr(model, "n_fft", 0))),
        "hs_tasnet.model.num_sources": str(int(getattr(model, "num_sources", 0))),
        "hs_tasnet.model.stereo": "true" if bool(getattr(model, "stereo", False)) else "false",
        "hs_tasnet.model.small": "true" if bool(getattr(model, "small", False)) else "false",
        "hs_tasnet.model.spec_branch_use_phase": (
            "true" if bool(getattr(model, "spec_branch_use_phase", False)) else "false"
        ),
        "hs_tasnet.model.use_branch_rnns": (
            "true" if bool(getattr(model, "use_branch_rnns", False)) else "false"
        ),
        "hs_tasnet.model.decoder_hann_baked": (
            "true"
            if bool(getattr(model.conv_decode, "hann_window_baked", False))
            else "false"
        ),
        "hs_tasnet.model.output_source_scales": json.dumps(
            [
                float(value)
                for value in (
                    model.output_source_scales.detach().cpu().tolist()
                    if isinstance(model.output_source_scales, torch.Tensor)
                    else []
                )
            ],
            separators=(",", ":"),
        ),
        "hs_tasnet.export.mixture_consistency": (
            "route_residual" if residual_source_index is not None else "disabled"
        ),
        "hs_tasnet.export.residual_source_index": (
            str(resolve_residual_source_index(model.num_sources, residual_source_index))
            if residual_source_index is not None
            else ""
        ),
        "hs_tasnet.export.auto_causal_pad": "false" if streaming else "true",
        "hs_tasnet.export.auto_curtail_length_to_multiple": "false",
        "hs_tasnet.export.input_names": json.dumps(input_names, separators=(",", ":")),
        "hs_tasnet.export.output_names": json.dumps(output_names, separators=(",", ":")),
    }
    if streaming:
        props.update(
            {
                "hs_tasnet.streaming.chunk_samples": str(int(model.overlap_len)),
                "hs_tasnet.streaming.analysis_window_samples": str(
                    int(model.segment_len)
                ),
                "hs_tasnet.streaming.output_alignment": "previous_input_chunk",
                "hs_tasnet.streaming.output_delay_hops": "1",
                "hs_tasnet.streaming.initial_state": "all_zeros",
                "hs_tasnet.streaming.preroll": "discard_first_output_chunk",
                "hs_tasnet.streaming.flush": "append_one_zero_audio_chunk",
                "hs_tasnet.streaming.fusion_hidden_shape": json.dumps(
                    [fusion.num_layers, 1, fusion.hidden_size], separators=(",", ":")
                ),
                "hs_tasnet.streaming.overlap_add_buffer_shape": json.dumps(
                    [1, model.num_sources, channels, model.segment_len],
                    separators=(",", ":"),
                ),
            }
        )
    else:
        props["hs_tasnet.offline.input_samples"] = str(dummy_len)

    try:
        props["hs_tasnet.export_script_sha256"] = _sha256_file(Path(__file__))
    except Exception:
        pass

    _write_onnx_metadata(output_path, props)

    _validate_onnx(
        output_path,
        expected_inputs=input_names,
        expected_outputs=output_names,
    )
    return output_path


def _validate_onnx(
    path: Path,
    *,
    expected_inputs: list[str],
    expected_outputs: list[str],
) -> None:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required to validate an exported model") from error

    try:
        # Pass the file path so external-data models can be resolved relative to it.
        onnx.checker.check_model(str(path))
        graph = onnx.load_model(str(path), load_external_data=False).graph
        actual_inputs = [value.name for value in graph.input]
        actual_outputs = [value.name for value in graph.output]
        if actual_inputs != expected_inputs:
            raise ValueError(
                f"ONNX input contract changed: {actual_inputs} != {expected_inputs}"
            )
        if actual_outputs != expected_outputs:
            raise ValueError(
                f"ONNX output contract changed: {actual_outputs} != {expected_outputs}"
            )
        logger.info("ONNX validation passed: %s", path)
    except Exception as error:
        raise RuntimeError(f"ONNX validation failed for {path}: {error}") from error


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HS-TasNet checkpoint to ONNX format")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint (.pt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18)")
    parser.add_argument(
        "--mode",
        choices=("offline", "streaming"),
        default="offline",
        help="Export a fixed-length offline graph or the explicit-state c91 streaming graph",
    )
    parser.add_argument(
        "--residual-source-index",
        type=int,
        default=None,
        help=(
            "Route the mixture reconstruction residual to this source index. "
            "For c91 source order [drums,bass,vocals,other], use 3 to make the "
            "deployment mixture-consistent. Disabled unless explicitly supplied."
        ),
    )
    parser.add_argument(
        "--offline-samples",
        type=int,
        default=None,
        help="Fixed input sample count for offline export; must be a segment_len multiple",
    )
    parser.add_argument(
        "--overlap-len",
        type=int,
        default=None,
        help=(
            "Override model overlap_len / STFT hop_length / conv stride used during export. "
            "Experimental; may change separation quality. "
            "For default checkpoints (segment_len=1024, overlap_len=512), use 256 for 2x smaller hop."
        ),
    )
    parser.add_argument("--no-external-data", action="store_true", help="Disable external data files")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ckpt_path.with_suffix(".onnx")

    external_data = not args.no_external_data
    device = torch.device("cpu")

    export_onnx(
        ckpt_path=ckpt_path,
        output_path=output_path,
        opset_version=args.opset,
        device=device,
        external_data=external_data,
        overlap_len=args.overlap_len,
        mode=args.mode,
        residual_source_index=args.residual_source_index,
        offline_samples=args.offline_samples,
    )
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
