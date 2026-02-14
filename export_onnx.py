"""
Export HS-TasNet PyTorch checkpoint to ONNX format.

Usage:
  python export_onnx.py checkpoints/hs-tasnet.ckpt.1725.pt --output model.onnx
  python export_onnx.py checkpoints/hs-tasnet.ckpt.1725.pt --output model.onnx --opset 18
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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

    def __init__(self, real_imag_tensor: torch.Tensor):
        if not isinstance(real_imag_tensor, torch.Tensor):
            raise TypeError("real_imag_tensor must be a torch.Tensor")
        if real_imag_tensor.ndim < 1 or real_imag_tensor.shape[-1] != 2:
            raise ValueError(f"expected trailing complex dim=2, got {tuple(real_imag_tensor.shape)}")
        self._tensor = real_imag_tensor

    @classmethod
    def from_real_imag(cls, real: torch.Tensor, imag: torch.Tensor) -> "FakeComplexTensor":
        return cls(torch.stack([real, imag], dim=-1))

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
        return torch.sqrt(self.real.square() + self.imag.square() + 1e-11)

    def angle(self) -> torch.Tensor:
        return torch.atan2(self.imag, self.real)

    def view_as_real(self) -> torch.Tensor:
        return self._tensor

    def __getitem__(self, key):
        out = self._tensor[key]
        if isinstance(out, torch.Tensor) and out.ndim >= 1 and out.shape[-1] == 2:
            return FakeComplexTensor(out)
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
        self.eps = float(getattr(original_stft, "eps", 1e-11))

        window = original_stft.window.clone()
        self.register_buffer("window", window)

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
        spec = FakeComplexTensor.from_real_imag(real, imag)
        mag = torch.sqrt(real.square() + imag.square() + self.eps)
        return spec, mag

    def inverse(self, spec: Any) -> torch.Tensor:
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

        frames = time_frames.shape[-1]
        output_size = (frames - 1) * self.hop_length + self.win_length

        y = F.fold(
            time_frames,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0]

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
        self.register_buffer("weight", original_conv.weight.clone())
        self.register_buffer("window", original_conv.window.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windowed_filters = self.weight * self.window
        # Match the original implementation (no bias passed).
        return F.conv_transpose1d(
            x, windowed_filters, stride=self.stride, padding=self.padding
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

        raise NotImplementedError(f"Unsupported Rearrange pattern for ONNX export: {p}")

    def extra_repr(self) -> str:
        return f"pattern={self.pattern!r}, axes_lengths={self.axes_lengths!r}"


def _get_rearrange_pattern_and_axes(rearrange_layer: nn.Module) -> tuple[str, dict]:
    pattern = getattr(rearrange_layer, "pattern", None)
    if not isinstance(pattern, str):
        pattern = getattr(rearrange_layer, "_pattern", None)
    if not isinstance(pattern, str):
        # Fallback to repr parsing: Rearrange('...', ...)
        import re

        m = re.search(r"Rearrange\\('([^']+)'", repr(rearrange_layer))
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
    # Patterns in this repo all map to broadcast division.
    return a / b


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


def patch_model_for_onnx(model: HSTasNet) -> HSTasNet:
    """
    Replace submodules that are problematic for ONNX export.
    """
    model.stft = ConvSTFTForONNX(model.stft)
    model.conv_decode = ConvTranspose1DWithHannWindowForONNX(model.conv_decode)
    replace_rearrange_layers(model)
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

        # Save in-place. Since we did not load external tensor data, the model stays
        # an external-data model when applicable.
        onnx.save_model(model, str(path))
    except Exception as e:
        logger.warning("Failed to embed ONNX metadata into %s: %s", path, e)


class HSTasNetONNXWrapper(nn.Module):
    """
    Stateless wrapper for non-streaming ONNX export.

    Input:  audio (b, channels, samples)
    Output: separated (b, sources, channels, samples)
    """

    def __init__(self, model: HSTasNet):
        super().__init__()
        self.model = model

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
        return separated


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
    model.load_state_dict(state, strict=False)
    return model.to(device)


def export_onnx(
    ckpt_path: Path,
    output_path: Path,
    *,
    opset_version: int = 18,
    device: torch.device = torch.device("cpu"),
    external_data: bool = True,
    overlap_len: int | None = None,
) -> Path:
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

    model = patch_model_for_onnx(model)
    wrapped = HSTasNetONNXWrapper(model).eval()

    channels = 2 if model.stereo else 1
    dummy_len = int(model.segment_len)
    dummy_audio = torch.randn(1, channels, dummy_len, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: dynamic_shapes currently results in a fixed-length export for this model
    # with the torch.export-based path. dynamic_axes produces the desired dynamic
    # sample dimension, so we keep it (and silence the exporter warning).
    dynamic_axes = {"audio": {2: "samples"}, "separated": {3: "samples"}}

    with onnx_export_patches():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*dynamic_axes.*dynamo=True.*",
                category=UserWarning,
            )
            torch.onnx.export(
                wrapped,
                (dummy_audio,),
                str(output_path),
                opset_version=opset_version,
                input_names=["audio"],
                output_names=["separated"],
                dynamic_axes=dynamic_axes,
                external_data=external_data,
                dynamo=True,
            )

    # Embed provenance metadata for traceability.
    props: dict[str, str] = {
        "hs_tasnet.checkpoint_name": Path(ckpt_path).name,
        "hs_tasnet.checkpoint_sha256": _sha256_file(Path(ckpt_path)),
        "hs_tasnet.exported_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hs_tasnet.export_script": Path(__file__).name,
        "hs_tasnet.opset_version": str(opset_version),
        "hs_tasnet.external_data": "true" if external_data else "false",
        "hs_tasnet.export.dynamo": "true",
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
        "hs_tasnet.export.auto_causal_pad": "true",
        "hs_tasnet.export.auto_curtail_length_to_multiple": "false",
    }

    try:
        props["hs_tasnet.export_script_sha256"] = _sha256_file(Path(__file__))
    except Exception:
        pass

    _write_onnx_metadata(output_path, props)

    _validate_onnx(output_path)
    return output_path


def _validate_onnx(path: Path) -> None:
    try:
        import onnx

        # Pass the file path so external-data models can be resolved relative to it.
        onnx.checker.check_model(str(path))
        logger.info("ONNX validation passed: %s", path)
    except ImportError:
        logger.warning("onnx not installed; skipping validation")
    except Exception as e:
        logger.warning("ONNX validation warning for %s: %s", path, e)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HS-TasNet checkpoint to ONNX format")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint (.pt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18)")
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
    )
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
