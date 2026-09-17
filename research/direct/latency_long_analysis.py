"""Opt-in trailing-2048 magnitude features for C191's native separator.

The native core and its 1024-point carrier/512-sample synthesis are retained.
Only the spectral embedding gains an explicit additive analysis feature. The
extra public state holds older input audio, never corrected output audio.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import research.direct.causal_core as core_ops
from research.direct.latency11_c191 import (
    C191Model, DEFAULT_CORE_CONFIG, FUSION_SCALE, HOP, _ExactAdd,
)


LONG_ANALYSIS_VERSION = "mag2048-v1"
LONG_ANALYSIS_METADATA = {
    "n_fft": 2048,
    "hop_samples": 512,
    "history_samples": 1536,
    "older_state_samples": 1024,
    "window": "hann",
    "periodic": True,
    "norm": "ortho",
    "feature": "magnitude",
    "stereo_flatten": "LR_frequency",
    "input_features": 2050,
    "output_features": 500,
    "bias": False,
}


def _native_core_config(core_config):
    config = dict(DEFAULT_CORE_CONFIG if core_config is None else core_config)
    if config != DEFAULT_CORE_CONFIG:
        changed = sorted(
            key for key in set(config) | set(DEFAULT_CORE_CONFIG)
            if config.get(key) != DEFAULT_CORE_CONFIG.get(key)
            or (key in config) != (key in DEFAULT_CORE_CONFIG)
        )
        raise ValueError(f"Long analysis requires the native C191 core configuration: {changed}")
    return config


class LongAnalysisFeatures(nn.Module):
    """Trailing stereo magnitude frames, then a zero-start 2050→500 map."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(2048, periodic=True, dtype=torch.float32))
        # nn.Linear's default initialization would consume additional RNG draws.
        self.weight = nn.Parameter(torch.zeros(500, 2050, dtype=torch.float32))

    def forward(self, joined: Tensor) -> Tensor:
        """Map [B,2,1536+T] to [B,T/512,500], for positive hop-multiple T."""
        if (joined.ndim != 3 or joined.shape[1] != 2 or joined.shape[-1] < 2048
                or (joined.shape[-1] - 1536) % HOP):
            raise ValueError("Long analysis requires [B,2,1536+T], T a positive multiple of 512")
        with torch.autocast(joined.device.type, enabled=False):
            frames = joined.float().unfold(-1, 2048, HOP)
            spectrum = torch.fft.rfft(
                frames * self.window.float(), n=2048, dim=-1, norm="ortho",
            )
            magnitude = spectrum.abs()
            # [B,stereo,frames,bins] -> [B,frames,all L bins then all R bins].
            features = magnitude.permute(0, 2, 1, 3).reshape(joined.shape[0], -1, 2050)
        return F.linear(features, self.weight, bias=None)


def _forward_c191_core(core, audio: Tensor, fusion_hidden: Tensor, delta: Tensor):
    """Native fixed-C191 path with an explicit spectral-embedding delta.

    ``audio`` contains native past512 plus current input; ``fusion_hidden``
    is in the core's physical units. Module-qualified operations and current
    submodule lookups also see the existing export-copy replacements.
    """
    batch = audio.shape[0]
    spec_audio_input = core_ops.rearrange(audio, "b s ... -> (b s) ...")
    complex_spec, _ = core.stft(spec_audio_input)
    real_imag_spec = core_ops.torch.view_as_real(complex_spec)
    spec = core.spec_encode(real_imag_spec)
    if delta.shape != spec.shape:
        raise ValueError("Long-analysis and native spectral frames must have the same shape")
    spec = _ExactAdd.apply(spec, delta.to(dtype=spec.dtype))

    synthesis_complex_spec = core.stft.current_chunk_carrier(spec_audio_input)
    synthesis_real_imag_spec = core_ops.torch.view_as_real(synthesis_complex_spec)
    to_relu, to_sigmoid = core.conv_encode(audio).chunk(2, dim=1)
    basis = to_relu.relu() * to_sigmoid.sigmoid()
    waveform = core.basis_to_embed(basis)
    spec_residual, waveform_residual = spec, waveform
    fusion_input = core_ops.cat((spec, waveform), dim=-1)
    fused, next_fusion_hidden = core_ops.residual(core.fusion_branch)(fusion_input, fusion_hidden)
    fused_spec, fused_waveform = fused.chunk(2, dim=-1)
    spec = fused_spec + spec_residual
    waveform = fused_waveform + waveform_residual

    spec_mask = core.apply_residual_source_softmax(core.to_spec_masks(spec))
    scaled_real_imag = core_ops.multiply(
        "b ..., b ... t -> (b t) ...", synthesis_real_imag_spec, spec_mask,
    )
    complex_per_source = core_ops.torch.view_as_complex(scaled_real_imag.contiguous())
    from_spec = core.stft.inverse(complex_per_source, is_streaming=True)
    from_spec = core_ops.rearrange(
        from_spec, "(b s t) ... -> b t s ...", b=batch, s=core.audio_channels,
    )
    waveform_mask = core.apply_residual_source_softmax(core.to_waveform_masks(waveform))
    basis_per_source = core_ops.multiply(
        "b basis n, b n basis t -> (b t) basis n", basis, waveform_mask,
    )
    from_waveform = core.conv_decode(basis_per_source, is_streaming=True)
    from_waveform = core_ops.rearrange(from_waveform, "(b t) ... -> b t ...", b=batch)
    raw = from_spec + from_waveform
    scales = core.output_source_scales[None, :, None, None]
    raw.mul_(scales)
    components = (from_spec * scales, from_waveform * scales)
    hiddens = (None, None, next_fusion_hidden, None, None)
    return raw, hiddens, components


class C191LongAnalysisModel(C191Model):
    """Native C191 plus mag2048-v1 and one older-input-history state."""

    def __init__(self, core_config=None, *, head_phase_features=False,
                 corrections_fp32=False, _native_state=None):
        if head_phase_features:
            raise ValueError("mag2048-v1 requires the native C191 head without phase expansion")
        config = _native_core_config(core_config)
        super().__init__(config, corrections_fp32=corrections_fp32, head_phase_features=False)
        if _native_state is not None:
            # At this point the state schema is exactly the original C191 schema.
            self.load_state_dict(_native_state, strict=True)
        self.long_analysis = LongAnalysisFeatures()

    @property
    def long_analysis_metadata(self):
        return dict(LONG_ANALYSIS_METADATA)

    @classmethod
    def from_native(cls, native_engine):
        """Upgrade an ordinary CPU/FP32 C191 without consuming global RNG."""
        if type(native_engine) is not C191Model or native_engine.head.phase_features:
            raise ValueError("from_native requires an ordinary C191 with its native head")
        config = _native_core_config(native_engine.core_config)
        state = native_engine.state_dict()
        if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in state.values()):
            raise ValueError("from_native requires every native parameter and buffer on CPU in FP32")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls(config, corrections_fp32=native_engine.corrections_fp32, _native_state=state)
        model.train(native_engine.training)
        return model

    def initial_state(self, batch_size, device=None, dtype=torch.float32):
        native = super().initial_state(batch_size, device=device, dtype=dtype)
        older = torch.zeros(batch_size, 2, 1024, device=native[0].device, dtype=dtype)
        return (*native, older)

    def forward(self, audio, state=None, *, return_raw=False):
        if (audio.ndim != 3 or audio.shape[1] != 2 or audio.shape[-1] < HOP
                or audio.shape[-1] % HOP):
            raise ValueError("C191 audio must have shape [B,2,T], T a positive multiple of 512")
        if state is None:
            state = self.initial_state(audio.shape[0], device=audio.device)
        if len(state) != 8:
            raise ValueError("mag2048-v1 requires the seven native states plus older_audio")
        native_state = state[:7]
        past, fusion = native_state[:2]
        older = state[7]
        if past.shape != (audio.shape[0], 2, 512) or older.shape != (audio.shape[0], 2, 1024):
            raise ValueError("mag2048-v1 input histories must have shapes [B,2,512] and [B,2,1024]")
        joined = torch.cat((older, past, audio), dim=-1)
        delta = self.long_analysis(joined)
        raw, hiddens, components = _forward_c191_core(
            self.core, torch.cat((past, audio), dim=-1), fusion / FUSION_SCALE, delta,
        )
        if self.corrections_fp32:
            with torch.autocast(audio.device.type, enabled=False):
                correction_state = (*native_state[:2], *(value.float() for value in native_state[2:]))
                stems, next_native = self._forward_corrections(
                    audio.float(), correction_state, raw.float(), hiddens,
                    tuple(value.float() for value in components), return_raw=return_raw,
                )
        else:
            stems, next_native = self._forward_corrections(
                audio, native_state, raw, hiddens, components, return_raw=return_raw,
            )
        return stems, (*next_native, joined[..., -1536:-512].clone())
