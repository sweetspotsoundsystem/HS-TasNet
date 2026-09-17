"""Causal trailing-4096 normalized power-band features for the magnitude model.

The extra feature path reads only received audio. The 1024-point carrier,
cropped256 synthesis, recurrent cadence and residual correction are retained.
The audio-history state is longer; this is a separate, unqualified state ABI.
"""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from research.direct.latency58 import (
    BASIS, CHANNELS, EMBED, FEATURE_HISTORY, FEATURE_SAMPLES, HOP, MASK_BINS,
    PUBLIC_FUSION_SCALE, SOURCES, SYNTHESIS_SAMPLES, require,
)
from research.direct.latency58_magnitude import Latency58MagnitudeModel
from research.direct.latency58_residual_model import ResidualShareOutput, corrected_estimates
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-long4096-normalized-power369-v1"
LONG_SAMPLES, LONG_HISTORY, LONG_BANDS = 4096, 3968, 369
ADAPTER = "long_projection.weight"


class LongMagnitudeState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


class Latency58LongMagnitudeModel(Latency58MagnitudeModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("long_window", torch.hann_window(LONG_SAMPLES, periodic=True))
        self.long_projection = nn.Linear(2 * LONG_BANDS, EMBED, bias=False)
        nn.init.zeros_(self.long_projection.weight)

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "version": VERSION, "state_family": VERSION,
                "feature_history_samples": LONG_HISTORY, "state_names": list(LongMagnitudeState._fields),
                "long_feature_n_fft": LONG_SAMPLES, "long_feature_window": "periodic Hann4096, trailing received input",
                "long_feature_bands_per_channel": LONG_BANDS,
                "long_feature_pooling": "bins 0:128 individually; bins 128:2048 in 240 groups of eight; Nyquist individually",
                "long_feature_normalization": "log1p(sqrt(pooled_power+1e-12)/sqrt(max(stereo_frame_mean_power,1e-8)))",
                "long_feature_projection_parameters": 369000,
                "additional_neural_parameters": 513000 + 369000,
                "extra_history_samples_relative_to_magnitude": LONG_HISTORY - FEATURE_HISTORY,
                "additional_audio_buffering_samples": 0, "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        inherited = super().initial_state(batch_size, device=device)
        history = inherited.audio_history.new_zeros((batch_size, CHANNELS, LONG_HISTORY))
        return LongMagnitudeState(history, *inherited[1:])

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "Invalid long-feature input")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(type(state) is LongMagnitudeState, "Use the distinct long-feature state family")
        shapes = ((audio.shape[0], CHANNELS, LONG_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "Long-feature state shape/dtype/device differs")
        return state

    def long_features(self, joined):
        with torch.autocast(joined.device.type, enabled=False):
            frames = joined.unfold(-1, LONG_SAMPLES, HOP)
            spectrum = torch.fft.rfft(frames * self.long_window, n=LONG_SAMPLES, dim=-1)
            power = torch.view_as_real(spectrum).square().sum(-1)
            middle = power[..., 128:2048].unflatten(-1, (240, 8)).mean(-1)
            pooled = torch.cat((power[..., :128], middle, power[..., -1:]), dim=-1)
            scale = power.mean((1, 3), keepdim=True).clamp_min(1e-8).sqrt()
            features = torch.log1p((pooled + 1e-12).sqrt() / scale)
            return features.permute(0, 2, 1, 3).flatten(2)

    def _render_fp32(self, audio, state):
        return self._render_impl(audio, state, tail_only=False)

    @torch.no_grad()
    def warm_state(self, audio, state=None):
        state = self._validate(audio, state)
        with torch.autocast(audio.device.type, enabled=False):
            return self._render_impl(audio, state, tail_only=True)

    def _render_impl(self, audio, state, *, tail_only):
        bf16 = self.training and self.training_precision == "bf16"
        require(not self.training or self.training_precision in ("fp32", "bf16"), "Unknown precision")
        require(not bf16 or audio.device.type == "cuda", "BF16 learned operations require CUDA")
        def learned():
            return torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16)
        batch, _, samples = audio.shape
        joined = torch.cat((state.audio_history, audio), dim=-1)
        short_joined = joined[..., -(FEATURE_HISTORY + samples):]
        feature = torch.fft.rfft(short_joined.unfold(-1, FEATURE_SAMPLES, HOP) * self.analysis_window,
                                 n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        additional = self.long_features(joined)
        with learned():
            spec = self.spec_encode(packed) + self.long_projection(additional)
            convolution = self.conv_encode(short_joined)
        to_relu, to_sigmoid = convolution.float().chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        with learned():
            waveform = self.basis_to_embed(basis).transpose(1, 2)
        spec, waveform = spec.float(), waveform.float()
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        with learned():
            recurrent, hidden = self.fusion_branch(
                fusion_input.to(torch.bfloat16) if bf16 else fusion_input,
                physical_hidden.to(torch.bfloat16) if bf16 else physical_hidden)
        if tail_only:
            spec, waveform, fusion_input, recurrent = (value[:, -1:] for value in
                                                       (spec, waveform, fusion_input, recurrent))
            basis, feature = basis[:, :, -1:], feature[:, :, -1:]
        frame_count = spec.shape[1]
        fused_spec, fused_waveform = (fusion_input + recurrent.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec)
        waveform = self.waveform_norm(fused_waveform + waveform)
        with learned():
            spec_logits = self.to_spec_masks(spec)
            waveform_logits = self.to_waveform_masks(waveform)
        spec_logits = spec_logits.float().reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        masked = torch.view_as_real(feature).unsqueeze(-1) * masks
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spec_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)
        waveform_logits = waveform_logits.float().reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        masks = self._residual_source_softmax(waveform_logits)
        source_basis = (basis.transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with learned():
            decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.float().reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
        waveform_audio, wave_tail = self.synthesis.waveform(decoded, state.waveform_tail)
        next_state = LongMagnitudeState(joined[..., -LONG_HISTORY:].clone(),
                                       hidden.float() * PUBLIC_FUSION_SCALE, spec_tail, wave_tail)
        if tail_only:
            return next_state
        scales = self.output_source_scales[None, :, None, None]
        native_raw = (spectral + waveform_audio) * scales
        mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        raw, deployed = corrected_estimates(native_raw, mixture, self.fixed_residual_share)
        return ResidualShareOutput(raw, deployed, spectral * scales, waveform_audio * scales,
                                   mixture, next_state, native_raw)

    def flush(self, state, *, return_raw=False):
        require(type(state) is LongMagnitudeState, "Flush requires a long-feature state")
        return self.forward_chunk(state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP)),
                                  state, return_raw=return_raw)

    @classmethod
    def from_parent(cls, parent):
        require(type(parent) is Latency58MagnitudeModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parent.state_dict().values()),
                "Authenticate the CPU magnitude parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        inherited = parent.state_dict()
        model.load_state_dict({**inherited, ADAPTER: model.long_projection.weight,
                               "long_window": model.long_window}, strict=True)
        require(state_sha256({k: v for k, v in model.state_dict().items() if k not in (ADAPTER, "long_window")})
                == state_sha256(inherited), "Long-feature initialization changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance), "long_magnitude_version": VERSION,
                            "long_magnitude_parent_model_state_sha256": state_sha256(inherited),
                            "long_magnitude_updates": 0}
        return model.eval().requires_grad_(False)
