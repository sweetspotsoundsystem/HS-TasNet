"""Add causal attention over 32 received fused-feature frames.

Queries and keys have 64 channels; values have 128. A zero output projection
preserves the selected fusion-refinement parent at initialization. Two public
FP32 caches retain 31 keys and values; no future samples or audio queue is added.
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
from research.direct.latency58_quadrature import cross_component_correction
from research.direct.latency58_fusion_refinement import Latency58FusionRefinementModel
from research.direct.latency58_residual_model import ResidualShareOutput, corrected_estimates
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-fusion-causal-attention-window32-qk64-v128-v1"
WINDOW, KEY, VALUE = 32, 64, 128
ADAPTERS = tuple("temporal_" + name + ".weight" for name in ("query", "key", "value", "output"))


class TemporalAttentionState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor
    attention_keys: torch.Tensor
    attention_values: torch.Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


class Latency58TemporalAttentionModel(Latency58FusionRefinementModel):
    def __init__(self):
        super().__init__()
        self.temporal_query = nn.Linear(2 * EMBED, KEY, bias=False)
        self.temporal_key = nn.Linear(2 * EMBED, KEY, bias=False)
        self.temporal_value = nn.Linear(2 * EMBED, VALUE, bias=False)
        self.temporal_output = nn.Linear(VALUE, 2 * EMBED, bias=False)
        nn.init.zeros_(self.temporal_output.weight)

    @property
    def architecture_metadata(self):
        parent = super().architecture_metadata
        parameters = 2 * EMBED * (2 * KEY + 2 * VALUE)
        return {**parent, "version": VERSION, "state_family": VERSION,
                "state_names": list(TemporalAttentionState._fields),
                "temporal_attention_parameters": parameters,
                "additional_neural_parameters": parent["additional_neural_parameters"] + parameters,
                "temporal_attention_window_frames": WINDOW,
                "temporal_attention_key_channels": KEY, "temporal_attention_value_channels": VALUE,
                "temporal_attention": "refined fused features plus causal softmax(Q K.T / sqrt(64)) V output",
                "temporal_attention_history_samples": (WINDOW - 1) * HOP,
                "additional_audio_buffering_samples": 0, "additional_state_tensors": 2,
                "additional_state_elements_per_stream": (WINDOW - 1) * (KEY + VALUE),
                "additional_fft_transforms": 0, "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        inherited = super().initial_state(batch_size, device=device)
        zeros = inherited.audio_history.new_zeros
        return TemporalAttentionState(*inherited, zeros((batch_size, WINDOW - 1, KEY)),
                                      zeros((batch_size, WINDOW - 1, VALUE)))

    def attention(self, fused, past_keys, past_values, *, tail_only=False):
        queries = self.temporal_query(fused[:, -1:] if tail_only else fused)
        keys = torch.cat((past_keys, self.temporal_key(fused).float()), dim=1)
        values = torch.cat((past_values, self.temporal_value(fused).float()), dim=1)
        if tail_only:
            key_windows = keys[:, -WINDOW:].unsqueeze(1)
            value_windows = values[:, -WINDOW:].unsqueeze(1)
        else:
            key_windows = keys.unfold(1, WINDOW, 1).transpose(-1, -2)
            value_windows = values.unfold(1, WINDOW, 1).transpose(-1, -2)
        # Explicit FP32 attention normalization even during learned BF16 projections.
        with torch.autocast(fused.device.type, enabled=False):
            logits = (queries.float().unsqueeze(-2) * key_windows).sum(-1) * (KEY ** -.5)
            weights = torch.softmax(logits, dim=-1)
            attended = (weights.unsqueeze(-1) * value_windows).sum(-2)
        return (self.temporal_output(attended), keys[:, -(WINDOW - 1):].clone(),
                values[:, -(WINDOW - 1):].clone())

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "Invalid temporal-attention input")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(type(state) is TemporalAttentionState, "Use the distinct temporal-attention state family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP),
                  (audio.shape[0], WINDOW - 1, KEY), (audio.shape[0], WINDOW - 1, VALUE))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "Temporal-attention state geometry changed")
        return state

    def _render_impl(self, audio, state, *, tail_only):
        bf16 = self.training and self.training_precision == "bf16"
        require(not self.training or self.training_precision in ("fp32", "bf16"), "Unknown precision")
        require(not bf16 or audio.device.type == "cuda", "BF16 learned operations require CUDA")
        def learned():
            return torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16)
        batch, _, samples = audio.shape
        joined = torch.cat((state.audio_history, audio), dim=-1)
        feature = torch.fft.rfft(joined.unfold(-1, FEATURE_SAMPLES, HOP) * self.analysis_window,
                                 n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        with learned():
            spec = self.spec_encode(packed)
            convolution = self.conv_encode(joined)
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
        fused = fusion_input + recurrent.float()
        with learned():
            refinement = self.refinement(fused)
            refined = fused + refinement.float()
            correction, attention_keys, attention_values = self.attention(
                refined, state.attention_keys, state.attention_values, tail_only=tail_only)
        if tail_only:
            spec, waveform, refined = (value[:, -1:] for value in (spec, waveform, refined))
            basis, feature = basis[:, :, -1:], feature[:, :, -1:]
        frame_count = spec.shape[1]
        fused_spec, fused_waveform = (refined + correction.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec)
        waveform = self.waveform_norm(fused_waveform + waveform)
        with learned():
            spec_logits = self.to_spec_masks(spec)
            waveform_logits = self.to_waveform_masks(waveform)
            phase = self.phase_coefficients(spec)
        spec_logits = spec_logits.float().reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        carrier = torch.view_as_real(feature)
        masked = carrier.unsqueeze(-1) * masks + cross_component_correction(carrier, phase)
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spec_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)
        waveform_logits = waveform_logits.float().reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        masks = self._residual_source_softmax(waveform_logits)
        source_basis = (basis.transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with learned():
            decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.float().reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
        waveform_audio, wave_tail = self.synthesis.waveform(decoded, state.waveform_tail)
        next_state = TemporalAttentionState(joined[..., -FEATURE_HISTORY:].clone(),
                                    hidden.float() * PUBLIC_FUSION_SCALE, spec_tail, wave_tail,
                                    attention_keys, attention_values)
        if tail_only:
            return next_state
        scales = self.output_source_scales[None, :, None, None]
        native_raw = (spectral + waveform_audio) * scales
        mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        raw, deployed = corrected_estimates(native_raw, mixture, self.fixed_residual_share)
        return ResidualShareOutput(raw, deployed, spectral * scales, waveform_audio * scales,
                                   mixture, next_state, native_raw)

    def flush(self, state, *, return_raw=False):
        require(type(state) is TemporalAttentionState, "Flush requires a temporal-attention state")
        return self.forward_chunk(state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP)),
                                  state, return_raw=return_raw)

    @classmethod
    def from_parent(cls, parent):
        require(type(parent) is Latency58FusionRefinementModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parent.state_dict().values()),
                "Authenticate the CPU fusion-refinement parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(202609141)
            model = cls()
        inherited = parent.state_dict()
        model.load_state_dict({**inherited, **{k: v for k, v in model.state_dict().items() if k in ADAPTERS}}, strict=True)
        require(state_sha256({k: v for k, v in model.state_dict().items() if k not in ADAPTERS})
                == state_sha256(inherited), "Temporal-attention initialization changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance), "temporal_attention_version": VERSION,
                            "temporal_attention_parent_model_state_sha256": state_sha256(inherited),
                            "temporal_attention_updates": 0, "quality_measured": False}
        return model.eval().requires_grad_(False)
