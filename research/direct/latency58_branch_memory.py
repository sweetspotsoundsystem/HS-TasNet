"""Prototype separate causal branch memories with zero output projections.

The two single-layer 500-channel GRUs read normalized, refined branch features
before attention. Their projected outputs are added after the existing branch
normalization. Zero projections retain the trained parent's audio and six old
states; two extra recurrent states add history without audio lookahead.
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
from research.direct.latency58_temporal_attention import Latency58TemporalAttentionModel, WINDOW, KEY, VALUE
from research.direct.latency58_residual_model import ResidualShareOutput, corrected_estimates
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-attention-private-branch-gru500-zero-projections-v1"
ADAPTERS = tuple(branch + "." + name for branch in ("spec_memory", "waveform_memory")
                 for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")) + (
                     "spec_memory_output.weight", "waveform_memory_output.weight")


class BranchMemoryState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor
    attention_keys: torch.Tensor
    attention_values: torch.Tensor
    spec_memory_hidden: torch.Tensor
    waveform_memory_hidden: torch.Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


class Latency58BranchMemoryModel(Latency58TemporalAttentionModel):
    def __init__(self):
        super().__init__()
        self.spec_memory = nn.GRU(EMBED, EMBED, num_layers=1, batch_first=True)
        self.waveform_memory = nn.GRU(EMBED, EMBED, num_layers=1, batch_first=True)
        self.spec_memory_output = nn.Linear(EMBED, EMBED, bias=False)
        self.waveform_memory_output = nn.Linear(EMBED, EMBED, bias=False)
        nn.init.zeros_(self.spec_memory_output.weight)
        nn.init.zeros_(self.waveform_memory_output.weight)

    @property
    def architecture_metadata(self):
        parent = super().architecture_metadata
        parameters = sum(p.numel() for name, p in self.named_parameters() if name in ADAPTERS)
        return {**parent, "version": VERSION, "state_family": VERSION,
                "state_names": list(BranchMemoryState._fields),
                "branch_memory_parameters": parameters, "branch_memory_channels": EMBED,
                "branch_memory_layers": 1,
                "branch_memory_input": "normalized refined branch features before attention",
                "branch_memory_output": "zero-initialized linear residual after branch normalization",
                "additional_neural_parameters": parent["additional_neural_parameters"] + parameters,
                "additional_audio_buffering_samples": 0, "additional_state_tensors": 4,
                "branch_memory_added_state_tensors": 2,
                "branch_memory_added_state_elements_per_stream": 2 * EMBED,
                "additional_state_elements_per_stream": (WINDOW - 1) * (KEY + VALUE) + 2 * EMBED,
                "additional_fft_transforms": 0, "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        inherited = super().initial_state(batch_size, device=device)
        zeros = inherited.audio_history.new_zeros
        return BranchMemoryState(*inherited, zeros((1, batch_size, EMBED)), zeros((1, batch_size, EMBED)))

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "Invalid branch-memory input")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(type(state) is BranchMemoryState, "Use the distinct branch-memory state family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP),
                  (audio.shape[0], WINDOW - 1, KEY), (audio.shape[0], WINDOW - 1, VALUE),
                  (1, audio.shape[0], EMBED), (1, audio.shape[0], EMBED))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "Branch-memory state geometry changed")
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
        # Warmup still advances both new memories through every received frame.
        # Their inputs do not depend on attention, preserving the parent's exact
        # final-query shortcut and its existing six-state arithmetic.
        private_spec, private_waveform = refined.chunk(2, dim=-1)
        private_spec = self.spec_norm(private_spec + spec)
        private_waveform = self.waveform_norm(private_waveform + waveform)
        spec_initial = state.spec_memory_hidden / PUBLIC_FUSION_SCALE
        waveform_initial = state.waveform_memory_hidden / PUBLIC_FUSION_SCALE
        with learned():
            spec_memory, spec_hidden = self.spec_memory(
                private_spec.to(torch.bfloat16) if bf16 else private_spec,
                spec_initial.to(torch.bfloat16) if bf16 else spec_initial)
            waveform_memory, waveform_hidden = self.waveform_memory(
                private_waveform.to(torch.bfloat16) if bf16 else private_waveform,
                waveform_initial.to(torch.bfloat16) if bf16 else waveform_initial)
            spec_correction = self.spec_memory_output(spec_memory)
            waveform_correction = self.waveform_memory_output(waveform_memory)
        if tail_only:
            spec_correction, waveform_correction = (value[:, -1:] for value in (spec_correction, waveform_correction))
            spec, waveform, refined = (value[:, -1:] for value in (spec, waveform, refined))
            basis, feature = basis[:, :, -1:], feature[:, :, -1:]
        frame_count = spec.shape[1]
        fused_spec, fused_waveform = (refined + correction.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec) + spec_correction.float()
        waveform = self.waveform_norm(fused_waveform + waveform) + waveform_correction.float()
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
        next_state = BranchMemoryState(joined[..., -FEATURE_HISTORY:].clone(),
                                    hidden.float() * PUBLIC_FUSION_SCALE, spec_tail, wave_tail,
                                    attention_keys, attention_values,
                                    spec_hidden.float() * PUBLIC_FUSION_SCALE,
                                    waveform_hidden.float() * PUBLIC_FUSION_SCALE)
        if tail_only:
            return next_state
        scales = self.output_source_scales[None, :, None, None]
        native_raw = (spectral + waveform_audio) * scales
        mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        raw, deployed = corrected_estimates(native_raw, mixture, self.fixed_residual_share)
        return ResidualShareOutput(raw, deployed, spectral * scales, waveform_audio * scales,
                                   mixture, next_state, native_raw)

    def flush(self, state, *, return_raw=False):
        require(type(state) is BranchMemoryState, "Flush requires a branch-memory state")
        return self.forward_chunk(state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP)),
                                  state, return_raw=return_raw)

    @classmethod
    def from_parent(cls, parent):
        require(type(parent) is Latency58TemporalAttentionModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parent.state_dict().values()),
                "Authenticate the CPU attention parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(20261021)
            model = cls()
        inherited = parent.state_dict()
        model.load_state_dict({**inherited, **{k: v for k, v in model.state_dict().items() if k in ADAPTERS}}, strict=True)
        require(state_sha256({k: v for k, v in model.state_dict().items() if k not in ADAPTERS})
                == state_sha256(inherited), "Branch-memory initialization changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance), "branch_memory_version": VERSION,
                            "branch_memory_parent_model_state_sha256": state_sha256(inherited),
                            "branch_memory_updates": 0, "quality_measured": False}
        return model.eval().requires_grad_(False)
