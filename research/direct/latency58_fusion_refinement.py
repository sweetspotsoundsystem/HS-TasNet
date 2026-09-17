"""Add a nonlinear residual map to fused quadrature features.

A biasless 1000-to-128-to-1000 map with SiLU and zero final weights starts
from the exact parent. The existing masks already use logits + 4*softmax;
this prototype tests nonlinear feature capacity rather than mask range.
It adds no FFT, history, stream-state tensor or audio buffering.
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
from research.direct.latency58_quadrature import Latency58QuadratureModel, cross_component_correction
from research.direct.latency58_residual_model import ResidualShareOutput, corrected_estimates
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-quadrature-fused-feature-silu-rank128-v1"
RANK = 128
ADAPTERS = ("fusion_refine_reduce.weight", "fusion_refine_expand.weight")


class FusionRefinementState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


class Latency58FusionRefinementModel(Latency58QuadratureModel):
    def __init__(self):
        super().__init__()
        self.fusion_refine_reduce = nn.Linear(2 * EMBED, RANK, bias=False)
        self.fusion_refine_expand = nn.Linear(RANK, 2 * EMBED, bias=False)
        nn.init.zeros_(self.fusion_refine_expand.weight)

    @property
    def architecture_metadata(self):
        parent = super().architecture_metadata
        parameters = 4 * EMBED * RANK
        return {**parent, "version": VERSION, "state_family": VERSION,
                "state_names": list(FusionRefinementState._fields), "fusion_refinement_rank": RANK,
                "fusion_refinement_parameters": parameters,
                "additional_neural_parameters": parent["additional_neural_parameters"] + parameters,
                "fusion_refinement": "fused + expand(SiLU(reduce(fused))) before branch normalization",
                "fusion_refinement_input_features": 2 * EMBED,
                "additional_audio_buffering_samples": 0, "additional_state_tensors": 0,
                "additional_fft_transforms": 0, "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        return FusionRefinementState(*super().initial_state(batch_size, device=device))

    def refinement(self, fused):
        return self.fusion_refine_expand(F.silu(self.fusion_refine_reduce(fused)))

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "Invalid fusion-refinement input")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(type(state) is FusionRefinementState, "Use the distinct fusion-refinement state family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "Fusion-refinement state geometry changed")
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
        if tail_only:
            spec, waveform, fusion_input, recurrent = (value[:, -1:] for value in
                                                       (spec, waveform, fusion_input, recurrent))
            basis, feature = basis[:, :, -1:], feature[:, :, -1:]
        frame_count = spec.shape[1]
        fused = fusion_input + recurrent.float()
        with learned():
            refinement = self.refinement(fused)
        fused_spec, fused_waveform = (fused + refinement.float()).chunk(2, dim=-1)
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
        next_state = FusionRefinementState(joined[..., -FEATURE_HISTORY:].clone(),
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
        require(type(state) is FusionRefinementState, "Flush requires a fusion-refinement state")
        return self.forward_chunk(state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP)),
                                  state, return_raw=return_raw)

    @classmethod
    def from_parent(cls, parent):
        require(type(parent) is Latency58QuadratureModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parent.state_dict().values()),
                "Authenticate the CPU quadrature parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(202609140)
            model = cls()
        inherited = parent.state_dict()
        model.load_state_dict({**inherited, **{k: v for k, v in model.state_dict().items() if k in ADAPTERS}}, strict=True)
        require(state_sha256({k: v for k, v in model.state_dict().items() if k not in ADAPTERS})
                == state_sha256(inherited), "Feature-refinement initialization changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance), "fusion_refinement_version": VERSION,
                            "fusion_refinement_parent_model_state_sha256": state_sha256(inherited),
                            "fusion_refinement_updates": 0, "quality_measured": False}
        return model.eval().requires_grad_(False)
