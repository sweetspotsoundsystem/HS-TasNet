"""A zero-initialized quadrature correction for the short magnitude model.

The existing spectral estimate multiplies real and imaginary carrier components
separately. An additional low-rank head predicts real coefficients multiplying
i times that same carrier, permitting cross-component corrections. Coefficients
are centered across sources and fixed to zero at DC/Nyquist. No future input,
additional FFT, audio history or recurrent tensor is introduced.
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

VERSION = "latency58-short-magnitude-centered-quadrature-rank64-v1"
RANK = 64
ADAPTERS = ("phase_reduce.weight", "phase_expand.weight")


class QuadratureState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


def cross_component_correction(carrier_ri, coefficients):
    """Return i*carrier times each real coefficient in explicit RI coordinates."""
    require(carrier_ri.shape[:-1] == coefficients.shape[:-1] and carrier_ri.shape[-1] == 2
            and coefficients.shape[-1] == SOURCES and carrier_ri.dtype == coefficients.dtype == torch.float32,
            "Require matching FP32 carrier and four-source coefficients")
    quadrature = torch.stack((-carrier_ri[..., 1], carrier_ri[..., 0]), -1)
    return quadrature.unsqueeze(-1) * coefficients.unsqueeze(-2)


class Latency58QuadratureModel(Latency58MagnitudeModel):
    def __init__(self):
        super().__init__()
        self.phase_reduce = nn.Linear(EMBED, RANK, bias=False)
        self.phase_expand = nn.Linear(RANK, CHANNELS * (MASK_BINS - 2) * SOURCES, bias=False)
        nn.init.zeros_(self.phase_expand.weight)

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "version": VERSION, "state_family": VERSION,
                "state_names": list(QuadratureState._fields), "phase_head_rank": RANK,
                "phase_head_parameters": RANK * (EMBED + CHANNELS * (MASK_BINS - 2) * SOURCES),
                "additional_neural_parameters": 513000 + RANK * (EMBED + CHANNELS * (MASK_BINS - 2) * SOURCES),
                "phase_correction": "i*carrier times source-centered real coefficients at interior bins",
                "phase_correction_coordinate": "before inherited source calibration and discrepancy correction",
                "phase_endpoint_coefficients": "exact zero at DC and Nyquist",
                "additional_audio_buffering_samples": 0, "additional_state_tensors": 0,
                "additional_fft_transforms": 0, "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        return QuadratureState(*super().initial_state(batch_size, device=device))

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "Invalid quadrature input")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(type(state) is QuadratureState, "Use the distinct quadrature state family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "Quadrature state geometry changed")
        return state

    def phase_coefficients(self, features):
        projected = self.phase_expand(self.phase_reduce(features))
        with torch.autocast(features.device.type, enabled=False):
            values = projected.float().reshape(features.shape[0], features.shape[1], CHANNELS, MASK_BINS - 2, SOURCES)
            # This centering is before inherited source calibration; it is not a
            # claim that the calibrated four-source correction sums to exactly zero.
            values = values - values.mean(-1, keepdim=True)
            return F.pad(values, (0, 0, 1, 1)).permute(0, 2, 1, 3, 4)

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
        fused_spec, fused_waveform = (fusion_input + recurrent.float()).chunk(2, dim=-1)
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
        next_state = QuadratureState(joined[..., -FEATURE_HISTORY:].clone(),
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
        require(type(state) is QuadratureState, "Flush requires a quadrature state")
        return self.forward_chunk(state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP)),
                                  state, return_raw=return_raw)

    @classmethod
    def from_parent(cls, parent):
        require(type(parent) is Latency58MagnitudeModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parent.state_dict().values()),
                "Authenticate the CPU magnitude parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(202609125)
            model = cls()
        inherited = parent.state_dict()
        model.load_state_dict({**inherited, **{k: v for k, v in model.state_dict().items() if k in ADAPTERS}}, strict=True)
        require(state_sha256({k: v for k, v in model.state_dict().items() if k not in ADAPTERS})
                == state_sha256(inherited), "Quadrature initialization changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance), "quadrature_version": VERSION,
                            "quadrature_parent_model_state_sha256": state_sha256(inherited),
                            "quadrature_updates": 0, "quality_measured": False}
        return model.eval().requires_grad_(False)
