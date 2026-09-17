"""Hop128 BF16 learned kernels with FP32 FFT, synthesis, parameters and states.

Adapted from the accepted hop256 training precision policy. This new geometry
requires its own bounded resource/gradient check. No workload runs on import.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from research.direct.latency58 import (
    Latency58Model, MASK_BINS, Latency58Output, Latency58State, BASIS, CHANNELS, FEATURE_HISTORY,
    FEATURE_SAMPLES, HOP, PUBLIC_FUSION_SCALE, SOURCES, SYNTHESIS_SAMPLES,
)

POLICY = 'latency58-bf16-learned-fp32-synthesis-state-v1'
PARENT_POLICY = 'cropped1024-bf16-learned-fp32-synthesis-state-v1'


class Latency58GPUModel(Latency58Model):
    def __init__(self):
        super().__init__()
        self.training_precision = 'fp32'

    def _render_fp32(self, audio, state):
        if not self.training or self.training_precision == "fp32":
            return super()._render_fp32(audio, state)
        if self.training_precision != "bf16" or audio.device.type != "cuda":
            raise ValueError("BF16 OLA training requires CUDA; no CPU training fallback")
        # Base render() already disabled autocast. Enable only learned dense
        # kernels below; FFT, nonlinear masks and explicit states stay FP32.
        batch, _, samples = audio.shape
        frame_count = samples // HOP
        joined = torch.cat((state.audio_history, audio), dim=-1)
        frames = joined.unfold(-1, FEATURE_SAMPLES, HOP)
        feature = torch.fft.rfft(frames * self.analysis_window, n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spec = self.spec_encode(packed)
            convolution = self.conv_encode(joined)
        to_relu, to_sigmoid = convolution.float().chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            waveform = self.basis_to_embed(basis).transpose(1, 2)
        spec, waveform = spec.float(), waveform.float()
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recurrent, next_hidden = self.fusion_branch(
                fusion_input.to(torch.bfloat16), physical_hidden.to(torch.bfloat16))
        fused_spec, fused_waveform = (fusion_input + recurrent.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec)
        waveform = self.waveform_norm(fused_waveform + waveform)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spec_logits = self.to_spec_masks(spec)
            waveform_logits = self.to_waveform_masks(waveform)
        spec_logits = spec_logits.float().reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        carrier = feature
        masked = torch.view_as_real(carrier).unsqueeze(-1) * masks
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spec_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)
        waveform_logits = waveform_logits.float().reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        masks = self._residual_source_softmax(waveform_logits)
        source_basis = (basis.transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.float().reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
        waveform_audio, wave_tail = self.synthesis.waveform(decoded, state.waveform_tail)
        scales = self.output_source_scales[None, :, None, None]
        raw = (spectral + waveform_audio) * scales
        mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        dbv = raw[:, :3]
        deployed = torch.cat((dbv, mixture.unsqueeze(1) - dbv.sum(dim=1, keepdim=True)), dim=1)
        next_state = Latency58State(joined[..., -FEATURE_HISTORY:].clone(),
                                 next_hidden.float() * PUBLIC_FUSION_SCALE, spec_tail, wave_tail)
        return Latency58Output(raw, deployed, spectral * scales, waveform_audio * scales, mixture, next_state)
