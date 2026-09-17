"""Training-only warmup that decodes just the final frame's two saved tails.

All encoder and recurrent frames are retained. Earlier decoder outputs cannot
affect the recurrent state or the final overlap tails, so they are unnecessary
when the caller discards warmup audio. This helper does not alter inference.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from research.direct.latency58 import (
    BASIS, CHANNELS, FEATURE_HISTORY, FEATURE_SAMPLES, HOP, MASK_BINS,
    PUBLIC_FUSION_SCALE, SOURCES, SYNTHESIS_SAMPLES, require,
)
from research.direct.latency58_magnitude import Latency58MagnitudeModel

VERSION = "latency58-magnitude-final-tail-state-warmup-v1"


@torch.no_grad()
def warm_state(model, audio, state=None):
    """Advance the native states without producing discarded warmup audio."""
    require(type(model) is Latency58MagnitudeModel, "Require the reviewed magnitude model")
    state = model._validate(audio, state)
    bf16 = model.training and model.training_precision == "bf16"
    require(not model.training or model.training_precision in ("fp32", "bf16"),
            "Unknown training precision")
    require(not bf16 or audio.device.type == "cuda", "BF16 warmup requires CUDA")
    with torch.autocast(audio.device.type, enabled=False):
        batch = audio.shape[0]
        joined = torch.cat((state.audio_history, audio), dim=-1)
        frames = joined.unfold(-1, FEATURE_SAMPLES, HOP)
        feature = torch.fft.rfft(frames * model.analysis_window, n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        def learned():
            return torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16)
        with learned():
            spec = model.spec_encode(packed)
            convolution = model.conv_encode(joined)
        to_relu, to_sigmoid = convolution.float().chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        with learned():
            waveform = model.basis_to_embed(basis).transpose(1, 2)
        spec, waveform = spec.float(), waveform.float()
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        with learned():
            recurrent, hidden = model.fusion_branch(
                fusion_input.to(torch.bfloat16) if bf16 else fusion_input,
                physical_hidden.to(torch.bfloat16) if bf16 else physical_hidden)

        # A final overlap tail depends only on the final decoder frame. Keep
        # the full recurrent execution above, including both native layers.
        last_spec, last_waveform = (fusion_input[:, -1:] + recurrent.float()[:, -1:]).chunk(2, dim=-1)
        last_spec = model.spec_norm(last_spec + spec[:, -1:])
        last_waveform = model.waveform_norm(last_waveform + waveform[:, -1:])
        with learned():
            spec_logits = model.to_spec_masks(last_spec)
            waveform_logits = model.to_waveform_masks(last_waveform)
        spec_logits = spec_logits.float().reshape(batch, 1, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = model._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        masked = torch.view_as_real(feature[:, :, -1:]).unsqueeze(-1) * masks
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral_frames = torch.fft.irfft(spectrum, n=FEATURE_SAMPLES, dim=-1)
        spectral_tail = spectral_frames[..., 0, -HOP:] * model.synthesis.spectral_window[-HOP:]

        waveform_logits = waveform_logits.float().reshape(batch, 1, SOURCES, BASIS).transpose(-1, -2)
        masks = model._residual_source_softmax(waveform_logits)
        source_basis = (basis[:, :, -1:].transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with learned():
            decoded = F.linear(source_basis, model.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.float().reshape(batch, SOURCES, 1, CHANNELS, SYNTHESIS_SAMPLES)
        waveform_tail = decoded[:, :, 0, :, -HOP:] * model.synthesis.window[-HOP:]
        return type(state)(joined[..., -FEATURE_HISTORY:].clone(),
                           hidden.float() * PUBLIC_FUSION_SCALE,
                           spectral_tail.clone(), waveform_tail.clone())
