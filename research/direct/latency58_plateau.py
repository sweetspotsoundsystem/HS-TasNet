"""Unqualified hop128 hypothesis: separate Hann features and plateau carrier.

The learned layers and waveform branch can be copied exactly from a hop128
Hann model. The spectral carrier and its overlap divisor change together.
No checkpoint is loaded, experiment run or deployment artifact made on import.
"""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch.nn import functional as F

from research.direct.latency58 import (
    BASIS, CHANNELS, FEATURE_HISTORY, FEATURE_SAMPLES, HOP, MASK_BINS,
    PUBLIC_FUSION_SCALE, SOURCES, SYNTHESIS_SAMPLES, VERSION as HANN_VERSION,
    Latency58Model, Latency58Output, Latency58State, require,
)

VERSION = "cropped1024-plateau256-hop128-v1"
PRECISION_POLICY = "latency58-plateau-bf16-learned-fp32-fft-state-v1"


class PlateauState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor

    def detached(self):
        return PlateauState(*(value.detach() for value in self))


def carrier_window(*, dtype=torch.float32, device="cpu"):
    require(dtype in (torch.float32, torch.float64), "Carrier window requires FP32 or FP64")
    index = torch.arange(FEATURE_SAMPLES, dtype=dtype, device=device)
    rise = 0.5 * (1.0 - torch.cos(torch.pi * index / 512.0))
    fall = 0.5 * (1.0 + torch.cos(torch.pi * (index - 896.0) / 128.0))
    return torch.where(index < 512, rise, torch.where(index < 896, torch.ones_like(index), fall))


class Latency58PlateauModel(Latency58Model):
    def __init__(self):
        super().__init__()
        self.register_buffer("carrier_window", carrier_window())
        self.synthesis.rebuild_denominator(self.carrier_window)
        self.training_precision = "fp32"
        self.provenance = {"version": VERSION, "initialization": "uninitialized_schema_only"}

    @property
    def architecture_metadata(self):
        value = super().architecture_metadata
        value.update(version=VERSION, state_family=VERSION, state_names=list(PlateauState._fields),
                     feature_window="unchanged periodic Hann1024",
                     carrier_window="half-cosine rise0:512, plateau512:896, half-cosine fall896:1024",
                     carrier_forward_fft_reused_from_features=False,
                     spectral_denominator_rule="carrier[768+p]*Hann256[p]+carrier[896+p]*Hann256[128+p]",
                     recurrent_cadence_changed_from_hann_hop128_parent=False)
        return value

    def initial_state(self, batch_size, *, device=None):
        return PlateauState(*super().initial_state(batch_size, device=device))

    def _validate(self, audio, state):
        if state is None:
            return super()._validate(audio, None)
        require(isinstance(state, PlateauState), "Use the distinct PlateauState family")
        checked = super()._validate(audio, Latency58State(*state))
        return PlateauState(*checked)

    def flush(self, state, *, return_raw=False):
        require(isinstance(state, PlateauState), "Flush requires PlateauState")
        zeros = state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP))
        return self.forward_chunk(zeros, state, return_raw=return_raw)

    @classmethod
    def from_accepted(cls, *args, **kwargs):
        raise ValueError("Load and authenticate the Hann parent, then call from_hann_model explicitly")

    @classmethod
    def from_hann_model(cls, parent):
        require(type(parent) is Latency58Model and parent.architecture_metadata["version"] == HANN_VERSION,
                "Expected an explicitly loaded original Hann hop128 model")
        require(all(value.device.type == "cpu" and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()) for value in parent.state_dict().values()),
                "Transfer requires finite CPU FP32 parent tensors")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        source, destination = parent.state_dict(), model.state_dict()
        require(set(destination) == set(source) | {"carrier_window"}, "Transfer inventory differs")
        with torch.no_grad():
            for name, tensor in source.items():
                require(destination[name].shape == tensor.shape, "Transfer shape differs: " + name)
                if name != "synthesis.spectral_denominator":
                    destination[name].copy_(tensor)
        model.synthesis.rebuild_denominator(model.carrier_window)
        model.provenance = {"version": VERSION, "initialization": "hann_hop128_weights_separate_plateau_carrier",
                            "parent_provenance": copy.deepcopy(parent.provenance),
                            "plateau_training_updates": 0, "equivalence_claimed": False}
        return model

    def _render_fp32(self, audio, state):
        bf16 = self.training and self.training_precision == "bf16"
        require(self.training_precision in ("fp32", "bf16") and (not bf16 or audio.device.type == "cuda"),
                "BF16 training requires CUDA; no CPU training fallback")
        batch, _, samples = audio.shape
        frame_count = samples // HOP
        joined = torch.cat((state.audio_history, audio), dim=-1)
        frames = joined.unfold(-1, FEATURE_SAMPLES, HOP)
        feature = torch.fft.rfft(frames * self.analysis_window, n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
            spec = self.spec_encode(packed)
            convolution = self.conv_encode(joined)
        to_relu, to_sigmoid = convolution.float().chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
            waveform = self.basis_to_embed(basis).transpose(1, 2)
        spec, waveform = spec.float(), waveform.float()
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
            recurrent, hidden = self.fusion_branch(
                fusion_input.to(torch.bfloat16) if bf16 else fusion_input,
                physical_hidden.to(torch.bfloat16) if bf16 else physical_hidden)
        fused_spec, fused_waveform = (fusion_input + recurrent.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec)
        waveform = self.waveform_norm(fused_waveform + waveform)
        with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
            spec_logits = self.to_spec_masks(spec)
            wave_logits = self.to_waveform_masks(waveform)
        spec_logits = spec_logits.float().reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        carrier = torch.fft.rfft(frames * self.carrier_window, n=FEATURE_SAMPLES, dim=-1)
        masked = torch.view_as_real(carrier).unsqueeze(-1) * masks
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spec_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)
        wave_logits = wave_logits.float().reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        masks = self._residual_source_softmax(wave_logits)
        source_basis = (basis.transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
            decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t())
        decoded = decoded.float().reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES)
        waveform_audio, wave_tail = self.synthesis.waveform(decoded.permute(0, 1, 3, 2, 4), state.waveform_tail)
        scales = self.output_source_scales[None, :, None, None]
        raw = (spectral + waveform_audio) * scales
        delayed = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        retained = raw[:, :3]
        deployed = torch.cat((retained, delayed.unsqueeze(1) - retained.sum(dim=1, keepdim=True)), dim=1)
        next_state = PlateauState(joined[..., -FEATURE_HISTORY:].clone(), hidden.float() * PUBLIC_FUSION_SCALE,
                                  spec_tail, wave_tail)
        return Latency58Output(raw, deployed, spectral * scales, waveform_audio * scales, delayed, next_state)
