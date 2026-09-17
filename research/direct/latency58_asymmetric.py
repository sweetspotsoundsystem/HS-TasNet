"""Unqualified hop128 model with a matched asymmetric spectral window pair.

The feature encoder is transformed once when copying the Hann parent. The
single analysis FFT feeds both features and masks. Waveform synthesis keeps
its Hann256 window. Neither weights nor experiments are loaded on import.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import NamedTuple

import torch

from research.direct.latency58 import (
    CHANNELS, CROP_START, FEATURE_SAMPLES, HOP, MASK_BINS, SYNTHESIS_SAMPLES,
    Cropped256Synthesis, Latency58Model, Latency58State, VERSION as HANN_VERSION,
    overlap_frames, require,
)
from research.direct.latency58_encoder_window import asymmetric_windows, convert_encoder_weight
from research.direct.latency58_gpu import Latency58GPUModel

VERSION = "cropped1024-asymmetric256-hop128-v1"
PRECISION_POLICY = "latency58-asymmetric-bf16-learned-fp32-synthesis-state-v1"


class AsymmetricState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor

    def detached(self):
        return AsymmetricState(*(value.detach() for value in self))


class AsymmetricSynthesis(Cropped256Synthesis):
    def __init__(self, analysis_window, spectral_window):
        torch.nn.Module.__init__(self)
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and analysis_window.dtype in (torch.float32, torch.float64), "Expected FP32/FP64 analysis")
        waveform = torch.hann_window(SYNTHESIS_SAMPLES, periodic=True,
                                     dtype=analysis_window.dtype, device=analysis_window.device)
        self.register_buffer("window", waveform)
        self.register_buffer("spectral_window", spectral_window.clone())
        self.register_buffer("spectral_denominator", torch.zeros_like(waveform[:HOP]))
        self.register_buffer("waveform_window_sum", waveform[:HOP] + waveform[HOP:])
        self.rebuild_denominator(analysis_window)

    def rebuild_denominator(self, analysis_window):
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and self.spectral_window.shape == (SYNTHESIS_SAMPLES,)
                and all(t.dtype == self.window.dtype and t.device == self.window.device
                        and bool(torch.isfinite(t).all()) for t in (analysis_window, self.spectral_window)),
                "Analysis and spectral synthesis window contracts differ")
        product = analysis_window[CROP_START:] * self.spectral_window
        denominator = product[:HOP] + product[HOP:]
        require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
                "Spectral overlap divisor must be finite and positive")
        with torch.no_grad():
            self.spectral_denominator.copy_(denominator)

    def spectral(self, spectrum, previous_tail):
        require(spectrum.ndim >= 2 and spectrum.is_complex() and spectrum.shape[-1] == MASK_BINS
                and spectrum.real.dtype == self.window.dtype and spectrum.device == self.window.device,
                "Expected matching complex [...,F,513] spectrum")
        frames = torch.fft.irfft(spectrum, n=FEATURE_SAMPLES, dim=-1)[..., CROP_START:]
        numerator, tail = overlap_frames(frames * self.spectral_window, previous_tail)
        return numerator / self.spectral_denominator.repeat(spectrum.shape[-2]), tail


class Latency58AsymmetricModel(Latency58GPUModel):
    def __init__(self):
        super().__init__()
        analysis, spectral = asymmetric_windows()
        self.analysis_window.copy_(analysis)
        self.synthesis = AsymmetricSynthesis(self.analysis_window, spectral)
        self.provenance = {"version": VERSION, "initialization": "uninitialized_schema_only"}

    @property
    def architecture_metadata(self):
        value = super().architecture_metadata
        value.update(version=VERSION, state_family=VERSION, state_names=list(AsymmetricState._fields),
                     feature_window="Wang2021 K1024 M128 d0 asymmetric analysis",
                     spectral_synthesis_window="matched pair with Hann256 analysis-synthesis product",
                     waveform_synthesis_window="unchanged periodic Hann256",
                     carrier_forward_fft_reused_from_features=True,
                     spectral_denominator_rule="sum of the two analysis-synthesis overlap products",
                     encoder_initialization="FP64 effective-kernel window transfer, rounded once to FP32",
                     encoder_transfer_bf16_equivalence_claimed=False,
                     recurrent_cadence_changed_from_hann_hop128_parent=False)
        return value

    def initial_state(self, batch_size, *, device=None):
        return AsymmetricState(*super().initial_state(batch_size, device=device))

    def _validate(self, audio, state):
        if state is None:
            return super()._validate(audio, None)
        require(isinstance(state, AsymmetricState), "Use the distinct AsymmetricState family")
        return AsymmetricState(*super()._validate(audio, Latency58State(*state)))

    def _render_fp32(self, audio, state):
        output = super()._render_fp32(audio, state)
        return replace(output, state=AsymmetricState(*output.state))

    def flush(self, state, *, return_raw=False):
        require(isinstance(state, AsymmetricState), "Flush requires AsymmetricState")
        zeros = state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP))
        return self.forward_chunk(zeros, state, return_raw=return_raw)

    @classmethod
    def from_accepted(cls, *args, **kwargs):
        raise ValueError("Authenticate the Hann parent, then call from_hann_model explicitly")

    @classmethod
    def from_hann_model(cls, parent):
        require(type(parent) is Latency58Model and parent.architecture_metadata["version"] == HANN_VERSION,
                "Expected the explicitly loaded original Hann hop128 model")
        require(all(t.device.type == "cpu" and t.dtype == torch.float32 and bool(torch.isfinite(t).all())
                    for t in parent.state_dict().values()), "Transfer requires finite CPU FP32 tensors")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        source, destination = parent.state_dict(), model.state_dict()
        require(set(destination) == set(source) | {"synthesis.spectral_window"}, "Transfer inventory differs")
        with torch.no_grad():
            for name, tensor in source.items():
                require(destination[name].shape == tensor.shape, "Transfer shape differs: " + name)
                if name not in ("analysis_window", "synthesis.spectral_denominator", "spec_encode.weight"):
                    destination[name].copy_(tensor)
            model.spec_encode.weight.copy_(convert_encoder_weight(
                parent.spec_encode.weight, parent.analysis_window, model.analysis_window))
        model.synthesis.rebuild_denominator(model.analysis_window)
        model.provenance = {"version": VERSION, "initialization": "hann_hop128_encoder_transfer_asymmetric_pair",
                            "parent_provenance": copy.deepcopy(parent.provenance),
                            "asymmetric_training_updates": 0, "equivalence_claimed": False}
        return model
