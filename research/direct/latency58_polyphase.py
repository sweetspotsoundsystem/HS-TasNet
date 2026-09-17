"""Untrained recurrent-cadence prototype with unchanged audio lookahead.

Four interleaved hidden trajectories each advance every 512 physical samples.
Every received 128-sample hop still produces an output. No weights are loaded,
no workloads are launched, and no inference-quality claims are made on import.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import NamedTuple

import torch
from torch import nn

from research.direct.latency58 import CHANNELS, FEATURE_HISTORY, HOP
from research.direct.latency58_asymmetric import AsymmetricState
from research.direct.latency58_residual_model import Latency58ResidualModel
from research.direct.latency58_evaluate import model_state_sha256
from research.direct.run_latency58_quality import require


class InterleavedGRU(nn.GRU):
    """Maintain a rotating bank; slot zero is the next hop's trajectory."""

    def __init__(self, *args, phases=4, **kwargs):
        require(type(phases) is int and phases in (1, 2, 4), "Unsupported recurrent cadence")
        super().__init__(*args, **kwargs)
        require(self.batch_first and not self.bidirectional and self.dropout == 0,
                "Require batch-first unidirectional deterministic GRU")
        self.phases = phases

    def forward(self, input, hx=None):
        require(isinstance(input, torch.Tensor) and input.ndim == 3 and input.shape[1] > 0,
                "Require a nonempty dense recurrent sequence")
        batch, frames, _ = input.shape
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch, self.phases * self.hidden_size)
        require(hx.shape == (self.num_layers, batch, self.phases * self.hidden_size), "Wrong hidden bank shape")
        if self.phases == 1:
            return super().forward(input, hx)
        bank = hx.reshape(self.num_layers, batch, self.phases, self.hidden_size)
        next_slots = list(bank.unbind(dim=2))
        output = input.new_empty(batch, frames, self.hidden_size)
        for phase in range(min(frames, self.phases)):
            sequence, hidden = super().forward(input[:, phase::self.phases].contiguous(), bank[:, :, phase].contiguous())
            output[:, phase::self.phases] = sequence
            next_slots[phase] = hidden
        # The next call may start at any hop, including mid-cycle partitions.
        rotation = frames % self.phases
        next_slots = next_slots[rotation:] + next_slots[:rotation]
        return output, torch.stack(next_slots, dim=2).flatten(2)


class InterleavedState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor

    def detached(self):
        return InterleavedState(*(value.detach() for value in self))


class Latency58InterleavedModel(Latency58ResidualModel):
    def __init__(self, *, phases=4):
        super().__init__()
        original = self.fusion_branch
        self.fusion_branch = InterleavedGRU(original.input_size, original.hidden_size,
            num_layers=original.num_layers, bias=original.bias, batch_first=True, dropout=0, phases=phases)
        self.phases = phases
        self.provenance = {"initialization": "uninitialized_interleaved_schema_only"}

    @property
    def architecture_metadata(self):
        version = f"latency58-interleaved-{self.phases}-hop128-v1"
        return {**super().architecture_metadata, "version": version, "state_family": version,
                "state_names": list(InterleavedState._fields), "recurrent_phases": self.phases,
                "recurrent_samples_between_updates_per_trajectory": HOP * self.phases,
                "hidden_state_shape": [2, "batch", self.phases * 1000],
                "hidden_bank_order": "next trajectory first; rotate once per received hop",
                "recurrent_cadence_changed_from_hann_hop128_parent": self.phases != 1,
                "additional_neural_parameters": 0, "additional_audio_buffering_samples": 0,
                "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        state = super().initial_state(batch_size, device=device)
        return InterleavedState(state.audio_history, state.fusion_hidden.repeat(1, 1, self.phases),
                                state.spectral_numerator_tail, state.waveform_tail)

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device,
                "Require matching FP32 stereo complete input hops")
        if state is None:
            state = self.initial_state(audio.shape[0], device=audio.device)
        shapes = ((audio.shape[0], 2, FEATURE_HISTORY), (2, audio.shape[0], self.phases * 1000),
                  (audio.shape[0], 4, 2, HOP), (audio.shape[0], 4, 2, HOP))
        require(isinstance(state, InterleavedState)
                and all(v.shape == shape and v.device == audio.device and v.dtype == torch.float32
                        for v, shape in zip(state, shapes, strict=True)), "Wrong interleaved state family, shape or dtype")
        # The shared render math accesses named fields; only this adapter accepts
        # the larger hidden bank. Public callers must use the distinct family.
        return AsymmetricState(*state)

    def _render_fp32(self, audio, state):
        output = super()._render_fp32(audio, state)
        return replace(output, state=InterleavedState(*output.state))

    def flush(self, state, *, return_raw=False):
        require(isinstance(state, InterleavedState), "Flush requires the interleaved state family")
        zeros = state.audio_history.new_zeros(state.audio_history.shape[0], 2, HOP)
        return self.forward_chunk(zeros, state, return_raw=return_raw)

    @classmethod
    def from_parent(cls, parent, *, phases=4):
        require(type(parent) is Latency58ResidualModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                        for v in parent.state_dict().values()), "Load and authenticate the CPU residual parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls(phases=phases)
        model.load_state_dict(parent.state_dict(), strict=True)
        require(model_state_sha256(model) == model_state_sha256(parent), "Cadence conversion changed neural tensors or buffers")
        model.provenance = {"initialization": "same_neural_tensors_changed_recurrent_cadence",
                            "parent_provenance": copy.deepcopy(parent.provenance),
                            "parent_model_state_sha256": model_state_sha256(parent),
                            "parent_training_updates": parent.provenance["training_updates"],
                            "training_updates": 0, "recurrent_phases": phases,
                            "recurrent_cadence_conversion_training_updates": 0,
                            "parent_output_equivalence_claimed": phases == 1}
        return model.eval().requires_grad_(False)
