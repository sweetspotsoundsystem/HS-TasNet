"""A causal output correction conditioned on the existing recurrent features."""
from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache

import torch
from torch import nn

from research.direct.latency58 import PUBLIC_FUSION_SCALE
from research.direct.latency58_residual_model import Latency58ResidualModel
from research.direct.run_latency58_quality import require, sha
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-recurrent-dynamic-source-mixer-v1"
HOP, WINDOW, FRAMES = 128, 44100, 690


def make_head():
    head = nn.Linear(1000, 12)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def mixing_scale():
    scale = torch.full((3, 4), .25)
    scale[torch.arange(3), torch.arange(3)] = 1.
    return scale


def mixing_ramp():
    return .5 - .5 * torch.cos(torch.linspace(0, torch.pi, HOP))


def coefficients(head, features, scale):
    return torch.tanh(head(features)).unflatten(-1, (3, 4)) * scale


def apply_correction(estimates, mixture, deltas, ramp):
    """deltas contains the preceding frame followed by every current frame."""
    frame_count = estimates.shape[-1] // HOP
    frames = estimates.unflatten(-1, (frame_count, HOP))
    before = torch.einsum("bfos,bscft->bocft", deltas[:, :-1], frames)
    after = torch.einsum("bfos,bscft->bocft", deltas[:, 1:], frames)
    correction = (before * (1 - ramp) + after * ramp).flatten(-2)
    retained = estimates[:, :3] + correction
    other = mixture - ((retained[:, 0] + retained[:, 1]) + retained[:, 2])
    return torch.cat((retained, other.unsqueeze(1)), 1)


class Latency58DynamicMixer(Latency58ResidualModel):
    def __init__(self):
        super().__init__()
        self.dynamic_mixer_head = make_head()
        self.register_buffer("dynamic_mixer_scale", mixing_scale())
        self.register_buffer("dynamic_mixer_ramp", mixing_ramp())

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "version": VERSION, "output_policy_version": VERSION,
                "dynamic_mixer_features": "last-layer GRU output from the current and previous received hops",
                "dynamic_mixer_coefficients": "identity plus bounded tanh linear projection; diagonal radius 1, off-diagonal radius 0.25",
                "dynamic_mixer_interpolation": "cosine ramp from previous to current coefficients over each output hop",
                "dynamic_mixer_previous_features": "recovered from existing incoming fusion_hidden state",
                "raw_output_semantics": "complete calibrated DBVO with residual Other",
                "component_semantics": "native spectral and waveform components precede dynamic source correction",
                "additional_learned_parameters": 12012, "extra_audio_buffering_samples": 0,
                "extra_stream_state_tensors": 0, "native_host_qualified": False}

    def _render_fp32(self, audio, state):
        # The hook exposes the sequence already computed by the inherited GRU.
        # It is local to this render and removed even if the parent raises.
        # The literal export graph obtains the same feature from next_hidden.
        captured = []
        handle = self.fusion_branch.register_forward_hook(lambda module, inputs, output: captured.append(output[0]))
        try:
            native = super()._render_fp32(audio, state)
        finally:
            handle.remove()
        require(len(captured) == 1, "Expected one inherited GRU sequence")
        previous = (state.fusion_hidden[-1] / PUBLIC_FUSION_SCALE).unsqueeze(1)
        features = torch.cat((previous, captured[0].float()), 1)
        deltas = coefficients(self.dynamic_mixer_head, features, self.dynamic_mixer_scale)
        output = apply_correction(native.deployed, native.delayed_mixture, deltas, self.dynamic_mixer_ramp)
        return replace(native, raw=output, deployed=output)

    @classmethod
    def from_parent(cls, parent, head_state=None):
        require(type(parent) is Latency58ResidualModel, "Authenticate the residual parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        inherited = parent.state_dict()
        values = model.state_dict()
        values.update(inherited)
        if head_state is not None:
            require(set(head_state) == {"weight", "bias"}, "Unexpected dynamic head inventory")
            values.update({"dynamic_mixer_head." + name: value for name, value in head_state.items()})
        model.load_state_dict(values, strict=True)
        require(state_sha256({name: value for name, value in model.state_dict().items() if name in inherited})
                == state_sha256(inherited), "Dynamic correction changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance), "dynamic_mixer_version": VERSION,
                            "dynamic_mixer_parent_state_sha256": state_sha256(inherited)}
        return model.eval().requires_grad_(False)


def load_model(binding):
    from research.direct.latency58_direct_sdr_checkpoint import load_model as load_parent
    require(sha(binding["path"]) == binding["sha256"], "Dynamic correction checkpoint changed")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    require(payload["schema"] == VERSION and not payload["fit_used_validation_or_test_audio"], "Unexpected dynamic checkpoint")
    parent, _ = load_parent(payload["parent_checkpoint"])
    model = Latency58DynamicMixer.from_parent(parent, payload["head"])
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and state_sha256(model.state_dict()) == payload["model_state_sha256"]
            and model.architecture_metadata == payload["architecture"], "Restored dynamic model differs")
    model.provenance = payload["provenance"]
    return model, payload


@lru_cache(maxsize=1)
def segment_layout():
    frames, windows, masks = [], [], []
    for window in range(2):
        start, end = window * WINDOW, (window + 1) * WINDOW
        for frame in range(start // HOP, (end + HOP - 1) // HOP):
            indices = frame * HOP + torch.arange(HOP)
            frames.append(frame)
            windows.append(window)
            masks.append(((indices >= start) & (indices < end)).double())
    return torch.tensor(frames), torch.tensor(windows), torch.stack(masks)


def frame_covariances(estimates, targets, mixture):
    require(estimates.shape == targets.shape and estimates.shape[1:] == (4, 2, FRAMES * HOP)
            and mixture.shape == (estimates.shape[0], 2, FRAMES * HOP), "Use the fixed scored suffix")
    indices, windows, masks = segment_layout()
    y = estimates.unflatten(-1, (FRAMES, HOP)).permute(0, 3, 1, 2, 4)[:, indices].double()
    t = targets.unflatten(-1, (FRAMES, HOP)).permute(0, 3, 1, 2, 4)[:, indices].double()
    m = mixture.unflatten(-1, (FRAMES, HOP)).permute(0, 2, 1, 3)[:, indices].double()
    ramp = mixing_ramp().double()
    expanded = torch.cat((y * (1 - ramp), y * ramp), 2).flatten(-2)
    mask = masks[:, None, :].expand(-1, 2, -1).flatten(-2)
    weighted = expanded * mask[None, :, None]
    truth = t.flatten(-2)
    return {"gram": weighted @ expanded.transpose(-1, -2), "cross": weighted @ truth.transpose(-1, -2),
            "truth_energy": (truth.square() * mask[None, :, None]).sum(-1),
            "mixture_energy": (m.square() * masks[None, :, None]).sum((-1, -2))}


def loss_from_frames(head, features, moments):
    indices, windows, _ = segment_layout()
    delta = coefficients(head, features, mixing_scale())
    identity = torch.eye(4, dtype=delta.dtype)[:3]
    current, previous = identity + delta[:, indices + 1], identity + delta[:, indices]
    current = torch.cat((current, 1 - current.sum(-2, keepdim=True)), -2)
    previous = torch.cat((previous, 1 - previous.sum(-2, keepdim=True)), -2)
    matrix = torch.cat((previous, current), -1).double()
    energy = torch.einsum("bnsi,bnij,bnsj->bns", matrix, moments["gram"], matrix)
    cross = torch.einsum("bnsi,bnis->bns", matrix, moments["cross"])
    frame_error = energy - 2 * cross + moments["truth_energy"]
    # Sum disjoint 128-sample fragments into the original 44100-sample windows.
    errors, powers, leakages, mixtures = [], [], [], []
    for window in range(2):
        mask = windows == window
        errors.append(frame_error[:, mask].sum(1))
        powers.append(moments["truth_energy"][:, mask].sum(1))
        leakages.append(energy[:, mask].sum(1))
        mixtures.append(moments["mixture_energy"][:, mask].sum(1))
    error, power, leakage = (torch.stack(values, 1) for values in (errors, powers, leakages))
    mixture = torch.stack(mixtures, 1)
    active = power / (2 * WINDOW) > 1e-5
    values = (10 * torch.log10((error.clamp_min(0) + 1e-12) / (power + 1e-12))).clamp(-60, 60)
    count = active.sum((0, 1))
    per_stem = torch.where(active, values, 0).sum((0, 1)) / count.clamp_min(1)
    primary = per_stem.sum() / (count > 0).sum().clamp_min(1)
    absence_values = 10 * torch.log10(1 + leakage.clamp_min(0) / mixture[:, :, None].clamp_min(2 * WINDOW * 1e-5))
    absent_count = (~active).sum((0, 1))
    absence = (torch.where(~active, absence_values, 0).sum((0, 1)) / absent_count.clamp_min(1)).sum() / (absent_count > 0).sum().clamp_min(1)
    return primary + .5 * absence + .01 * delta.square().mean(), -per_stem, absence
