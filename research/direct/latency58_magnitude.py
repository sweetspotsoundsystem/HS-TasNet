"""Zero-initialized phase-invariant spectral features for the 256-sample model."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from research.direct.latency58_residual_model import Latency58ResidualModel
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-normalized-magnitude-adapter-v1"
ADAPTER = "spec_encode.magnitude_projection.weight"


class MagnitudeEncoder(nn.Linear):
    """Retain the phase-sensitive linear map and add a learned magnitude map.

    The new map starts at zero. Features use only the already available FFT;
    normalization pools channels and frequency within the current frame.
    """

    def __init__(self, in_features=2052, out_features=500):
        require(in_features == 2052 and out_features == 500, "Unexpected inherited encoder")
        super().__init__(in_features, out_features)
        self.magnitude_projection = nn.Linear(1026, out_features, bias=False)
        nn.init.zeros_(self.magnitude_projection.weight)

    @staticmethod
    def magnitude_features(packed):
        with torch.autocast(packed.device.type, enabled=False):
            pairs = packed.float().unflatten(-1, (1026, 2))
            power = pairs.square().sum(-1)
            magnitude = (power + 1e-12).sqrt()
            scale = power.mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
            return torch.log1p(magnitude / scale)

    def forward(self, packed):
        original = F.linear(packed, self.weight, self.bias)
        additional = self.magnitude_projection(self.magnitude_features(packed))
        return original + additional


class Latency58MagnitudeModel(Latency58ResidualModel):
    def __init__(self):
        super().__init__()
        self.spec_encode = MagnitudeEncoder()

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "version": VERSION,
                "magnitude_features": "log1p(sqrt(real^2+imag^2+1e-12)/sqrt(max(frame_mean_power,1e-8)))",
                "magnitude_normalization_axes": "current-frame stereo and frequency only",
                "additional_neural_parameters": 513000,
                "additional_audio_buffering_samples": 0,
                "additional_state_tensors": 0, "native_host_qualified": False}

    def train_adapter_only(self):
        self.train().requires_grad_(False)
        self.spec_encode.magnitude_projection.weight.requires_grad_(True)
        return self

    @classmethod
    def from_parent(cls, parent):
        require(type(parent) is Latency58ResidualModel
                and all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parent.state_dict().values()),
                "Authenticate the CPU residual parent first")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        values = {**parent.state_dict(), ADAPTER: torch.zeros(500, 1026)}
        model.load_state_dict(values, strict=True)
        require(state_sha256({k: v for k, v in model.state_dict().items() if k != ADAPTER})
                == state_sha256(parent.state_dict()), "Magnitude initialization changed inherited tensors")
        model.provenance = {**copy.deepcopy(parent.provenance),
                            "magnitude_version": VERSION,
                            "magnitude_parent_state_sha256": state_sha256(parent.state_dict()),
                            "magnitude_parent_provenance": copy.deepcopy(parent.provenance),
                            "magnitude_updates": 0}
        return model.eval().requires_grad_(False)
