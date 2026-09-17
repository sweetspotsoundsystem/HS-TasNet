"""FP32 parameter EMA with raw-endpoint binding; no inference graph changes.

This averages after every optimizer update with a constant decay. It does not
replicate the upstream EMA package's optional warmup or update-frequency rules.
Raw weights and their optimizer remain together; EMA weights are separate.
"""
from __future__ import annotations

import copy
import math

import torch

from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256

SCHEMA = "latency58-branch-parameter-ema-v1"


class BranchParameterEMA:
    def __init__(self, model, *, decay, base_state_sha256):
        require(type(decay) is float and math.isfinite(decay) and 0 <= decay < 1,
                "Invalid EMA decay")
        require(type(model) is Latency58BranchMemoryModel,
                "EMA requires the qualified branch-memory model")
        self.decay, self.updates = decay, 0
        self.base_state_sha256 = base_state_sha256
        self.architecture = copy.deepcopy(model.architecture_metadata)
        self.parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        self.raw_state_sha256 = state_sha256(model.state_dict())
        require(self.raw_state_sha256 == base_state_sha256, "EMA base model differs")
        self._validate_model(model)

    def _validate_model(self, model):
        require(type(model) is Latency58BranchMemoryModel
                and model.architecture_metadata == self.architecture,
                "EMA model architecture differs")
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        require(list(parameters) == list(self.parameters) and len(parameters) == 40
                and list(buffers) == list(self.buffers), "EMA tensor inventory differs")
        require(all(p.dtype == average.dtype == torch.float32 and p.shape == average.shape
                    and p.device == average.device and bool(torch.isfinite(p).all())
                    and bool(torch.isfinite(average).all())
                    for p, average in zip(parameters.values(), self.parameters.values(), strict=True)),
                "Invalid EMA parameter tensors")
        require(all(b.dtype == fixed.dtype == torch.float32 and b.shape == fixed.shape
                    and b.device == fixed.device and bool(torch.isfinite(b).all())
                    and torch.equal(b.view(torch.int32), fixed.view(torch.int32))
                    for b, fixed in zip(buffers.values(), self.buffers.values(), strict=True)),
                "EMA fixed buffers changed")

    @torch.no_grad()
    def update(self, model, *, step):
        require(type(step) is int and step == self.updates + 1,
                "EMA updates must be contiguous")
        self._validate_model(model)
        # Validate every tensor before mutating any average.
        for name, parameter in model.named_parameters():
            if self.decay == 0:
                self.parameters[name].copy_(parameter)
            else:
                self.parameters[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)
        self.raw_state_sha256 = state_sha256(model.state_dict())
        self.updates = step

    def state_dict(self, model):
        self._validate_model(model)
        require(state_sha256(model.state_dict()) == self.raw_state_sha256,
                "Raw model changed since the last EMA update")
        parameters = {name: p.detach().cpu().clone() for name, p in self.parameters.items()}
        buffers = {name: b.detach().cpu().clone() for name, b in self.buffers.items()}
        return {"schema": SCHEMA, "decay": self.decay, "updates": self.updates,
                "base_state_sha256": self.base_state_sha256,
                "raw_state_sha256": self.raw_state_sha256,
                "architecture": copy.deepcopy(self.architecture),
                "parameters": parameters, "buffers": buffers,
                "parameter_names": list(parameters),
                "ema_parameters_sha256": state_sha256(parameters),
                "fixed_buffers_sha256": state_sha256(buffers)}

    @classmethod
    def from_state_dict(cls, model, payload, *, expected_step, decay, base_state_sha256):
        keys = {"schema", "decay", "updates", "base_state_sha256", "raw_state_sha256",
                "architecture", "parameters", "buffers", "parameter_names",
                "ema_parameters_sha256", "fixed_buffers_sha256"}
        require(isinstance(payload, dict) and set(payload) == keys and payload["schema"] == SCHEMA,
                "Invalid EMA payload schema")
        require(type(expected_step) is int and expected_step >= 0
                and type(payload["updates"]) is int and payload["updates"] == expected_step
                and type(payload["decay"]) is float and type(decay) is float
                and math.isfinite(decay) and 0 <= decay < 1 and payload["decay"] == decay
                and payload["base_state_sha256"] == base_state_sha256
                and payload["raw_state_sha256"] == state_sha256(model.state_dict()),
                "EMA payload belongs to another raw endpoint or policy")
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        require(type(model) is Latency58BranchMemoryModel
                and payload["architecture"] == model.architecture_metadata
                and payload["parameter_names"] == list(parameters)
                and isinstance(payload["parameters"], dict) and list(payload["parameters"]) == list(parameters)
                and isinstance(payload["buffers"], dict) and list(payload["buffers"]) == list(buffers)
                and all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                        and value.shape == reference.shape and bool(torch.isfinite(value).all())
                        for stored, current in ((payload["parameters"], parameters), (payload["buffers"], buffers))
                        for name, value in stored.items() for reference in (current[name],)),
                "Invalid EMA serialized tensors")
        require(state_sha256(payload["parameters"]) == payload["ema_parameters_sha256"]
                and state_sha256(payload["buffers"]) == payload["fixed_buffers_sha256"]
                == state_sha256(buffers), "EMA tensor fingerprints differ")
        result = cls.__new__(cls)
        result.decay, result.updates = decay, expected_step
        result.base_state_sha256 = base_state_sha256
        result.raw_state_sha256 = payload["raw_state_sha256"]
        result.architecture = copy.deepcopy(payload["architecture"])
        result.parameters = {name: p.detach().to(parameters[name].device).clone()
                             for name, p in payload["parameters"].items()}
        result.buffers = {name: b.detach().to(buffers[name].device).clone()
                          for name, b in payload["buffers"].items()}
        result._validate_model(model)
        return result

    def inference_copy(self, model):
        """Return a separate model, explicitly marking its averaging provenance."""
        payload = self.state_dict(model)
        result = copy.deepcopy(model).eval().requires_grad_(False)
        result.load_state_dict({**payload["parameters"], **payload["buffers"]}, strict=True)
        result.provenance = {**result.provenance, "weight_averaging": {
            key: payload[key] for key in ("schema", "decay", "updates", "base_state_sha256",
                                          "raw_state_sha256", "ema_parameters_sha256")},
            "quality_measured": False}
        return result
