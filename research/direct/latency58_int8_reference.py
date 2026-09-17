"""Independent CPU PyTorch arithmetic for the reviewed dynamic U8U8 graph.

The nine integer weight matrices are recomputed from the FP32 checkpoint and
compared with the saved graph. ONNX Runtime is never used to produce an oracle.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn

from research.direct.run_latency58_quality import require


class IntegerLinear(nn.Module):
    def __init__(self, weight, bias, stored, prefix):
        super().__init__()
        weight = weight.detach().float().cpu().reshape(weight.shape[0], -1).numpy()
        low = np.minimum(weight.min(axis=1), np.float32(0))
        high = np.maximum(weight.max(axis=1), np.float32(0))
        scale64 = (high - low).astype(np.float64) / np.float64(255)
        degenerate = scale64 < np.finfo(np.float32).tiny
        scale64 = np.where(degenerate, np.float64(1), scale64)
        zero = np.round(-low / scale64).astype(np.uint8)
        zero[degenerate] = 0
        scale = scale64.astype(np.float32)
        quantized = np.clip(np.round(weight / scale[:, None]) + zero[:, None], 0, 255).astype(np.uint8)
        require(np.array_equal(quantized.T, stored[prefix + "_quantized"])
                and np.array_equal(scale, stored[prefix + "_scale"])
                and np.array_equal(zero, stored[prefix + "_zero_point"]),
                "Independent checkpoint quantization differs: " + prefix)
        self.register_buffer("centered_weight", torch.from_numpy(
            np.ascontiguousarray(quantized.T.astype(np.int32) - zero.astype(np.int32))))
        self.register_buffer("scale", torch.from_numpy(scale.copy()))
        self.register_buffer("bias", None if bias is None else bias.detach().float().cpu().clone())
        self.proof = {"initializer": prefix, "source_float_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                      "quantized_matrix_sha256": hashlib.sha256(np.ascontiguousarray(quantized.T).tobytes()).hexdigest(),
                      "weight_scale_zero_point_bit_exact": True}

    def forward(self, x):
        require(x.device.type == "cpu" and x.dtype == torch.float32
                and x.numel() == self.centered_weight.shape[0], "Oracle requires one CPU FP32 frame")
        minimum, maximum = x.min().clamp_max(0), x.max().clamp_min(0)
        scale = (maximum - minimum) / 255.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(-minimum / scale).clamp(0, 255).to(torch.int32)
        quantized = (torch.round(x / scale) + zero).clamp(0, 255).to(torch.int32)
        integer = (quantized.reshape(1, -1) - zero) @ self.centered_weight
        value = integer.float() * (scale * self.scale)
        if self.bias is not None:
            value = value + self.bias
        return value.reshape(*x.shape[:-1], self.centered_weight.shape[1])


class IntegerConv(nn.Module):
    def __init__(self, original, stored, prefix):
        super().__init__()
        self.projection = IntegerLinear(original.weight, original.bias, stored, prefix)

    def forward(self, x):
        return self.projection(x.reshape(1, -1)).unsqueeze(-1)


class IntegerGRU(nn.Module):
    def __init__(self, original, stored):
        super().__init__()
        self.input_maps = nn.ModuleList()
        self.hidden_maps = nn.ModuleList()
        for layer in range(2):
            for maps, kind in ((self.input_maps, "ih"), (self.hidden_maps, "hh")):
                maps.append(IntegerLinear(getattr(original, f"weight_{kind}_l{layer}"),
                    getattr(original, f"bias_{kind}_l{layer}"), stored,
                    f"model.fusion_branch.weight_{kind}_l{layer}"))

    def forward(self, x, hidden):
        current, returned = x[:, 0], []
        for layer in range(2):
            ir, iz, inn = self.input_maps[layer](current).chunk(3, dim=-1)
            hr, hz, hn = self.hidden_maps[layer](hidden[layer]).chunk(3, dim=-1)
            reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            candidate = (inn + reset * hn).tanh()
            current = candidate + update * (hidden[layer] - candidate)
            returned.append(current)
        return current[:, None], torch.stack(returned)


def make_reference(native, graph):
    from onnx import numpy_helper
    from research.direct.latency58_asymmetric_onnx import make_export_copy
    stored = {value.name: numpy_helper.to_array(value) for value in graph.graph.initializer}
    wrapper = make_export_copy(native)
    model = wrapper.model
    model.conv_encode = IntegerConv(model.conv_encode, stored, "model.conv_encode.weight")
    model.basis_to_embed = IntegerConv(model.basis_to_embed, stored, "model.basis_to_embed.weight")
    model.fusion_branch = IntegerGRU(model.fusion_branch, stored)
    for name, prefix in (("spec_encode", "onnx::MatMul_362"), ("to_spec_masks", "onnx::MatMul_363"),
                         ("to_waveform_masks", "onnx::MatMul_376")):
        previous = getattr(model, name)
        setattr(model, name, IntegerLinear(previous.weight, previous.bias, stored, prefix))
    proof = [module.proof for module in model.modules() if isinstance(module, IntegerLinear)]
    require(len(proof) == 9, "Expected nine independently quantized projections")
    return wrapper.eval().requires_grad_(False), proof
