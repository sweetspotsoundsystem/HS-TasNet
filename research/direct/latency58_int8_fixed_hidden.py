"""Experimental fixed U8 ranges for the three bounded GRU projections."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from research.direct.run_latency58_quality import require

PREFIXES = ("model.fusion_branch.weight_hh_l0", "model.fusion_branch.weight_hh_l1",
            "model.fusion_branch.weight_ih_l1")
SCALE = np.float32(1.0 / 127.0)
ZERO = np.uint8(128)


def convert(graph):
    from onnx import helper, numpy_helper
    changed = copy.deepcopy(graph)
    nodes = list(changed.graph.node)
    producers = {value: node for node in nodes for value in node.output}
    selected = [node for node in nodes if node.op_type == "MatMulInteger"
                and node.input[1].removesuffix("_quantized") in PREFIXES]
    require(len(selected) == 3, "Expected the two hidden maps and second-layer input map")
    replacements = {}
    for node in selected:
        quant = producers[node.input[0]]
        require(quant.op_type == "DynamicQuantizeLinear" and len(quant.output) == 3,
                "Expected dynamic U8 activation input")
        replacements[quant.name] = helper.make_node("QuantizeLinear",
            [quant.input[0], quant.output[1], quant.output[2]], [quant.output[0]],
            name=quant.name + "_fixed_hidden")
        changed.graph.initializer.extend([
            numpy_helper.from_array(np.asarray(SCALE), name=quant.output[1]),
            numpy_helper.from_array(np.asarray(ZERO), name=quant.output[2])])
    del changed.graph.node[:]
    changed.graph.node.extend([replacements.get(node.name, node) for node in nodes])
    props = {item.key: item.value for item in graph.metadata_props}
    props.update({"hs_tasnet.runtime_variant": "c204-hop128-u8u8-fixed-hidden-v1",
                  "hs_tasnet.fixed_hidden_activation_scale": repr(float(SCALE)),
                  "hs_tasnet.fixed_hidden_activation_zero_point": str(int(ZERO)),
                  "hs_tasnet.fixed_hidden_range_scope": ";".join(PREFIXES)})
    helper.set_model_props(changed, props)
    require([v.SerializeToString() for v in changed.graph.initializer[:len(graph.graph.initializer)]]
            == [v.SerializeToString() for v in graph.graph.initializer], "Original weights changed")
    return changed


class FixedIntegerLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("centered_weight", original.centered_weight.clone())
        self.register_buffer("scale", original.scale.clone())
        self.register_buffer("bias", None if original.bias is None else original.bias.clone())

    def forward(self, x):
        require(x.device.type == "cpu" and x.dtype == torch.float32
                and x.numel() == self.centered_weight.shape[0], "Require one CPU FP32 frame")
        require(float(x.abs().max()) <= 1.000001, "Unexpected hidden values outside the bounded-state contract")
        quantized = (torch.round(x / float(SCALE)) + int(ZERO)).clamp(0, 255).to(torch.int32)
        integer = (quantized.reshape(1, -1) - int(ZERO)) @ self.centered_weight
        value = integer.float() * (float(SCALE) * self.scale)
        if self.bias is not None:
            value = value + self.bias
        return value.reshape(*x.shape[:-1], self.centered_weight.shape[1])


def make_reference(native, graph):
    from research.direct.latency58_int8_reference import make_reference as dynamic_reference
    reference, proof = dynamic_reference(native, graph)
    gru = reference.model.fusion_branch
    gru.hidden_maps[0] = FixedIntegerLinear(gru.hidden_maps[0])
    gru.hidden_maps[1] = FixedIntegerLinear(gru.hidden_maps[1])
    gru.input_maps[1] = FixedIntegerLinear(gru.input_maps[1])
    return reference, proof
