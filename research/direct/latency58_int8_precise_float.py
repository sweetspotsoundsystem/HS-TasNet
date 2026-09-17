"""Higher-precision floating arithmetic around unchanged U8U8 projections.

The public audio/state tensors remain FP32. Dynamic activation quantization
also receives FP32. Complete integer projection blocks retain their original
FP32 dequantization and bias addition; FFTs, nonlinearities and synthesis use
FP64. This is an approximate inference variant, not a parity relaxation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from research.direct.run_latency58_quality import require


def rewrite(graph):
    import copy
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    original = copy.deepcopy(graph)
    require(sum(n.op_type == "DynamicQuantizeLinear" for n in original.graph.node) == 9
            and sum(n.op_type == "MatMulInteger" for n in original.graph.node) == 9,
            "Expected the reviewed nine-projection graph")
    consumers = {}
    producers = {value: n for n in original.graph.node for value in n.output}
    initializers = {t.name: t for t in original.graph.initializer}
    for node in original.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    islands, terminals, kept_float = set(), set(), set()
    def single_consumer(value, operation):
        matches = consumers[value]
        require(len(matches) == 1 and matches[0].op_type == operation, "Integer projection topology changed")
        return matches[0]
    for integer in (n for n in original.graph.node if n.op_type == "MatMulInteger"):
        quantize = producers[integer.input[0]]
        require(quantize.op_type == "DynamicQuantizeLinear", "Unexpected integer input")
        cast = single_consumer(integer.output[0], "Cast")
        scaled = single_consumer(cast.output[0], "Mul")
        scale_name = next(value for value in scaled.input if value != cast.output[0])
        scales = producers[scale_name]
        bias = single_consumer(scaled.output[0], "Add")
        require(scales.op_type == "Mul" and quantize.output[1] in scales.input, "Wrong dequantization scales")
        members = (quantize, integer, cast, scaled, scales, bias)
        islands.update(n.name for n in members)
        terminals.add(bias.name)
        kept_float.update(value for n in members for value in n.input
                          if value in initializers and initializers[value].data_type == TensorProto.FLOAT)
    require(len(islands) == 54 and len(terminals) == 9 and len(kept_float) == 18,
            "Expected nine original six-node projection blocks and their scales/biases")
    integer_before = {t.name: t.SerializeToString() for t in original.graph.initializer
                      if t.data_type != TensorProto.FLOAT}
    float_before = {t.name: numpy_helper.to_array(t).copy() for t in original.graph.initializer
                    if t.data_type == TensorProto.FLOAT}
    for tensor in graph.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT and tensor.name not in kept_float:
            tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float64), tensor.name))
    aliases = {v.name: v.name + "__float64" for v in [*graph.graph.input, *graph.graph.output]}
    rewritten = [helper.make_node("Cast", [v.name], [aliases[v.name]],
                 name="/precise_float/input/" + v.name, to=TensorProto.DOUBLE) for v in graph.graph.input]
    for node in original.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = aliases.get(name, name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT:
                attribute.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attribute.t).astype(np.float64), attribute.t.name))
            if node.name not in islands and node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                attribute.i = TensorProto.DOUBLE
        if node.op_type == "DynamicQuantizeLinear":
            input32 = node.name + "/input32"
            rewritten.append(helper.make_node("Cast", [node.input[0]], [input32],
                             name=node.name + "/cast_input32", to=TensorProto.FLOAT))
            node.input[0] = input32
            rewritten.append(node)
        elif node.name in terminals:
            value = node.output[0]
            node.output[0] = value + "__projection32"
            rewritten.append(node)
            rewritten.append(helper.make_node("Cast", [node.output[0]], [value],
                             name=node.name + "/cast_result64", to=TensorProto.DOUBLE))
        else:
            rewritten.append(node)
    rewritten.extend(helper.make_node("Cast", [aliases[v.name]], [v.name],
                     name="/precise_float/output/" + v.name, to=TensorProto.FLOAT) for v in graph.graph.output)
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    # Type inference must recompute internal floating types. Public tensor
    # names, shapes and FP32 element types remain exactly as in the source.
    del graph.graph.value_info[:]
    require(all(t.SerializeToString() == integer_before[t.name] for t in graph.graph.initializer
                if t.name in integer_before)
            and all(np.array_equal(numpy_helper.to_array(t), float_before[t.name].astype(
                    np.float32 if t.name in kept_float else np.float64))
                    for t in graph.graph.initializer if t.name in float_before), "Initializer values changed")
    metadata = {"source." + item.key: item.value for item in graph.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "c204-hop128-u8u8-fp64-surrounding-arithmetic-v1",
                     "hs_tasnet.native_host_qualified": "false"})
    helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    return graph


class PreciseProjection(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("weight", original.centered_weight.detach().clone())
        self.register_buffer("scale", original.scale.detach().double().clone())
        self.register_buffer("bias", None if original.bias is None else original.bias.detach().double().clone())

    def forward(self, values):
        require(values.dtype == torch.float64 and values.device.type == "cpu", "Require CPU FP64 values")
        x = values.float()
        minimum, maximum = x.min().clamp_max(0), x.max().clamp_min(0)
        scale = (maximum - minimum) / 255.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(-minimum / scale).clamp(0, 255).to(torch.int32)
        quantized = (torch.round(x / scale) + zero).clamp(0, 255).to(torch.int32)
        integer = (quantized.reshape(1, -1) - zero) @ self.weight
        output = integer.float() * (scale * self.scale.float())
        if self.bias is not None:
            output = output + self.bias.float()
        return output.double().reshape(*x.shape[:-1], self.weight.shape[1])


class PublicFloat32(nn.Module):
    def __init__(self, internal):
        super().__init__()
        self.internal = internal.double()

    def forward(self, *values):
        require(all(v.dtype == torch.float32 and v.device.type == "cpu" for v in values), "Require public FP32")
        return tuple(v.float() for v in self.internal(*(v.double() for v in values)))


def make_reference(native, original_graph):
    from research.direct.latency58_int8_reference import make_reference as original_reference, IntegerLinear
    wrapper, proof = original_reference(native, original_graph)
    replaced = []
    def replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            path = prefix + name
            if isinstance(child, IntegerLinear):
                setattr(parent, name, PreciseProjection(child))
                replaced.append(path)
            else:
                replace(child, path + ".")
    replace(wrapper)
    require(len(replaced) == 9, "Expected nine independently rebuilt projections")
    return PublicFloat32(wrapper).eval().requires_grad_(False), proof
