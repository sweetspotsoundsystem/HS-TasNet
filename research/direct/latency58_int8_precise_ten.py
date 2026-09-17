"""Keep higher precision around ten reviewed integer projections, one biasless.

Output decoding returns to the original FP32 arithmetic. The graph transform
derives this boundary from data dependencies and inserts explicit casts.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.direct.run_latency58_quality import require


def rewrite(graph):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper

    original = onnx.shape_inference.infer_shapes(copy.deepcopy(graph), strict_mode=True, check_type=True)
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*original.graph.input, *original.graph.output, *original.graph.value_info]}
    types.update({v.name: v.data_type for v in original.graph.initializer})
    require(all(v in types for n in original.graph.node for v in n.output), "Missing inferred value type")
    producers = {v: n for n in original.graph.node for v in n.output}
    consumers = {}
    for node in original.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    def single(value, operation):
        nodes = consumers[value]
        require(len(nodes) == 1 and nodes[0].op_type == operation, "Projection topology changed")
        return nodes[0]
    islands, quantizers, biasless = set(), [], []
    initializers = {value.name: value for value in original.graph.initializer}
    for integer in (n for n in original.graph.node if n.op_type == "MatMulInteger"):
        quantize = producers[integer.input[0]]
        cast = single(integer.output[0], "Cast")
        scaled = single(cast.output[0], "Mul")
        scales = producers[next(v for v in scaled.input if v != cast.output[0])]
        following = single(scaled.output[0], "Add")
        other = next(value for value in following.input if value != scaled.output[0])
        # Nine inherited projections have an explicit constant bias. The new
        # magnitude projection has none: its next Add combines two learned
        # projections and must remain outside the FP32 integer block.
        members = [quantize, integer, cast, scaled, scales]
        if other in initializers:
            require(initializers[other].data_type == T.FLOAT and len(initializers[other].dims) == 1,
                    "Unexpected projection bias")
            members.append(following)
        else:
            require(integer.name.endswith("/magnitude_u8u8/matmul"), "Unexpected biasless projection")
            biasless.append(integer.name)
        require(quantize.op_type == "DynamicQuantizeLinear" and scales.op_type == "Mul"
                and quantize.output[1] in scales.input, "Wrong integer block")
        islands.update(node.name for node in members)
        quantizers.append(quantize)
    require(len(islands) == 59 and len(quantizers) == 10 and len(biasless) == 1,
            "Require nine six-node blocks and one five-node biasless magnitude block")
    precise = set()
    def visit(value):
        node = producers.get(value)
        if node is None or node.name in islands or node.name in precise:
            return
        precise.add(node.name)
        for parent in node.input:
            visit(parent)
    for quantizer in quantizers:
        visit(quantizer.input[0])
    require(all(n.name not in precise for n in original.graph.node if n.name in islands), "Precision boundary crossed a projection")
    require(next(n for n in original.graph.node if n.op_type == "DFT").name in precise
            and sum(n.op_type == "DFT" and n.name in precise for n in original.graph.node) == 1,
            "Only the feature FFT should need FP64")
    available = {v.name: v.type.tensor_type.elem_type for v in original.graph.input}
    available.update({t.name: t.data_type for t in original.graph.initializer})
    nodes, casts = [], {}
    public_names = {v.name for v in original.graph.output}
    aliases = {name: name + "__internal" for name in public_names}
    def input_as(name, expected):
        actual_name = aliases.get(name, name)
        actual = available[name]
        if actual != T.FLOAT and actual != T.DOUBLE:
            return actual_name
        if actual == expected:
            return actual_name
        key = (name, expected)
        if key not in casts:
            value = name + ("__to64" if expected == T.DOUBLE else "__to32")
            nodes.append(helper.make_node("Cast", [actual_name], [value],
                         name="/precise_core/cast_" + str(len(casts)), to=expected))
            casts[key] = value
        return casts[key]
    for original_node in original.graph.node:
        node = copy.deepcopy(original_node)
        precision = T.DOUBLE if node.name in precise else T.FLOAT
        for index, value in enumerate(node.input):
            if value:
                node.input[index] = input_as(value, precision)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == T.FLOAT and precision == T.DOUBLE:
                attribute.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attribute.t).astype(np.float64), attribute.t.name))
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == T.FLOAT:
                attribute.i = precision
        for index, value in enumerate(node.output):
            available[value] = precision if types[value] == T.FLOAT else types[value]
            node.output[index] = aliases.get(value, value)
        nodes.append(node)
    for output in original.graph.output:
        source = input_as(output.name, T.FLOAT)
        nodes.append(helper.make_node("Identity", [source], [output.name], name="/precise_core/output/" + output.name))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.value_info[:]
    # Every initializer is left byte-exact; constant casts can be folded by ORT.
    require([t.SerializeToString() for t in graph.graph.initializer] ==
            [t.SerializeToString() for t in original.graph.initializer], "Initializer bytes changed")
    metadata = {"source." + item.key: item.value for item in graph.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "quadrature-hop128-ten-u8u8-fp64-quantizer-ancestors-v1",
                     "hs_tasnet.native_host_qualified": "false",
                     "hs_tasnet.precise_node_count": str(len(precise))})
    helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    return graph

