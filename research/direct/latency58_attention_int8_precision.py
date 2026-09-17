"""Keep higher precision around the attention model's ten signed integer projections, one biasless.

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


def portable_silu(graph):
    """Avoid ORT 1.26's unsupported double QuickGelu optimization of x*sigmoid(x)."""
    from onnx import helper, numpy_helper
    producers = {value: node for node in graph.graph.node for value in node.output}
    matches = [node for node in graph.graph.node if node.op_type == "Sigmoid"
               and producers.get(node.input[0]) is not None
               and producers[node.input[0]].name.startswith("/fusion_refine_reduce/")]
    require(len(matches) == 1, "Require the single refinement SiLU")
    sigmoid = matches[0]
    consumers = [node for node in graph.graph.node if sigmoid.output[0] in node.input]
    require(len(consumers) == 1 and consumers[0].op_type == "Mul"
            and sigmoid.input[0] in consumers[0].input, "Refinement SiLU topology differs")
    product = consumers[0]
    prefix = "/refinement_silu_portable/"
    replacements = [
        helper.make_node("Neg", list(sigmoid.input), [prefix + "negative"], name=prefix + "neg"),
        helper.make_node("Exp", [prefix + "negative"], [prefix + "exponential"], name=prefix + "exp"),
        helper.make_node("Constant", [], [prefix + "one"], name=prefix + "constant",
                         value=numpy_helper.from_array(np.asarray(1., np.float32))),
        helper.make_node("Add", [prefix + "exponential", prefix + "one"], [prefix + "denominator"],
                         name=prefix + "add"),
        helper.make_node("Div", [sigmoid.input[0], prefix + "denominator"], list(product.output),
                         name=prefix + "divide")]
    nodes = [replacement for node in graph.graph.node
             for replacement in (replacements if node.name == sigmoid.name else
                                 [] if node.name == product.name else [node])]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)


def rewrite(graph):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper

    portable_silu(graph)
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
            require(integer.name.endswith("/magnitude_u8s8/matmul"), "Unexpected biasless projection")
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
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "temporal-attention-hop128-ten-u8s8-fp64-quantizer-ancestors-v1",
                     "hs_tasnet.native_host_qualified": "false",
                     "hs_tasnet.precise_node_count": str(len(precise))})
    helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    return graph
