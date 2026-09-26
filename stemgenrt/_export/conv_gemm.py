"""Express fixed one-frame hop128 convolutions as prepackable ONNX Gemm."""
from __future__ import annotations

import hashlib

import numpy as np
from onnx import helper, numpy_helper


def convert(graph, names):
    """Mutate only the selected convolutions, their weight shapes, and plumbing.

    Each original convolution has exactly one spatial output. Flattening its
    entire input and weight therefore preserves the underlying dot products.
    Runtime accumulation order still requires numerical parity verification.
    """
    expected = {"/conv_encode/Conv": ((3000, 2, 1024), 128),
                "/basis_to_embed/Conv": ((500, 1500, 1), 1)}
    assert names and len(set(names)) == len(names) and set(names) <= set(expected)
    assert list(graph.graph.input[0].type.tensor_type.shape.dim[i].dim_value for i in range(3)) == [1, 2, 128]
    original_nodes = {node.name: node.SerializeToString() for node in graph.graph.node if node.name not in names}
    weights = {tensor.name: tensor for tensor in graph.graph.initializer}
    original_initializers = {t.name: t.SerializeToString() for t in graph.graph.initializer}
    converted_weights = set()
    replacements, proof = {}, []
    for name in names:
        nodes = [node for node in graph.graph.node if node.name == name]
        assert len(nodes) == 1
        node = nodes[0]
        shape, stride = expected[name]
        assert node.op_type == "Conv" and len(node.input) == 3 and len(node.output) == 1
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        assert attributes == {"dilations": [1], "group": 1, "kernel_shape": [shape[-1]], "pads": [0, 0], "strides": [stride]}
        weight = weights[node.input[1]]
        array = numpy_helper.to_array(weight)
        assert array.dtype == np.float32 and array.shape == shape
        assert sum(node.input[1] in other.input for other in graph.graph.node) == 1
        bias = numpy_helper.to_array(weights[node.input[2]])
        assert bias.shape == (shape[0],) and bias.dtype == np.float32
        elements = shape[1] * shape[2]
        raw_sha = hashlib.sha256(array.tobytes()).hexdigest()
        reshaped = numpy_helper.from_array(array.reshape(shape[0], elements), weight.name)
        assert hashlib.sha256(numpy_helper.to_array(reshaped).tobytes()).hexdigest() == raw_sha
        weight.CopyFrom(reshaped)
        converted_weights.add(weight.name)
        prefix = name + "_as_gemm"
        input_shape, output_shape = prefix + "/input_shape", prefix + "/output_shape"
        graph.graph.initializer.extend([
            numpy_helper.from_array(np.asarray([1, elements], dtype=np.int64), input_shape),
            numpy_helper.from_array(np.asarray([1, shape[0], 1], dtype=np.int64), output_shape)])
        replacements[name] = [
            helper.make_node("Reshape", [node.input[0], input_shape], [prefix + "/flat_input"], name=prefix + "/flatten"),
            helper.make_node("Gemm", [prefix + "/flat_input", node.input[1], node.input[2]], [prefix + "/matrix_output"],
                             name=prefix + "/Gemm", alpha=1., beta=1., transA=0, transB=1),
            helper.make_node("Reshape", [prefix + "/matrix_output", output_shape], list(node.output), name=prefix + "/restore")]
        proof.append({"node": name, "weight": weight.name, "weight_shape_before": list(shape),
                      "weight_shape_after": [shape[0], elements], "weight_values_sha256": raw_sha,
                      "weight_values_bit_exact": True, "bias_unchanged": True, "spatial_output_count": 1})
    nodes = [replacement for node in graph.graph.node for replacement in replacements.get(node.name, [node])]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    assert {node.name: node.SerializeToString() for node in graph.graph.node if node.name in original_nodes} == original_nodes
    assert all(t.SerializeToString() == original_initializers[t.name] for t in graph.graph.initializer
               if t.name in original_initializers and t.name not in converted_weights)
    return proof
