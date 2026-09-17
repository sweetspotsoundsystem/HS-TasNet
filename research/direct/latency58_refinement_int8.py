"""Quantize only the two measured refinement matrices in the PR #17 graph."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from research.direct.run_latency58_quality import require, sha

VERSION = "branch-memory-eighteen-u8s8-reduced-precise-v1"
PARENT_SHA = "c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61"
TARGETS = {
    "/fusion_refine_reduce/MatMul": ("fusion_refine_reduce", "onnx::MatMul_745", (1000, 128)),
    "/fusion_refine_expand/MatMul": ("fusion_refine_expand", "onnx::MatMul_746", (128, 1000)),
}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build(parent):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    from onnxruntime.quantization.quant_utils import quantize_data

    require(hashlib.sha256(parent.SerializeToString()).hexdigest() == PARENT_SHA,
            "Require exact PR #17 graph")
    graph = copy.deepcopy(parent)
    producers = {value: node for node in parent.graph.node for value in node.output}
    initializers = {value.name: value for value in parent.graph.initializer}
    removed_nodes, removed_initializers = set(), set()
    replacements, additions, proof = {}, [], []
    for node in parent.graph.node:
        if node.name not in TARGETS:
            continue
        module, expected_name, expected_shape = TARGETS[node.name]
        require(node.op_type == "MatMul", "Refinement topology changed")
        cast = producers[node.input[1]]
        require(cast.op_type == "Cast" and cast.input[0] in initializers
                and sum(node.input[1] in n.input for n in parent.graph.node) == 1
                and sum(cast.input[0] in n.input for n in parent.graph.node) == 1,
                "Refinement matrix sharing or cast changed")
        name = cast.input[0]
        weight = nh.to_array(initializers[name])
        require(name == expected_name and weight.shape == expected_shape and weight.dtype == np.float32,
                "Refinement matrix identity or layout changed")
        channels = [quantize_data(np.ascontiguousarray(c), T.INT8, symmetric=True, reduce_range=True)
                    for c in weight.T]
        zero = np.asarray([c[0] for c in channels], np.int8).reshape(-1)
        scale = np.asarray([c[1] for c in channels], np.float32).reshape(-1)
        quantized = np.ascontiguousarray(np.stack([c[2] for c in channels], axis=1), dtype=np.int8)
        require(np.all(zero == 0) and np.abs(quantized.astype(np.int16)).max() <= 64
                and weight.shape[0] * 255 * 64 < 2**31, "Integer range changed")
        for suffix, value in (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero)):
            additions.append(nh.from_array(value, name + suffix))
        prefix = node.name + "/refinement_u8s8"
        x, q, xs, xz, integer, f32, scales, y32 = [prefix + suffix for suffix in
            ("/input32", "/quantized", "/input_scale", "/input_zero", "/integer", "/float", "/scale", "/output32")]
        replacements[node.name] = [
            helper.make_node("Cast", [node.input[0]], [x], name=prefix + "/cast_input32", to=T.FLOAT),
            helper.make_node("DynamicQuantizeLinear", [x], [q, xs, xz], name=prefix + "/quantize"),
            helper.make_node("MatMulInteger", [q, name + "_quantized", xz, name + "_zero_point"],
                             [integer], name=prefix + "/matmul"),
            helper.make_node("Cast", [integer], [f32], name=prefix + "/cast_product32", to=T.FLOAT),
            helper.make_node("Mul", [xs, name + "_scale"], [scales], name=prefix + "/scales"),
            helper.make_node("Mul", [f32, scales], [y32], name=prefix + "/dequantize"),
            helper.make_node("Cast", [y32], list(node.output), name=prefix + "/cast_output64", to=T.DOUBLE),
        ]
        removed_nodes.add(cast.name)
        removed_initializers.add(name)
        proof.append({"module": module, "initializer": name, "shape": list(weight.shape),
            "source_matrix_sha256": digest(weight), "quantized_matrix_sha256": digest(quantized),
            "scale_sha256": digest(scale), "weight_zero_points_all_zero": True,
            "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64,
            "worst_centered_dot_absolute_sum": weight.shape[0] * 255 * 64,
            "bias_policy": "Both source refinement projections have no bias"})
    require(len(proof) == 2 and set(replacements) == set(TARGETS), "Expected two refinement products")
    nodes = [v for n in graph.graph.node if n.name not in removed_nodes for v in replacements.get(n.name, [n])]
    tensors = [v for v in graph.graph.initializer if v.name not in removed_initializers]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*tensors, *additions])
    del graph.graph.value_info[:]
    old_nodes = {n.name: n.SerializeToString() for n in parent.graph.node}
    require(all(n.SerializeToString() == old_nodes[n.name] for n in graph.graph.node if n.name in old_nodes)
            and all(v.SerializeToString() == initializers[v.name].SerializeToString()
                    for v in graph.graph.initializer if v.name in initializers)
            and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 18,
            "Unrelated node, original integer product or initializer changed")
    require([v.SerializeToString() for v in graph.graph.input] == [v.SerializeToString() for v in parent.graph.input]
            and [v.SerializeToString() for v in graph.graph.output] == [v.SerializeToString() for v in parent.graph.output],
            "Public interface changed")
    properties = {v.key: v.value for v in graph.metadata_props}
    properties["hs_tasnet.parent_sixteen_exporter_sha256"] = properties["hs_tasnet.integer_exporter_sha256"]
    properties.update({"hs_tasnet.runtime_variant": VERSION,
        "hs_tasnet.parent_graph_sha256": PARENT_SHA,
        "hs_tasnet.integer_exporter_sha256": sha(__file__),
        "hs_tasnet.integer_weight_precision": "Eighteen S8 symmetric reduced-range matrices [-64,64]",
        "hs_tasnet.floating_additional_layers": "phase factors, temporal attention, GRU biases/nonlinearities and refinement SiLU",
        "hs_tasnet.additional_refinement_integer_projections": "2",
        "hs_tasnet.deployment_quality_status": "unmeasured; source and parent scores do not score this graph"})
    helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"version": VERSION, "projections": proof,
        "only_two_refinement_products_and_their_weight_casts_replaced": True,
        "all_other_nodes_and_initializers_byte_exact": True,
        "public_interface_changed": False, "quality_measured": False, "native_host_qualified": False}
