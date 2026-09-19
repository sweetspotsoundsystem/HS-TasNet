"""Quantize the two profiled branch output products in the PR #15 graph.

This changes inference numerics and needs new quality and native-host evidence.
Biases, nonlinearities and all public states retain the parent graph's
arithmetic. Existing fourteen integer projections are unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from torch import nn

from .helpers import require, sha

VERSION = "branch-memory-sixteen-u8s8-reduced-precise-v1"
TARGETS = {f"/{name}/MatMul": name
           for name in ("spec_memory_output", "waveform_memory_output")}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build(parent, *, expected_parent_sha256="878c74694fa4c558de1c5a75837893a0afeadcf57f6e3b860d5904cab04e9fc9"):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    from onnxruntime.quantization.quant_utils import quantize_data
    require(hashlib.sha256(parent.SerializeToString()).hexdigest()
            == expected_parent_sha256,
            "Start with the exact PR #15 graph")
    graph = copy.deepcopy(parent)
    producers = {v: n for n in parent.graph.node for v in n.output}
    initializers = {v.name: v for v in parent.graph.initializer}
    removed_nodes, removed_initializers, replacements, additions, proof = set(), set(), {}, [], []
    for node in parent.graph.node:
        if node.name not in TARGETS:
            continue
        require(node.op_type == "MatMul", "Profiled product topology changed")
        cast = producers[node.input[1]]
        require(cast.op_type == "Cast" and cast.input[0] in initializers
                and sum(node.input[1] in n.input for n in parent.graph.node) == 1
                and sum(cast.input[0] in n.input for n in parent.graph.node) == 1,
                "Profiled matrix is shared or no longer a cast initializer")
        name = cast.input[0]
        weight = nh.to_array(initializers[name])
        require(weight.shape == (500, 500) and weight.dtype == np.float32
                and name == {"spec_memory_output": "onnx::MatMul_747", "waveform_memory_output": "onnx::MatMul_748"}[TARGETS[node.name]], "Wrong branch matrix identity or layout")
        channels = [quantize_data(np.ascontiguousarray(c), T.INT8, symmetric=True, reduce_range=True)
                    for c in weight.T]
        zero = np.asarray([c[0] for c in channels], np.int8).reshape(-1)
        scale = np.asarray([c[1] for c in channels], np.float32).reshape(-1)
        quantized = np.ascontiguousarray(np.stack([c[2] for c in channels], axis=1), dtype=np.int8)
        require(np.all(zero == 0) and np.abs(quantized.astype(np.int16)).max() <= 64
                and weight.shape[0] * 255 * 64 < 2**31, "Integer accumulator range changed")
        for suffix, value in (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero)):
            additions.append(nh.from_array(value, name + suffix))
        prefix = node.name + "/branch_output_u8s8"
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
            helper.make_node("Cast", [y32], list(node.output), name=prefix + "/cast_output64", to=T.DOUBLE)]
        removed_nodes.add(cast.name)
        removed_initializers.add(name)
        proof.append({"module": TARGETS[node.name], "initializer": name, "shape": list(weight.shape),
            "source_matrix_sha256": digest(weight), "quantized_matrix_sha256": digest(quantized),
            "scale_sha256": digest(scale), "weight_zero_points_all_zero": True,
            "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64,
            "worst_centered_dot_absolute_sum": weight.shape[0] * 255 * 64,
            "bias_policy": "These two source projections have no bias"})
    require(len(proof) == 2 and set(replacements) == set(TARGETS), "Expected exactly two branch output products")
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
            and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 16,
            "An unrelated node, original integer product or initializer changed")
    properties = {v.key: v.value for v in graph.metadata_props}
    properties["hs_tasnet.parent_fourteen_exporter_sha256"] = properties["hs_tasnet.integer_exporter_sha256"]
    properties.update({"hs_tasnet.runtime_variant": VERSION,
        "hs_tasnet.parent_graph_sha256": hashlib.sha256(parent.SerializeToString()).hexdigest(),
        "hs_tasnet.integer_exporter_sha256": sha(__file__),
        "hs_tasnet.integer_weight_precision": "Sixteen S8 symmetric reduced-range matrices [-64,64]",
        "hs_tasnet.floating_additional_layers": "phase factors, fusion refinement, temporal attention, branch GRU biases/nonlinearities",
        "hs_tasnet.additional_branch_output_integer_projections": "2",
        "hs_tasnet.deployment_quality_status": "unmeasured; retained source FP32 score is not this graph's quality"})
    helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"version": VERSION, "projections": proof,
        "only_two_output_products_and_their_weight_casts_replaced": True,
        "all_other_nodes_and_initializers_byte_exact": True, "public_interface_changed": False,
        "quality_measured": False, "native_host_qualified": False}



def make_reference(native, ten_graph, ten_conversion, fourteen_graph, fourteen_conversion, candidate, conversion):
    from onnx import numpy_helper as nh
    from .integer_branch_gru import make_reference as parent_reference
    from .integer import SignedProjection
    reference, independent = parent_reference(native, ten_graph, ten_conversion, fourteen_graph, fourteen_conversion)
    stored = {v.name: nh.to_array(v) for v in candidate.graph.initializer}
    prefixes = {v["module"]: v["initializer"] for v in conversion["projections"]}
    require(set(prefixes) == set(TARGETS.values()), "Unexpected output projections")
    for name in TARGETS.values():
        original = getattr(native, name)
        require(original.bias is None and tuple(original.weight.shape) == (500, 500), "Source output projection changed")
        projection = SignedProjection(original.weight, None, stored, prefixes[name])
        setattr(reference.model, name, projection)
        independent.append({"module": name, **projection.proof, "bias": None})
    require(len(independent) == 16, "Expected sixteen independently reconstructed integer projections")
    return reference.eval().requires_grad_(False), independent
