"""Quantize the four profiled branch GRU products in the released graph.

This changes inference numerics and needs new quality and native-host evidence.
Biases, nonlinearities, output projections and all public states retain the
parent graph's arithmetic. Existing ten integer projections are unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from torch import nn

from .helpers import require, sha

VERSION = "branch-memory-fourteen-u8s8-reduced-precise-v1"
TARGETS = {f"/{branch}/{node}_MatMul": f"{branch}.weight_{kind}_l0"
           for branch in ("spec_memory", "waveform_memory")
           for node, kind in (("Gemm", "ih"), ("Gemm_1", "hh"))}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build(parent, *, expected_parent_sha256="d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa"):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    from onnxruntime.quantization.quant_utils import quantize_data
    require(hashlib.sha256(parent.SerializeToString()).hexdigest()
            == expected_parent_sha256,
            "Start with the exact released graph")
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
        require(weight.shape == (500, 1500) and weight.dtype == np.float32
                and name == "model." + TARGETS[node.name], "Wrong branch matrix identity or layout")
        channels = [quantize_data(np.ascontiguousarray(c), T.INT8, symmetric=True, reduce_range=True)
                    for c in weight.T]
        zero = np.asarray([c[0] for c in channels], np.int8).reshape(-1)
        scale = np.asarray([c[1] for c in channels], np.float32).reshape(-1)
        quantized = np.ascontiguousarray(np.stack([c[2] for c in channels], axis=1), dtype=np.int8)
        require(np.all(zero == 0) and np.abs(quantized.astype(np.int16)).max() <= 64
                and weight.shape[0] * 255 * 64 < 2**31, "Integer accumulator range changed")
        for suffix, value in (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero)):
            additions.append(nh.from_array(value, name + suffix))
        prefix = node.name + "/branch_u8s8"
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
            "bias_policy": "Unchanged subsequent FP64 Add; bias is outside the new FP32 integer product"})
    require(len(proof) == 4 and set(replacements) == set(TARGETS), "Expected exactly four branch products")
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
            and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 14,
            "An unrelated node, original integer product or initializer changed")
    properties = {v.key: v.value for v in graph.metadata_props}
    for key in ("hs_tasnet.precise_node_count", "hs_tasnet.integer_exporter_sha256"):
        if key in properties:
            properties["parent." + key] = properties.pop(key)
    properties.update({"hs_tasnet.runtime_variant": VERSION,
        "hs_tasnet.parent_graph_sha256": hashlib.sha256(parent.SerializeToString()).hexdigest(),
        "hs_tasnet.integer_exporter_sha256": sha(__file__),
        "hs_tasnet.integer_weight_precision": "Fourteen S8 symmetric reduced-range matrices [-64,64]",
        "hs_tasnet.floating_additional_layers": "phase factors, fusion refinement, temporal attention, branch GRU biases/nonlinearities and output projections",
        "hs_tasnet.additional_branch_integer_projections": "4",
        "hs_tasnet.deployment_quality_status": "unmeasured; retained source FP32 score is not this graph's quality"})
    helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"version": VERSION, "projections": proof,
        "only_four_products_and_their_weight_casts_replaced": True,
        "all_other_nodes_and_initializers_byte_exact": True, "public_interface_changed": False,
        "quality_measured": False, "native_host_qualified": False}


def make_reference(native, parent, parent_conversion, candidate, conversion):
    from onnx import numpy_helper as nh
    from .integer import make_reference as parent_reference, SignedProjection
    reference, independent = parent_reference(native, parent, parent_conversion)
    stored = {v.name: nh.to_array(v) for v in candidate.graph.initializer}
    prefixes = {v["module"]: v["initializer"] for v in conversion["projections"]}

    class OneLayerIntegerGRU(nn.Module):
        def __init__(self, original, name):
            super().__init__()
            require(original.num_layers == 1 and original.hidden_size == original.input_size == 500,
                    "Branch GRU geometry changed")
            for kind in ("ih", "hh"):
                key = name + ".weight_" + kind + "_l0"
                projection = SignedProjection(getattr(original, "weight_" + kind + "_l0"),
                                              None, stored, prefixes[key])
                setattr(self, kind, projection)
                self.register_buffer("bias_" + kind, getattr(original, "bias_" + kind + "_l0").double().clone())
                independent.append({"module": key, **projection.proof, "bias_added_separately_in_fp64": True})

        def forward(self, x, hidden):
            require(x.shape == (1, 1, 500) and hidden.shape == (1, 1, 500), "Unexpected branch input")
            ir, iz, inn = (self.ih(x[:, 0]) + self.bias_ih).chunk(3, dim=-1)
            hr, hz, hn = (self.hh(hidden[0]) + self.bias_hh).chunk(3, dim=-1)
            reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            proposal = (inn + reset * hn).tanh()
            current = proposal + update * (hidden[0] - proposal)
            return current[:, None], current[None]

    for name in ("spec_memory", "waveform_memory"):
        setattr(reference.model, name, OneLayerIntegerGRU(getattr(reference.model, name), name))
    require(len(independent) == 14, "Expected fourteen independently reconstructed signed projections")
    return reference.eval().requires_grad_(False), independent
