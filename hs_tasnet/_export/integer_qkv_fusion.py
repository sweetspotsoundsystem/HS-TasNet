"""Fuse PR #17 attention projections while preserving weights and precision."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from .helpers import require, sha

PARENT_SHA = "c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61"
VERSION = "branch-memory-sixteen-fused-qkv-precise-v1"
TARGETS = (
    ("/temporal_query/MatMul", "onnx::MatMul_753", 64),
    ("/temporal_key/MatMul", "onnx::MatMul_754", 64),
    ("/temporal_value/MatMul", "onnx::MatMul_755", 128),
)


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build(parent, *, expected_parent_sha256=PARENT_SHA):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh

    require(hashlib.sha256(parent.SerializeToString()).hexdigest() == expected_parent_sha256,
            "Require exact PR #17 graph")
    graph = copy.deepcopy(parent)
    nodes = {node.name: node for node in parent.graph.node}
    producers = {value: node for node in parent.graph.node for value in node.output}
    initializers = {value.name: value for value in parent.graph.initializer}
    removed, matrices, outputs, proofs = set(), [], [], []
    for name, expected_initializer, width in TARGETS:
        node = nodes[name]
        require(node.op_type == "MatMul" and len(node.input) == 2, "Attention projection changed")
        cast = producers[node.input[1]]
        require(cast.op_type == "Cast" and cast.input[0] == expected_initializer
                and next(a.i for a in cast.attribute if a.name == "to") == T.DOUBLE
                and sum(node.input[1] in n.input for n in parent.graph.node) == 1
                and sum(expected_initializer in n.input for n in parent.graph.node) == 1,
                "Matrix identity, FP64 cast or sharing changed")
        matrix = nh.to_array(initializers[expected_initializer])
        require(matrix.shape == (1000, width) and matrix.dtype == np.float32, "Matrix layout changed")
        matrices.append(matrix)
        outputs.append(node.output[0])
        proofs.append({"node": name, "initializer": expected_initializer,
                       "shape": list(matrix.shape), "source_matrix_sha256": digest(matrix)})
        removed.update((name, cast.name))

    query, key, value = (nodes[name] for name, _, _ in TARGETS)
    query_slice = producers[query.input[0]]
    require(query_slice.op_type == "Slice" and len(query_slice.input) == 5
            and query_slice.input[0] == key.input[0] == value.input[0]
            and [n.name for n in parent.graph.node if query.input[0] in n.input] == [query.name],
            "Query last-frame slice or shared key/value input changed")
    parameters = []
    for name in query_slice.input[1:]:
        constant = producers[name]
        require(constant.op_type == "Constant" and len(constant.attribute) == 1
                and constant.attribute[0].name == "value", "Slice constant changed")
        tensor = nh.to_array(constant.attribute[0].t)
        require(tensor.dtype == np.int64 and tensor.shape == (1,), "Slice parameter type changed")
        parameters.append(tensor.tolist())
    require(parameters == [[-1], [np.iinfo(np.int64).max], [1], [1]], "Expected last-frame temporal slice")
    removed.add(query_slice.name)
    combined = np.ascontiguousarray(np.concatenate(matrices, axis=1))
    start = 0
    for matrix in matrices:
        require(np.array_equal(combined[:, start:start + matrix.shape[1]], matrix), "Packed weight values changed")
        start += matrix.shape[1]
    prefix = "/attention_qkv_fusion"
    packed_name, double_name = prefix + "/weight32", prefix + "/weight64"
    result_name, query_all = prefix + "/all_features", prefix + "/query_all_frames"
    splits = prefix + "/split_lengths"
    replacements = [
        helper.make_node("Cast", [packed_name], [double_name], name=prefix + "/cast_weight64", to=T.DOUBLE),
        helper.make_node("MatMul", [key.input[0], double_name], [result_name], name=prefix + "/matmul"),
        helper.make_node("Split", [result_name, splits], [query_all, outputs[1], outputs[2]],
                         name=prefix + "/split", axis=-1),
        helper.make_node("Slice", [query_all, *query_slice.input[1:]], [outputs[0]],
                         name=prefix + "/query_last_frame"),
    ]
    rewritten = []
    for node in parent.graph.node:
        if node.name == query.name:
            rewritten.extend(replacements)
        elif node.name not in removed:
            rewritten.append(copy.deepcopy(node))
    removed_weights = {name for _, name, _ in TARGETS}
    retained_weights = [copy.deepcopy(v) for v in parent.graph.initializer if v.name not in removed_weights]
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*retained_weights, nh.from_array(combined, packed_name),
                                   nh.from_array(np.asarray([64, 64, 128], np.int64), splits)])
    del graph.graph.value_info[:]
    old_nodes = {n.name: n.SerializeToString() for n in parent.graph.node}
    require(all(n.SerializeToString() == old_nodes[n.name] for n in graph.graph.node if n.name in old_nodes)
            and all(v.SerializeToString() == initializers[v.name].SerializeToString()
                    for v in graph.graph.initializer if v.name in initializers)
            and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 16,
            "Unrelated node, initializer or integer product changed")
    require([v.SerializeToString() for v in graph.graph.input] == [v.SerializeToString() for v in parent.graph.input]
            and [v.SerializeToString() for v in graph.graph.output] == [v.SerializeToString() for v in parent.graph.output],
            "Public interface changed")
    properties = {v.key: v.value for v in graph.metadata_props}
    properties.update({"hs_tasnet.runtime_variant": VERSION,
        "hs_tasnet.parent_graph_sha256": expected_parent_sha256,
        "hs_tasnet.attention_fusion_exporter_sha256": sha(__file__),
        "hs_tasnet.attention_projection_fusion": "QKV columns concatenated; FP64 product and original query last-frame slice",
        "hs_tasnet.deployment_quality_status": "unmeasured; parent score does not score this graph"})
    helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"version": VERSION, "original_projections": proofs,
        "combined_shape": list(combined.shape), "combined_weight_sha256": digest(combined),
        "query_slice_parameters": parameters, "query_slice_commuted_with_feature_projection": True,
        "stored_weight_values_unchanged": True, "fp64_product_precision_unchanged": True,
        "all_other_nodes_and_initializers_byte_exact": True, "public_interface_changed": False,
        "floating_reduction_order_may_differ": True, "quality_measured": False, "native_host_qualified": False}
