"""One signed integer product for the fused PR #17 QKV projections."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from .integer_qkv_fusion import build as fuse, PARENT_SHA
from .helpers import require, sha

VERSION = "branch-memory-seventeen-fused-qkv-u8s8-reduced-precise-v1"
FUSED_SHA = "b575979838431b1745bfa8dfb4efc2dec5788fde5ffb7076dda4e81e7bc5746e"


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build(parent, *, expected_parent_sha256=PARENT_SHA):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    from onnxruntime.quantization.quant_utils import quantize_data

    fused, fusion_proof = fuse(parent, expected_parent_sha256=expected_parent_sha256)
    # Source-location metadata changes when the maintained exporter moves.
    # The parent digest plus the fusion transform's byte-preservation and
    # topology checks bind this intermediate without pinning metadata bytes.
    graph = copy.deepcopy(fused)
    node = next(n for n in fused.graph.node if n.name == "/attention_qkv_fusion/matmul")
    cast = next(n for n in fused.graph.node if n.name == "/attention_qkv_fusion/cast_weight64")
    name = "/attention_qkv_fusion/weight32"
    initializers = {v.name: v for v in fused.graph.initializer}
    weight = nh.to_array(initializers[name])
    require(node.op_type == "MatMul" and cast.op_type == "Cast" and cast.input[0] == name
            and weight.shape == (1000, 256) and weight.dtype == np.float32,
            "Fused QKV matrix identity changed")
    channels = [quantize_data(np.ascontiguousarray(c), T.INT8, symmetric=True, reduce_range=True)
                for c in weight.T]
    zero = np.asarray([c[0] for c in channels], np.int8).reshape(-1)
    scale = np.asarray([c[1] for c in channels], np.float32).reshape(-1)
    quantized = np.ascontiguousarray(np.stack([c[2] for c in channels], axis=1), dtype=np.int8)
    require(np.all(zero == 0) and np.abs(quantized.astype(np.int16)).max() <= 64
            and weight.shape[0] * 255 * 64 < 2**31, "Integer accumulator range changed")
    additions = [nh.from_array(value, name + suffix) for suffix, value in
                 (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero))]
    prefix = "/attention_qkv_u8s8"
    x, q, xs, xz, integer, f32, scales, y32 = [prefix + suffix for suffix in
        ("/input32", "/quantized", "/input_scale", "/input_zero", "/integer", "/float", "/scale", "/output32")]
    replacements = [
        helper.make_node("Cast", [node.input[0]], [x], name=prefix + "/cast_input32", to=T.FLOAT),
        helper.make_node("DynamicQuantizeLinear", [x], [q, xs, xz], name=prefix + "/quantize"),
        helper.make_node("MatMulInteger", [q, name + "_quantized", xz, name + "_zero_point"],
                         [integer], name=prefix + "/matmul"),
        helper.make_node("Cast", [integer], [f32], name=prefix + "/cast_product32", to=T.FLOAT),
        helper.make_node("Mul", [xs, name + "_scale"], [scales], name=prefix + "/scales"),
        helper.make_node("Mul", [f32, scales], [y32], name=prefix + "/dequantize"),
        helper.make_node("Cast", [y32], list(node.output), name=prefix + "/cast_output64", to=T.DOUBLE),
    ]
    rewritten = [v for n in graph.graph.node if n.name != cast.name
                 for v in (replacements if n.name == node.name else [n])]
    tensors = [v for v in graph.graph.initializer if v.name != name]
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*tensors, *additions])
    old_nodes = {n.name: n.SerializeToString() for n in fused.graph.node}
    require(all(n.SerializeToString() == old_nodes[n.name] for n in graph.graph.node if n.name in old_nodes)
            and all(v.SerializeToString() == initializers[v.name].SerializeToString()
                    for v in graph.graph.initializer if v.name in initializers)
            and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 17,
            "Unrelated fused-graph operation or initializer changed")
    require([v.SerializeToString() for v in graph.graph.input] == [v.SerializeToString() for v in parent.graph.input]
            and [v.SerializeToString() for v in graph.graph.output] == [v.SerializeToString() for v in parent.graph.output],
            "Public interface changed")
    properties = {v.key: v.value for v in graph.metadata_props}
    properties["hs_tasnet.parent_sixteen_exporter_sha256"] = properties["hs_tasnet.integer_exporter_sha256"]
    properties.update({"hs_tasnet.runtime_variant": VERSION,
        "hs_tasnet.parent_graph_sha256": expected_parent_sha256,
        "hs_tasnet.integer_exporter_sha256": sha(__file__),
        "hs_tasnet.integer_weight_precision": "Seventeen S8 symmetric reduced-range matrices [-64,64]",
        "hs_tasnet.floating_additional_layers": "phase factors, fusion refinement, temporal output, GRU biases/nonlinearities",
        "hs_tasnet.additional_attention_integer_products": "One fused QKV product; three column groups",
        "hs_tasnet.deployment_quality_status": "unmeasured; parent score does not score this graph"})
    helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"version": VERSION, "fusion": fusion_proof,
        "initializer": name, "shape": list(weight.shape),
        "source_matrix_sha256": digest(weight), "quantized_matrix_sha256": digest(quantized),
        "scale_sha256": digest(scale), "weight_zero_points_all_zero": True,
        "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64,
        "worst_centered_dot_absolute_sum": weight.shape[0] * 255 * 64,
        "all_other_fused_nodes_and_initializers_byte_exact": True,
        "public_interface_changed": False, "quality_measured": False, "native_host_qualified": False}
