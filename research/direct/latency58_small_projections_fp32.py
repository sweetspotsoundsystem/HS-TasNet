"""Use FP32 for three measured small products, preserving trained weights."""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from research.direct.run_latency58_quality import require, sha

PARENT_SHA = "08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748"
TARGETS = {
    "/fusion_refine_reduce/MatMul": ("fusion_refine_reduce", "onnx::MatMul_745", (1000, 128)),
    "/fusion_refine_expand/MatMul": ("fusion_refine_expand", "onnx::MatMul_746", (128, 1000)),
    "/temporal_output/MatMul": ("temporal_output", "onnx::MatMul_764", (128, 1000)),
}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build(parent):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    require(hashlib.sha256(parent.SerializeToString()).hexdigest() == PARENT_SHA,
            "Require the exact current fused-QKV graph")
    graph = copy.deepcopy(parent)
    producers = {value: node for node in parent.graph.node for value in node.output}
    initializers = {value.name: value for value in parent.graph.initializer}
    replacements, removed, proof = {}, set(), []
    for node in parent.graph.node:
        if node.name not in TARGETS:
            continue
        module, name, shape = TARGETS[node.name]
        cast = producers[node.input[1]]
        require(node.op_type == "MatMul" and cast.op_type == "Cast" and list(cast.input) == [name]
                and sum(node.input[1] in n.input for n in parent.graph.node) == 1
                and sum(name in n.input for n in parent.graph.node) == 1,
                "Small-projection topology or sharing changed")
        weight = nh.to_array(initializers[name])
        require(weight.shape == shape and weight.dtype == np.float32, "Trained matrix layout changed")
        prefix = node.name + "/small_fp32"
        x, y = prefix + "/input", prefix + "/output"
        replacements[node.name] = [
            helper.make_node("Cast", [node.input[0]], [x], name=prefix + "/cast_input", to=T.FLOAT),
            helper.make_node("MatMul", [x, name], [y], name=prefix + "/matmul"),
            helper.make_node("Cast", [y], list(node.output), name=prefix + "/cast_output", to=T.DOUBLE),
        ]
        removed.add(cast.name)
        proof.append({"module": module, "initializer": name, "shape": list(shape),
                      "weight_sha256": digest(weight), "weights_changed": False,
                      "input_and_product_precision": "float32", "outer_precision": "float64"})
    require(set(replacements) == set(TARGETS) and len(proof) == 3, "Expected three small projections")
    nodes = [v for n in graph.graph.node if n.name not in removed for v in replacements.get(n.name, [n])]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    old_nodes = {n.name: n.SerializeToString() for n in parent.graph.node}
    require(all(n.SerializeToString() == old_nodes[n.name] for n in graph.graph.node if n.name in old_nodes)
            and [v.SerializeToString() for v in graph.graph.initializer]
                == [v.SerializeToString() for v in parent.graph.initializer]
            and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 17,
            "Unrelated operation or trained initializer changed")
    require([v.SerializeToString() for v in graph.graph.input] == [v.SerializeToString() for v in parent.graph.input]
            and [v.SerializeToString() for v in graph.graph.output] == [v.SerializeToString() for v in parent.graph.output],
            "Public interface changed")
    properties = {v.key: v.value for v in graph.metadata_props}
    properties.update({"hs_tasnet.runtime_variant": "seventeen-integer-three-small-fp32-projections-v1",
        "hs_tasnet.parent_graph_sha256": PARENT_SHA, "hs_tasnet.small_projection_exporter_sha256": sha(__file__),
        "hs_tasnet.small_projection_precision": "FP32 input, weight and product; FP64 surrounding arithmetic",
        "hs_tasnet.deployment_quality_status": "unmeasured; no deployment promotion"})
    helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"projections": proof, "all_initializers_byte_exact": True,
        "all_other_nodes_byte_exact": True, "public_interface_changed": False,
        "training_weights_changed": False, "quality_measured": False, "native_host_qualified": False}
