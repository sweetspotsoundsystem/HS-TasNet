"""Bounded recurrence execution variants; checkpoint and streaming ABI stay fixed."""
import copy
import hashlib
from research.direct.run_latency58_quality import require, sha


def fuse_current(graph):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    prefix = "/fusion_branch/"
    removed = [node for node in graph.graph.node if node.name.startswith(prefix)]
    from collections import Counter
    operations = Counter(node.op_type for node in removed)
    require(operations["Sigmoid"] == 4 and operations["Tanh"] == 2
            and operations["Gemm"] + operations["MatMul"] == 4,
            "Expected the four projections and six nonlinearities of two GRU layers: " + str(operations))
    produced = {value for node in removed for value in node.output}
    external_inputs = {value for node in removed for value in node.input if value not in produced}
    weights = {value.name: value for value in graph.graph.initializer
               if value.name.startswith("model.fusion_branch.")}
    def axis(node):
        return next((a.i for a in node.attribute if a.name == "axis"), 0)
    inputs = [node.input[0] for node in removed if node.op_type == "Gather"
              and node.input[0] in external_inputs and axis(node) == 1]
    hidden = {node.input[0] for node in removed if node.op_type == "Gather"
              and node.input[0] in external_inputs and axis(node) == 0}
    require(len(inputs) == 1 and len(hidden) == 1 and len(weights) == 8
            and len(external_inputs - weights.keys()) == 3, "Recurrence boundary changed")
    feature_input, hidden_input = inputs[0], next(iter(hidden))
    retained = [node for node in graph.graph.node if not node.name.startswith(prefix)]
    observed = {value for node in retained for value in node.input if value in produced}
    producer = {value: node for node in removed for value in node.output}
    stacked_outputs = [value for value in observed if producer[value].op_type == "Concat"]
    frame_outputs = [value for value in observed if producer[value].op_type == "Unsqueeze"]
    shared_constants = [node for node in removed if node.op_type == "Constant" and any(value in observed for value in node.output)]
    constant_outputs = {value for node in shared_constants for value in node.output}
    require(len(observed - constant_outputs) == 2 and len(stacked_outputs) == len(frame_outputs) == 1,
            "Recurrence output boundary changed")
    stack_output, frame_output = stacked_outputs[0], frame_outputs[0]
    original_retained = [node.SerializeToString() for node in retained]
    initializers = [value for value in graph.graph.initializer if value.name not in weights]
    original_nonrecurrent = {value.name: value.SerializeToString() for value in initializers}
    nodes, proof = [], []

    def constant(name, value):
        key = "/fused_gru/" + name
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), key))
        return key

    indices = [constant("layer0", [0]), constant("layer1", [1])]
    layer_input = feature_input  # [batch=1, time=1, feature=1000]
    for layer in range(2):
        name = f"/fused_gru/layer{layer}/"
        converted = {}
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            source = numpy_helper.to_array(weights[f"model.fusion_branch.{kind}_l{layer}"])
            require(source.dtype == np.float32 and source.shape == (
                (3000, 1000) if kind.startswith("weight") else (3000,)), "Weight ABI changed")
            reordered = np.concatenate((source[1000:2000], source[:1000], source[2000:]), axis=0)
            round_trip = np.concatenate((reordered[1000:2000], reordered[:1000], reordered[2000:]), axis=0)
            require(source.tobytes() == round_trip.tobytes(), "GRU weight permutation lost bytes")
            proof.append({"layer": layer, "tensor": kind,
                          "original_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                          "reordered_sha256": hashlib.sha256(reordered.tobytes()).hexdigest(),
                          "inverse_permutation_bit_exact": True})
            converted[kind] = reordered
        for key, values in (("W", converted["weight_ih"]), ("R", converted["weight_hh"]),
                            ("B", np.concatenate((converted["bias_ih"], converted["bias_hh"])))):
            initializers.append(numpy_helper.from_array(np.ascontiguousarray(values[None]), name + key))
        nodes.append(helper.make_node("Gather", [hidden_input, indices[layer]],
                                      [name + "initial_h"], name=name + "gather", axis=0))
        # Both sequence and batch axes are statically one. Y_h therefore also
        # serves directly as the next layer's [seq, batch, feature] input.
        nodes.append(helper.make_node("GRU", [layer_input, name + "W", name + "R", name + "B", "",
                                               name + "initial_h"], ["", name + "next_h"],
                                      name=name + "GRU", hidden_size=1000, linear_before_reset=1,
                                      direction="forward", layout=0))
        layer_input = name + "next_h"
    nodes.append(helper.make_node("Concat", ["/fused_gru/layer0/next_h", "/fused_gru/layer1/next_h"],
                                  [stack_output], name="/fused_gru/stack", axis=0))
    nodes.append(helper.make_node("Identity", [layer_input], [frame_output],
                                  name="/fused_gru/output"))
    rewritten, inserted = [], False
    for node in graph.graph.node:
        if node.name.startswith(prefix):
            if not inserted:
                rewritten.extend(shared_constants)
                rewritten.extend(nodes)
                inserted = True
        else:
            rewritten.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    require([node.SerializeToString() for node in graph.graph.node
             if not node.name.startswith(("/fused_gru/", prefix))] == original_retained,
            "Non-recurrent node bytes changed")
    require(all(original_nonrecurrent[value.name] == value.SerializeToString()
                for value in graph.graph.initializer if value.name in original_nonrecurrent),
            "Non-recurrent initializer changed")
    props = {item.key: item.value for item in graph.metadata_props}
    props.update({"hs_tasnet.gru_implementation": "ONNX GRU; two layers; linear_before_reset=1; zrh",
                  "hs_tasnet.gru_transform_source_sha256": sha(__file__)})
    helper.set_model_props(graph, props)
    onnx.checker.check_model(graph, full_check=True)
    return {"removed_nodes": len(removed), "replacement_nodes": len(nodes),
            "shared_constant_nodes_preserved": [node.name for node in shared_constants],
            "nonrecurrent_nodes_and_initializers_byte_exact": True, "weight_permutations": proof}


def gemm_variant(graph, variant):
    from onnx import helper, TensorProto as T
    original_initializers = {value.name: value.SerializeToString() for value in graph.graph.initializer}
    nodes, changed = [], []
    for original in graph.graph.node:
        if not original.name.startswith("/fusion_branch/") or original.op_type != "Gemm":
            nodes.append(original)
            continue
        node = copy.deepcopy(original)
        require(len(node.input) == 3 and len(node.output) == 1, "Require biased one-frame GEMM")
        output = node.output[0]
        if variant == "split_gru_bias":
            bias = node.input[2]
            del node.input[2:]
            node.output[0] = output + "__unbiased"
            nodes.extend((node, helper.make_node("Add", [node.output[0], bias], [output],
                                                 name=node.name + "/separate_bias")))
        else:
            require(variant == "precise_gru_gemm", "Unknown GEMM variant")
            for index, value in enumerate(original.input):
                converted = node.name + "/input64_" + str(index)
                nodes.append(helper.make_node("Cast", [value], [converted],
                                              name=converted, to=T.DOUBLE))
                node.input[index] = converted
            node.output[0] = output + "__gemm64"
            nodes.extend((node, helper.make_node("Cast", [node.output[0]], [output],
                                                 name=node.name + "/output32", to=T.FLOAT)))
        changed.append(original.name)
    require(len(changed) == 4, "Require exactly four recurrent GEMMs")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.value_info[:]
    require(original_initializers == {v.name: v.SerializeToString() for v in graph.graph.initializer},
            "GEMM transformation changed initializer bytes")
    return {"changed_gemm_nodes": changed, "all_initializers_byte_exact": True}


def transform(graph, variant):
    import onnx
    if variant == "fused_gru":
        proof = fuse_current(graph)
        description = "ONNX GRU; two layers; linear_before_reset=1; zrh"
    else:
        proof = gemm_variant(graph, variant)
        description = ("FP32 GEMM then separate bias Add" if variant == "split_gru_bias"
                       else "four GRU GEMMs in FP64, each rounded to FP32 before gate arithmetic")
    props = {item.key: item.value for item in graph.metadata_props}
    props.update({"hs_tasnet.runtime_variant": "temporal_attention-" + variant + "-v1",
                  "hs_tasnet.gru_implementation": description,
                  "hs_tasnet.gru_transform_source_sha256": sha(__file__),
                  "hs_tasnet.onnx_arithmetic_precision": "FP32 with four FP64 recurrent GEMM islands" if variant == "precise_gru_gemm" else "FP32"})
    onnx.helper.set_model_props(graph, props)
    onnx.checker.check_model(graph, full_check=True)
    return proof


def build(model, payload, training, checkpoint, quality_path, *, variant):
    from research.direct.latency58_best_onnx_export import build as original_build
    wrapper, graph = original_build(model, payload, training, checkpoint, quality_path)
    transform(graph, variant)
    return wrapper, graph
