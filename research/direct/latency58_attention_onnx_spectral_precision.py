"""Small spectral arithmetic islands; every checkpoint tensor stays FP32."""
import copy

from research.direct.run_latency58_quality import require, sha


def analysis_dft(graph):
    from onnx import helper, TensorProto as T
    nodes, changed = [], []
    for original in graph.graph.node:
        inverse = next((a.i for a in original.attribute if a.name == "inverse"), 0)
        if original.op_type != "DFT" or inverse:
            nodes.append(original)
            continue
        require(len(original.input) == 2 and len(original.output) == 1,
                "Require analysis DFT with a fixed transform length")
        node = copy.deepcopy(original)
        node.input[0] = original.input[0] + "__analysis64"
        node.output[0] = original.output[0] + "__analysis64"
        nodes.extend((helper.make_node("Cast", [original.input[0]], [node.input[0]],
                                      name=node.name + "/input64", to=T.DOUBLE),
                      node,
                      helper.make_node("Cast", [node.output[0]], [original.output[0]],
                                       name=node.name + "/output32", to=T.FLOAT)))
        changed.append(original.name)
    require(len(changed) == 1, "Require exactly one forward DFT")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return {"analysis_dft_nodes": changed, "inverse_dft_unchanged": True}


def magnitude_log1p(graph):
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto as T
    producers = {v: n for n in graph.graph.node for v in n.output}
    constants = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Constant":
            value = next((a.t for a in node.attribute if a.name == "value"), None)
            if value is not None:
                constants[node.output[0]] = numpy_helper.to_array(value)
    nodes, changed = [], []
    for original in graph.graph.node:
        if original.op_type != "Log" or not original.name.startswith("/spec_encode/"):
            nodes.append(original)
            continue
        addition = producers[original.input[0]]
        require(addition.op_type == "Add" and len(addition.input) == 2,
                "Require exported log1p = Log(Add(x, 1))")
        units = [v for v in addition.input if v in constants
                 and constants[v].size == 1 and float(constants[v].reshape(-1)[0]) == 1.0]
        require(len(units) == 1, "Require one scalar unit addition")
        argument = next(v for v in addition.input if v != units[0])
        prefix = original.name + "/log1p64/"
        nodes.extend((helper.make_node("Cast", [argument], [prefix + "x"],
                                      name=prefix + "input64", to=T.DOUBLE),
                      helper.make_node("Constant", [], [prefix + "one"],
                                       name=prefix + "constant",
                                       value=numpy_helper.from_array(np.asarray(1., np.float64))),
                      helper.make_node("Add", [prefix + "x", prefix + "one"], [prefix + "sum"],
                                       name=prefix + "add"),
                      helper.make_node("Log", [prefix + "sum"], [prefix + "log"],
                                       name=original.name),
                      helper.make_node("Cast", [prefix + "log"], list(original.output),
                                       name=prefix + "output32", to=T.FLOAT)))
        changed.append(original.name)
    require(len(changed) == 1, "Require one magnitude log1p")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return {"magnitude_log1p_nodes": changed, "argument_computed_in_original_fp32": True}


def transform(graph, variant):
    import onnx
    initializers = {v.name: v.SerializeToString() for v in graph.graph.initializer}
    require(variant in ("precise_analysis_dft", "precise_magnitude_log1p", "precise_dft_and_log1p"),
            "Unknown spectral arithmetic variant")
    proof = {}
    if variant in ("precise_analysis_dft", "precise_dft_and_log1p"):
        proof.update(analysis_dft(graph))
    if variant in ("precise_magnitude_log1p", "precise_dft_and_log1p"):
        proof.update(magnitude_log1p(graph))
    require(initializers == {v.name: v.SerializeToString() for v in graph.graph.initializer},
            "Spectral arithmetic changed checkpoint initializer bytes")
    proof["all_initializers_byte_exact"] = True
    props = {v.key: v.value for v in graph.metadata_props}
    props.update({"hs_tasnet.runtime_variant": "temporal_attention-" + variant + "-v1",
                  "hs_tasnet.analysis_implementation": variant,
                  "hs_tasnet.analysis_transform_source_sha256": sha(__file__),
                  "hs_tasnet.onnx_arithmetic_precision": "FP32 with selected FP64 spectral islands; float32 public I/O and checkpoint tensors"})
    onnx.helper.set_model_props(graph, props)
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph, full_check=True)
    return proof


def build(model, payload, training, checkpoint, quality_path, *, variant):
    from research.direct.latency58_best_onnx_export import build as original_build
    wrapper, graph = original_build(model, payload, training, checkpoint, quality_path)
    transform(graph, variant)
    return wrapper, graph
