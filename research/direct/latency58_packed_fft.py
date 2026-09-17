"""Pack pairs of real transforms into complex DFTs in the precise C204 graph.

The synthesis identity is IFFT(X + iY) = x + iy for Hermitian X and Y.
The optional analysis identity recovers the two real spectra from FFT(x+iy).
All graph weights, public state shapes and physical sample coordinates stay
unchanged. Floating-point operation order changes and needs independent checks.
"""
from __future__ import annotations

import copy

import numpy as np

from research.direct.run_latency58_quality import require


def rewrite(source, *, analysis=False):
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh

    graph = copy.deepcopy(source)
    by_name = {node.name: node for node in graph.graph.node}
    require(len(by_name) == len(graph.graph.node)
            and [(x.domain, x.version) for x in graph.opset_import] == [("", 17)],
            "Require the reviewed single-hop graph and opset")
    names = ("/Gather_2", "/Mul_9", "/Concat_2", "/DFT_1", "/Gather_3")
    removed = set(names)
    positive = by_name["/Mul_8"]
    require(list(positive.input) == ["/Reshape_4_output_0", "/Constant_22_output_0"]
            and positive.op_type == "Mul", "Synthesis endpoint projection moved")
    endpoint = nh.to_array(next(a.t for a in by_name["/Constant_22"].attribute if a.name == "value"))
    expected = np.ones((513, 2), dtype=np.float32)
    expected[(0, -1), 1] = 0
    require(np.array_equal(endpoint, expected), "Require exactly real DC and Nyquist values")
    reverse = nh.to_array(next(a.t for a in by_name["/Constant_23"].attribute if a.name == "value"))
    require(np.array_equal(reverse, np.arange(511, 0, -1, dtype=np.int64)), "Conjugate-mirror indices differ")
    require(list(by_name["/Gather_2"].input) == [positive.output[0], "/Constant_23_output_0"]
            and list(by_name["/Mul_9"].input) == ["/Gather_2_output_0", "/Constant_24_output_0"]
            and list(by_name["/Concat_2"].input) == [positive.output[0], "/Mul_9_output_0"]
            and list(by_name["/DFT_1"].input) == ["/Concat_2_output_0", "/Constant_25_output_0"]
            and list(by_name["/Gather_3"].input) == ["/DFT_1_output_0", "/Constant_26_output_0"],
            "Synthesis DFT topology changed")
    for name in names[:-1]:
        consumers = [n.name for n in graph.graph.node if by_name[name].output[0] in n.input]
        require(len(consumers) == 1 and consumers[0] in removed, "A removed FFT value has an external consumer")
    attributes = {a.name: a.i for a in by_name["/DFT_1"].attribute}
    require(attributes == {"axis": 1, "inverse": 1, "onesided": 0}, "Unexpected inverse DFT")

    prefix = "/packed_fft/"
    constants, nodes = [], []
    def constant(name, value):
        name = prefix + name
        constants.append(nh.from_array(np.asarray(value), name))
        return name
    def node(kind, name, inputs, *, output=None, **attributes):
        name = prefix + name
        output = name + "_output" if output is None else output
        nodes.append(h.make_node(kind, inputs, [output], name=name, **attributes))
        return output
    zero = constant("zero", np.array(0, np.int64))
    one = constant("one", np.array(1, np.int64))
    swap = constant("swap_components", np.array([1, 0], np.int64))
    rotate = constant("rotate_i", np.array([-1, 1], np.float32))
    conjugate = constant("conjugate", np.array([1, -1], np.float32))
    pair_shape = constant("pair_shape", np.array([4, 2, 513, 2], np.int64))
    output_shape = constant("output_shape", np.array([8, 1024], np.int64))
    pairs = node("Reshape", "synthesis_pairs", [positive.output[0], pair_shape])
    left = node("Gather", "synthesis_left", [pairs, zero], axis=1)
    right = node("Gather", "synthesis_right", [pairs, one], axis=1)
    right = node("Gather", "synthesis_swap", [right, swap], axis=2)
    imaginary_right = node("Mul", "synthesis_times_i", [right, rotate])
    plus = node("Add", "synthesis_positive", [left, imaginary_right])
    difference = node("Sub", "synthesis_difference", [left, imaginary_right])
    mirror = node("Gather", "synthesis_reverse", [difference, "/Constant_23_output_0"], axis=1)
    mirror = node("Mul", "synthesis_conjugate", [mirror, conjugate])
    full = node("Concat", "synthesis_full", [plus, mirror], axis=1)
    transformed = node("DFT", "synthesis_dft", [full, "/Constant_25_output_0"], axis=1, inverse=1, onesided=0)
    separated = node("Transpose", "synthesis_unpack", [transformed], perm=[0, 2, 1])
    node("Reshape", "synthesis_output", [separated, output_shape], output=by_name["/Gather_3"].output[0])
    synthesis_nodes = list(nodes)
    nodes.clear()
    if analysis:
        original = by_name["/DFT"]
        require({a.name: a.i for a in original.attribute} == {"axis": 1, "inverse": 0, "onesided": 1},
                "Unexpected analysis DFT")
        shape = constant("analysis_complex_shape", np.array([1, 1024, 2], np.int64))
        select = constant("positive_indices", np.arange(513, dtype=np.int64))
        reverse = constant("analysis_reverse_indices", (-np.arange(513, dtype=np.int64)) % 1024)
        conjugate64 = constant("conjugate64", np.array([1, -1], np.float64))
        half = constant("half64", np.array(.5, np.float64))
        rotate_negative = constant("rotate_negative_i64", np.array([1, -1], np.float64))
        # The input is double [2,1024,1], channels become real/imag components.
        joined = node("Transpose", "analysis_interleave", [original.input[0]], perm=[2, 1, 0])
        joined = node("Reshape", "analysis_complex", [joined, shape])
        transformed = node("DFT", "analysis_dft", [joined, original.input[1]], axis=1, inverse=0, onesided=0)
        direct = node("Gather", "analysis_positive", [transformed, select], axis=1)
        reflected = node("Gather", "analysis_reverse", [transformed, reverse], axis=1)
        reflected = node("Mul", "analysis_conjugate", [reflected, conjugate64])
        left = node("Add", "analysis_left_sum", [direct, reflected])
        left = node("Mul", "analysis_left", [left, half])
        right = node("Sub", "analysis_right_difference", [direct, reflected])
        right = node("Gather", "analysis_right_swap", [right, swap], axis=2)
        right = node("Mul", "analysis_right_rotation", [right, rotate_negative])
        right = node("Mul", "analysis_right", [right, half])
        node("Concat", "analysis_output", [left, right], axis=0, output=original.output[0])
        removed.add("/DFT")
    analysis_nodes = list(nodes)
    rewritten = []
    for original in graph.graph.node:
        if original.name == "/DFT" and analysis:
            rewritten.extend(analysis_nodes)
        elif original.name == "/Gather_3":
            rewritten.extend(synthesis_nodes)
        elif original.name not in removed:
            rewritten.append(original)
    require({n.name: n.SerializeToString() for n in rewritten if n.name in by_name}
            == {n.name: n.SerializeToString() for n in source.graph.node if n.name not in removed},
            "Unrelated nodes changed")
    require(not ({t.name for t in constants} & {t.name for t in graph.graph.initializer}), "Constant collision")
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    graph.graph.initializer.extend(constants)
    del graph.graph.value_info[:]
    require([t.SerializeToString() for t in graph.graph.initializer[:len(source.graph.initializer)]]
            == [t.SerializeToString() for t in source.graph.initializer], "Original initializer bytes changed")
    require([v.SerializeToString() for v in graph.graph.input] == [v.SerializeToString() for v in source.graph.input]
            and [v.SerializeToString() for v in graph.graph.output] == [v.SerializeToString() for v in source.graph.output],
            "Public graph ABI changed")
    metadata = {"source." + x.key: x.value for x in source.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "c204-precise-int8-packed-" + ("both" if analysis else "synthesis"),
                     "hs_tasnet.native_host_qualified": "false"})
    h.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"packed_analysis": analysis, "synthesis_complex_transforms_before_after": [8, 4],
                   "analysis_complex_transforms_before_after": [2, 1] if analysis else [2, 2],
                   "original_initializers_byte_exact": True, "unrelated_nodes_byte_exact": True,
                   "public_inputs_outputs_byte_exact": True, "endpoint_projection_verified": True,
                   "added_audio_buffering_samples": 0, "removed_nodes": sorted(removed),
                   "added_nodes": [n.name for n in (*analysis_nodes, *synthesis_nodes)],
                   "operation_order_changed": True, "native_host_qualified": False}
