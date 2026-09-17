"""Change only the saved quadrature encoder to reduced-range symmetric S8 weights."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256

BASELINE_SHA = "6cfcc9d9ad70473dcaf9ce822af413e3d01fc303820798b8b726976e0b08cfcf"
PREFIX = "model.conv_encode.weight"
VERSION = "quadrature-encoder-u8s8-reduced-v1"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build(native, original):
    import onnx
    from onnx import helper, numpy_helper as nh, TensorProto as T
    from onnxruntime.quantization.quant_utils import quantize_data
    original_bytes = original.SerializeToString()
    require(hashlib.sha256(original_bytes).hexdigest() == BASELINE_SHA, "Use the checked saved quadrature graph")
    fingerprint = state_sha256(native.state_dict())
    props = {v.key: v.value for v in original.metadata_props}
    require(props["hs_tasnet.source_model_state_sha256"] == fingerprint, "Graph belongs to another checkpoint")
    weight = np.ascontiguousarray(native.conv_encode.weight.detach().numpy().reshape(3000, 2048).T)
    quantized = [quantize_data(np.ascontiguousarray(column), T.INT8, symmetric=True, reduce_range=True)
                 for column in weight.T]
    zeros = np.asarray([v[0] for v in quantized], np.int8).reshape(3000)
    scales = np.asarray([v[1] for v in quantized], np.float32).reshape(3000)
    values = np.ascontiguousarray(np.stack([v[2] for v in quantized], axis=1), dtype=np.int8)
    require(np.all(zeros == 0) and np.abs(values.astype(np.int16)).max() <= 64,
            "Signed range does not prevent pair saturation")
    replacements = {PREFIX + suffix: value for suffix, value in
                    (("_quantized", values), ("_scale", scales), ("_zero_point", zeros))}
    graph = copy.deepcopy(original)
    before = {v.name: v.SerializeToString() for v in graph.graph.initializer}
    require(set(replacements) <= set(before), "Encoder quantization tensors are absent")
    for tensor in graph.graph.initializer:
        if tensor.name in replacements:
            tensor.CopyFrom(nh.from_array(replacements[tensor.name], tensor.name))
    helper.set_model_props(graph, {**props, "hs_tasnet.runtime_variant": VERSION,
                                  "hs_tasnet.encoder_weight_precision": "S8 symmetric reduced range [-64,64]",
                                  "hs_tasnet.source_integer_graph_sha256": BASELINE_SHA,
                                  "hs_tasnet.native_host_qualified": "false"})
    onnx.checker.check_model(graph, full_check=True)
    require(all(v.SerializeToString() == before[v.name] for v in graph.graph.initializer if v.name not in replacements)
            and all(a.SerializeToString() == b.SerializeToString() for a, b in
                    zip(original.graph.node, graph.graph.node, strict=True))
            and state_sha256(native.state_dict()) == fingerprint, "Conversion changed unrelated graph or model content")
    return graph, {"version": VERSION, "baseline_graph_sha256": BASELINE_SHA,
                   "source_model_state_sha256": fingerprint, "changed_initializers": list(replacements),
                   "all_nodes_and_other_initializers_unchanged": True,
                   "source_matrix_sha256": digest(weight), "s8_weights_sha256": digest(values),
                   "s8_scales_sha256": digest(scales), "weight_zero_points_all_zero": True,
                   "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64,
                   "worst_centered_dot_absolute_sum": 2048 * 255 * 64,
                   "additional_operations": 0, "additional_stream_states": 0,
                   "additional_audio_buffering_samples": 0, "quality_measured": False}


def make_reference(native, original, candidate, original_proof):
    import torch
    from onnx import numpy_helper as nh
    from research.direct.latency58_quadrature_magint8 import make_reference as original_reference
    # Reconstruct the complete checked unsigned reference, then independently
    # derive only the new signed matrix from the original FP32 checkpoint.
    reference, proofs = original_reference(native, original, original_proof)
    weight = native.conv_encode.weight.detach().numpy().reshape(3000, 2048)
    maximum = np.max(np.abs(weight), axis=1)
    scales = ((maximum - (-maximum)).astype(np.float64) / np.float64(128)).astype(np.float32)
    scales = np.where(scales < np.finfo(np.float32).tiny, np.float32(1), scales)
    quantized = np.clip(np.rint(weight / scales[:, None]), -64, 64).astype(np.int8)
    stored = {v.name: nh.to_array(v) for v in candidate.graph.initializer}
    require(np.array_equal(quantized.T, stored[PREFIX + "_quantized"])
            and np.array_equal(scales, stored[PREFIX + "_scale"])
            and np.array_equal(np.zeros(3000, np.int8), stored[PREFIX + "_zero_point"]),
            "Independent signed encoder reconstruction differs")
    projection = reference.model.conv_encode.projection
    with torch.no_grad():
        projection.weight.copy_(torch.from_numpy(np.ascontiguousarray(quantized.T.astype(np.int32))))
        projection.scale.copy_(torch.from_numpy(scales.astype(np.float64)))
    proof = {"module": "conv_encode", "initializer": PREFIX,
             "source_float_sha256": digest(weight), "quantized_matrix_sha256": digest(quantized.T),
             "weight_scale_zero_point_bit_exact": True, "weight_precision": "S8 symmetric reduced",
             "reference_arithmetic": "Independent scalar-range NumPy reconstruction and PyTorch int32 dot products"}
    require(len(proofs) == 10 and sum(v["module"] == "conv_encode" for v in proofs) == 1,
            "Expected exactly one encoder among ten independent integer projections")
    return reference.eval().requires_grad_(False), [proof if v["module"] == "conv_encode" else v for v in proofs]
