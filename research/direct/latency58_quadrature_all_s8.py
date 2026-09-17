"""Requantize all ten existing integer projections to reduced-range S8 weights.

Only weight, scale and zero-point initializers change. All graph operations,
activation quantizers, precision boundaries and public states stay identical.
The complete independent reference is first checked against the saved unsigned
graph, then each signed matrix is reconstructed independently from FP32 weights.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from research.direct.latency58_quadrature_encoder_s8 import BASELINE_SHA, digest
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256

VERSION = "quadrature-ten-u8s8-reduced-v1"


def source_matrices(native):
    modules = {name: getattr(native, name) for name in
               ("conv_encode", "basis_to_embed", "spec_encode", "to_spec_masks", "to_waveform_masks")}
    matrices = {name: np.ascontiguousarray(module.weight.detach().numpy().reshape(module.weight.shape[0], -1).T)
                for name, module in modules.items()}
    matrices.update({f"fusion_branch.weight_{kind}_l{layer}": np.ascontiguousarray(
        getattr(native.fusion_branch, f"weight_{kind}_l{layer}").detach().numpy().T)
        for layer in range(2) for kind in ("ih", "hh")})
    matrices["spec_encode.magnitude_projection"] = np.ascontiguousarray(
        native.spec_encode.magnitude_projection.weight.detach().numpy().T)
    require(len(matrices) == 10 and all(v.dtype == np.float32 and v.ndim == 2 for v in matrices.values()),
            "Expected ten FP32 source matrices")
    return matrices


def build(native, original, original_proof):
    import onnx
    from onnx import helper, numpy_helper as nh, TensorProto as T
    from onnxruntime.quantization.quant_utils import quantize_data
    require(hashlib.sha256(original.SerializeToString()).hexdigest() == BASELINE_SHA,
            "Use the checked saved unsigned quadrature graph")
    fingerprint = state_sha256(native.state_dict())
    props = {v.key: v.value for v in original.metadata_props}
    require(props["hs_tasnet.source_model_state_sha256"] == fingerprint, "Wrong source checkpoint")
    matrices = source_matrices(native)
    entries = original_proof["projections"]
    require(len(entries) == 10 and {v["module"] for v in entries} == set(matrices),
            "Unsigned proof must map all ten projections exactly once")
    replacements, proofs = {}, []
    stored = {t.name: nh.to_array(t) for t in original.graph.initializer}
    for entry in entries:
        weight, prefix = matrices[entry["module"]], entry["initializer"]
        require(digest(weight) == entry["source_matrix_sha256"]
                and digest(stored[prefix + "_quantized"]) == entry["quantized_matrix_sha256"],
                "Source matrix or saved unsigned quantization differs")
        quantized = [quantize_data(np.ascontiguousarray(column), T.INT8, symmetric=True, reduce_range=True)
                     for column in weight.T]
        zeros = np.asarray([v[0] for v in quantized], np.int8).reshape(-1)
        scales = np.asarray([v[1] for v in quantized], np.float32).reshape(-1)
        values = np.ascontiguousarray(np.stack([v[2] for v in quantized], axis=1), dtype=np.int8)
        require(np.all(zeros == 0) and np.abs(values.astype(np.int16)).max() <= 64
                and weight.shape[0] * 255 * 64 < 2**31, "Signed pair or dot product can overflow")
        for suffix, value in (("_quantized", values), ("_scale", scales), ("_zero_point", zeros)):
            name = prefix + suffix
            require(name in stored and name not in replacements, "Missing or repeated quantization initializer")
            replacements[name] = value
        proofs.append({"module": entry["module"], "initializer": prefix, "shape": list(weight.shape),
                       "source_matrix_sha256": digest(weight), "s8_weights_sha256": digest(values),
                       "s8_scales_sha256": digest(scales), "weight_zero_points_all_zero": True,
                       "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64,
                       "worst_centered_dot_absolute_sum": weight.shape[0] * 255 * 64})
    graph = copy.deepcopy(original)
    before = {v.name: v.SerializeToString() for v in graph.graph.initializer}
    for tensor in graph.graph.initializer:
        if tensor.name in replacements:
            tensor.CopyFrom(nh.from_array(replacements[tensor.name], tensor.name))
    helper.set_model_props(graph, {**props, "hs_tasnet.runtime_variant": VERSION,
        "hs_tasnet.integer_weight_precision": "Ten S8 symmetric reduced-range matrices [-64,64]",
        "hs_tasnet.source_integer_graph_sha256": BASELINE_SHA, "hs_tasnet.native_host_qualified": "false"})
    onnx.checker.check_model(graph, full_check=True)
    require(len(replacements) == 30
            and all(v.SerializeToString() == before[v.name] for v in graph.graph.initializer if v.name not in replacements)
            and all(a.SerializeToString() == b.SerializeToString() for a, b in
                    zip(original.graph.node, graph.graph.node, strict=True))
            and state_sha256(native.state_dict()) == fingerprint, "Unrelated graph or model content changed")
    return graph, {"version": VERSION, "baseline_graph_sha256": BASELINE_SHA,
                   "source_model_state_sha256": fingerprint, "projections": proofs,
                   "changed_initializers": list(replacements), "all_nodes_and_other_initializers_unchanged": True,
                   "additional_operations": 0, "additional_stream_states": 0,
                   "additional_audio_buffering_samples": 0, "quality_measured": False}


def make_reference(native, original, candidate, original_proof):
    import torch
    from onnx import numpy_helper as nh
    from research.direct.latency58_quadrature_magint8 import make_reference as original_reference
    reference, unsigned_proofs = original_reference(native, original, original_proof)
    m = reference.model
    projections = {"conv_encode": m.conv_encode.projection, "basis_to_embed": m.basis_to_embed.projection,
                   "spec_encode": m.spec_encode.original_projection,
                   "spec_encode.magnitude_projection": m.spec_encode.magnitude_projection,
                   "to_spec_masks": m.to_spec_masks, "to_waveform_masks": m.to_waveform_masks}
    projections.update({f"fusion_branch.weight_{kind}_l{layer}": getattr(m.fusion_branch, maps)[layer]
                        for maps, kind in (("input_maps", "ih"), ("hidden_maps", "hh")) for layer in range(2)})
    stored = {v.name: nh.to_array(v) for v in candidate.graph.initializer}
    independent = []
    require(len(unsigned_proofs) == len(projections) == 10, "Missing unsigned reference projection")
    for proof in unsigned_proofs:
        name, prefix = proof["module"], proof["initializer"]
        # Use original row-major PyTorch parameters, separately from builder matrices.
        if name.startswith("fusion_branch.weight_"):
            tensor = getattr(native.fusion_branch, name.split(".", 1)[1])
        else:
            owner = native
            for component in name.split("."):
                owner = getattr(owner, component)
            tensor = owner.weight
        weight = tensor.detach().numpy().reshape(tensor.shape[0], -1)
        maximum = np.max(np.abs(weight), axis=1)
        scales = ((maximum - (-maximum)).astype(np.float64) / np.float64(128)).astype(np.float32)
        scales = np.where(scales < np.finfo(np.float32).tiny, np.float32(1), scales)
        quantized = np.clip(np.rint(weight / scales[:, None]), -64, 64).astype(np.int8)
        require(np.array_equal(quantized.T, stored[prefix + "_quantized"])
                and np.array_equal(scales, stored[prefix + "_scale"])
                and np.array_equal(np.zeros(weight.shape[0], np.int8), stored[prefix + "_zero_point"]),
                "Independent signed reconstruction differs: " + name)
        projection = projections[name]
        with torch.no_grad():
            projection.weight.copy_(torch.from_numpy(np.ascontiguousarray(quantized.T.astype(np.int32))))
            projection.scale.copy_(torch.from_numpy(scales.astype(np.float64)))
        independent.append({"module": name, "initializer": prefix, "source_float_sha256": digest(weight),
                            "quantized_matrix_sha256": digest(quantized.T),
                            "weight_scale_zero_point_bit_exact": True, "weight_precision": "S8 symmetric reduced",
                            "reference_arithmetic": "Independent NumPy range reconstruction and PyTorch int32 dots"})
    return reference.eval().requires_grad_(False), independent
