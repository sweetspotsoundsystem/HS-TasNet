"""Combine the reviewed rank-128 spectral head with nine U8U8 projections.

The small rank-reduction matrix stays FP64 before the output quantizer.
All eight other integer projections and the recurrent arithmetic are kept.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.direct.run_latency58_quality import require


class PreciseFactoredMask(nn.Module):
    def __init__(self, factors, stored):
        super().__init__()
        from research.direct.latency58_int8_reference import IntegerLinear
        from research.direct.latency58_int8_precise_float import PreciseProjection
        self.register_buffer("right", factors.right.detach().double().clone())
        integer = IntegerLinear(factors.left, factors.bias, stored, "/rank128_spec_mask/left")
        self.proof = integer.proof
        self.expand = PreciseProjection(integer)

    def forward(self, values):
        require(values.dtype == torch.float64 and values.device.type == "cpu", "Require CPU FP64")
        return self.expand(F.linear(values, self.right))


def build(native, float_graph, integer_graph):
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from research.direct.latency58_low_rank_mask import convert as factor
    from research.direct.latency58_conv_gemm import convert as convolutions
    from research.direct.latency58_int8_precise_core import rewrite, make_reference

    # Build the independently implemented eight unchanged projections from
    # the original checkpoint, before replacing the factored output head.
    reference, original_proofs = make_reference(native, integer_graph)
    candidate, graph = copy.deepcopy(native), copy.deepcopy(float_graph)
    factor_proof = factor(candidate, graph)
    convolution_proof = convolutions(graph, ["/conv_encode/Conv", "/basis_to_embed/Conv"])
    with tempfile.TemporaryDirectory(prefix="latency58-rank-int8-") as temporary:
        path = Path(temporary) / "quantized.onnx"
        quantize_dynamic(graph, path, op_types_to_quantize=["MatMul"], per_channel=True,
            reduce_range=False, weight_type=QuantType.QUInt8, use_external_data_format=False,
            nodes_to_exclude=["/rank128_spec_mask/reduce", "/MatMul"],
            extra_options={"WeightSymmetric": False, "MatMulConstBOnly": True})
        quantized = onnx.load(path, load_external_data=False)
    require(sum(n.op_type == "MatMulInteger" for n in quantized.graph.node) == 9,
            "Require eight original blocks and one factored output block")
    remaining = {(n.name, n.op_type) for n in quantized.graph.node if n.op_type in ("MatMul", "Gemm", "Conv")}
    require(remaining == {("/rank128_spec_mask/reduce", "MatMul"), ("/MatMul", "MatMul")},
            "Only the rank reduction and waveform decoder may stay floating")
    stored = {t.name: numpy_helper.to_array(t) for t in quantized.graph.initializer}
    original = {t.name: numpy_helper.to_array(t) for t in integer_graph.graph.initializer}
    retained_proofs = [p for p in original_proofs if p["initializer"] != "onnx::MatMul_363"]
    require(len(retained_proofs) == 8, "Wrong original projection inventory")
    for proof in retained_proofs:
        for suffix in ("_quantized", "_scale", "_zero_point"):
            name = proof["initializer"] + suffix
            require(np.array_equal(stored[name], original[name]), "Unrelated integer tensor changed: " + name)
    reference.model.to_spec_masks = PreciseFactoredMask(candidate.to_spec_masks, stored)
    require(np.array_equal(stored["/rank128_spec_mask/right"], candidate.to_spec_masks.right.numpy().T)
            and np.array_equal(stored["model.to_spec_masks.bias"], candidate.to_spec_masks.bias.numpy()),
            "Rank reduction or bias differs from the independent reference")
    transformed = rewrite(quantized)
    metadata = {p.key: p.value for p in transformed.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "c204-hop128-rank128-u8u8-precise-v1",
                     "hs_tasnet.native_host_qualified": "false",
                     "hs_tasnet.quantized_full14_quality_qualified": "false"})
    onnx.helper.set_model_props(transformed, metadata)
    onnx.checker.check_model(transformed, full_check=True)
    proof = {"factorization": factor_proof, "convolution_rewrites": convolution_proof,
             "independent_integer_proofs": [*retained_proofs, reference.model.to_spec_masks.proof],
             "eight_other_integer_projections_bit_exact": True,
             "rank_reduction_dtype": "float64", "integer_projection_count": 9,
             "additional_buffering_samples": 0}
    return transformed, reference.eval().requires_grad_(False), proof
