"""Quantize the magnitude projection in addition to the nine large projections.

This is a separate approximate inference candidate. Both phase factors remain
FP32. No graph or intermediate floating model is written by this module.
"""
import copy
import hashlib

import numpy as np
import torch
from torch import nn

from research.direct.run_latency58_quality import require


def build(native, *, return_baseline=False):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper
    from onnxruntime.quantization.quant_utils import quantize_data
    from research.direct.latency58_quadrature_int8 import build as base_build
    from research.direct.latency58_int8_precise_ten import rewrite
    from research.direct.train_latency58 import state_sha256
    fingerprint = state_sha256(native.state_dict())
    float_graph, graph, previous, inherited = base_build(native)
    baseline_bytes = previous.SerializeToString()
    del previous
    name = inherited["preserved_floating_initializers"]["magnitude"]
    initializers = {value.name: value for value in graph.graph.initializer}
    weight = numpy_helper.to_array(initializers[name])
    require(np.array_equal(weight, native.spec_encode.magnitude_projection.weight.detach().numpy().T),
            "Magnitude initializer differs from the saved checkpoint")
    matches = [node for node in graph.graph.node if name in node.input]
    require(len(matches) == 1 and matches[0].op_type == "MatMul" and matches[0].input[1] == name,
            "Require one unshared magnitude matrix")
    node = matches[0]
    values = [quantize_data(np.ascontiguousarray(column), T.UINT8, symmetric=False, reduce_range=False)
              for column in weight.T]
    zero = np.asarray([value[0] for value in values], np.uint8).reshape(-1)
    scale = np.asarray([value[1] for value in values], np.float32).reshape(-1)
    quantized = np.ascontiguousarray(np.stack([value[2] for value in values], axis=1), dtype=np.uint8)
    additions = [numpy_helper.from_array(value, name + suffix) for suffix, value in
                 (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero))]
    prefix = node.name + "/magnitude_u8u8"
    quant, act_scale, act_zero = (prefix + suffix for suffix in ("/input", "/input_scale", "/input_zero"))
    integer, cast, combined = (prefix + suffix for suffix in ("/integer", "/float", "/scale"))
    replacements = [
        helper.make_node("DynamicQuantizeLinear", [node.input[0]], [quant, act_scale, act_zero], name=prefix + "/quantize"),
        helper.make_node("MatMulInteger", [quant, name + "_quantized", act_zero, name + "_zero_point"],
                         [integer], name=prefix + "/matmul"),
        helper.make_node("Cast", [integer], [cast], name=prefix + "/cast", to=T.FLOAT),
        helper.make_node("Mul", [act_scale, name + "_scale"], [combined], name=prefix + "/scales"),
        helper.make_node("Mul", [cast, combined], list(node.output), name=prefix + "/dequantize")]
    nodes = [new for old in graph.graph.node for new in (replacements if old.name == node.name else [old])]
    old_initializers = [value for value in graph.graph.initializer if value.name != name]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*old_initializers, *additions])
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph, full_check=True)
    integer_graph = copy.deepcopy(graph)
    precise = rewrite(graph)
    props = {item.key: item.value for item in precise.metadata_props}
    props.update({"hs_tasnet.runtime_variant": "quadrature-ten-u8u8-precise-v1",
                  "hs_tasnet.source_model_state_sha256": fingerprint,
                  "hs_tasnet.magnitude_projection_quantized": "true",
                  "hs_tasnet.phase_projections_quantized": "false",
                  "hs_tasnet.native_host_qualified": "false"})
    helper.set_model_props(precise, props)
    proof = {**inherited, "baseline_nine_projection_graph_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
             "preserved_floating_initializers": {key: value for key, value in
                 inherited["preserved_floating_initializers"].items() if key != "magnitude"},
             "projections": [*inherited["projections"], {"module": "spec_encode.magnitude_projection",
                 "initializer": name, "source_matrix_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                 "quantized_matrix_sha256": hashlib.sha256(quantized.tobytes()).hexdigest()}]}
    require(len(proof["projections"]) == 10 and state_sha256(native.state_dict()) == fingerprint,
            "Ten-projection conversion changed the saved model")
    result = (float_graph, integer_graph, precise, proof)
    return (*result, baseline_bytes) if return_baseline else result


class QuantizedMagnitudeEncoder(nn.Module):
    def __init__(self, original_projection, magnitude_projection):
        super().__init__()
        self.original_projection = original_projection
        self.magnitude_projection = magnitude_projection

    def forward(self, packed):
        power = packed.unflatten(-1, (1026, 2)).square().sum(-1)
        magnitude = (power + float(np.float32(1e-12))).sqrt()
        scale = power.mean(-1, keepdim=True).clamp_min(float(np.float32(1e-8))).sqrt()
        return self.original_projection(packed) + self.magnitude_projection(torch.log1p(magnitude / scale))


def make_reference(native, graph, proof):
    from onnx import numpy_helper
    from research.direct.latency58_quadrature_int8 import make_reference as base_reference
    from research.direct.latency58_int8_reference import IntegerLinear
    from research.direct.latency58_int8_precise_float import PreciseProjection
    reference, independent = base_reference(native, graph, proof)
    stored = {value.name: numpy_helper.to_array(value) for value in graph.graph.initializer}
    entries = [entry for entry in proof["projections"] if entry["module"] == "spec_encode.magnitude_projection"]
    require(len(entries) == 1, "Require one independently reconstructed magnitude projection")
    integer = IntegerLinear(native.spec_encode.magnitude_projection.weight, None, stored, entries[0]["initializer"])
    reference.model.spec_encode = QuantizedMagnitudeEncoder(
        reference.model.spec_encode.original_projection, PreciseProjection(integer))
    independent = [*independent, {"module": "spec_encode.magnitude_projection", **integer.proof}]
    require(len(independent) == 10, "Every integer matrix needs independent reconstruction")
    return reference.eval().requires_grad_(False), independent
