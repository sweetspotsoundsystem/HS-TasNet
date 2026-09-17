"""Unqualified precise U8U8 execution of the saved quadrature model.

Nine large projections are quantized with ORT's per-channel weight routine.
The magnitude projection and both phase factors remain floating point. All transformations stay in
memory; no intermediate FP32 graph is written to disk.
"""
from __future__ import annotations

import copy
import hashlib
import io

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.direct.run_latency58_quality import require


def build(native):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper
    from onnxruntime.quantization.onnx_model import ONNXModel
    from onnxruntime.quantization.quant_utils import quantize_data
    from research.direct.latency58_quadrature_onnx import INPUT_NAMES, OUTPUT_NAMES, OUTPUT_SHAPES, make_export_copy
    from research.direct.latency58_conv_gemm import convert
    from research.direct.latency58_int8_precise_core import rewrite
    from research.direct.train_latency58 import state_sha256
    fingerprint = state_sha256(native.state_dict())
    wrapper = make_export_copy(native)
    output = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(wrapper, (torch.zeros(1, 2, 128), *native.initial_state(1)), output,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES), dynamo=False, external_data=False)
    float_graph = onnx.load_model_from_string(output.getvalue())
    output.close()
    for value, shape in zip(float_graph.graph.output, OUTPUT_SHAPES, strict=True):
        dims = value.type.tensor_type.shape
        dims.ClearField("dim")
        for size in shape:
            dims.dim.add().dim_value = size
    helper.set_model_props(float_graph, {"hs_tasnet.runtime_variant": "quadrature-fp32-unqualified-v1",
                                        "hs_tasnet.source_model_state_sha256": fingerprint,
                                        "hs_tasnet.native_host_qualified": "false"})
    onnx.checker.check_model(float_graph, full_check=True)
    graph = copy.deepcopy(float_graph)
    conversion = convert(graph, ["/conv_encode/Conv", "/basis_to_embed/Conv"])
    helper_model = ONNXModel(graph)
    helper_model.replace_gemm_with_matmul()
    graph = helper_model.model
    weights = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    modules = {name: getattr(native, name) for name in
               ("conv_encode", "basis_to_embed", "spec_encode", "to_spec_masks", "to_waveform_masks")}
    expected = {name: module.weight.detach().numpy().reshape(module.weight.shape[0], -1).T
                for name, module in modules.items()}
    expected.update({f"fusion_branch.weight_{kind}_l{layer}":
                     getattr(native.fusion_branch, f"weight_{kind}_l{layer}").detach().numpy().T
                     for layer in range(2) for kind in ("ih", "hh")})
    floating = {"magnitude": native.spec_encode.magnitude_projection.weight.detach().numpy().T,
                "phase_reduce": native.phase_reduce.weight.detach().numpy().T,
                "phase_expand": native.phase_expand.weight.detach().numpy().T}
    replacements, removed, proofs, additions = {}, set(), [], []
    floating_names = {}
    for node in graph.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in weights:
            continue
        name, weight = node.input[1], weights[node.input[1]]
        floating_matches = [key for key, value in floating.items() if np.array_equal(weight, value)]
        if floating_matches:
            require(len(floating_matches) == 1 and floating_matches[0] not in floating_names,
                    "Repeated or ambiguous floating projection")
            floating_names[floating_matches[0]] = name
            continue
        matched = [key for key, value in expected.items() if np.array_equal(weight, value)]
        require(len(matched) == 1 and name not in removed, "Unmapped or repeated dense matrix")
        require(sum(name in other.input for other in graph.graph.node) == 1, "Shared matrix needs separate review")
        module_name = matched[0]
        # ORT routine operates independently on each output channel. The
        # separate PyTorch oracle recomputes every byte from checkpoint weights.
        values = [quantize_data(np.ascontiguousarray(column), T.UINT8, symmetric=False, reduce_range=False)
                  for column in weight.T]
        zero = np.asarray([v[0] for v in values], np.uint8).reshape(-1)
        scale = np.asarray([v[1] for v in values], np.float32).reshape(-1)
        quantized = np.ascontiguousarray(np.stack([v[2] for v in values], axis=1), dtype=np.uint8)
        for suffix, value in (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero)):
            additions.append(numpy_helper.from_array(value, name + suffix))
        prefix = node.name + "/u8u8"
        quant, act_scale, act_zero = (prefix + suffix for suffix in ("/input", "/input_scale", "/input_zero"))
        integer, cast, combined = (prefix + suffix for suffix in ("/integer", "/float", "/scale"))
        replacements[node.name] = [
            helper.make_node("DynamicQuantizeLinear", [node.input[0]], [quant, act_scale, act_zero], name=prefix + "/quantize"),
            helper.make_node("MatMulInteger", [quant, name + "_quantized", act_zero, name + "_zero_point"],
                             [integer], name=prefix + "/matmul"),
            helper.make_node("Cast", [integer], [cast], name=prefix + "/cast", to=T.FLOAT),
            helper.make_node("Mul", [act_scale, name + "_scale"], [combined], name=prefix + "/scales"),
            helper.make_node("Mul", [cast, combined], list(node.output), name=prefix + "/dequantize")]
        removed.add(name)
        proofs.append({"module": module_name, "initializer": name,
                       "source_matrix_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                       "quantized_matrix_sha256": hashlib.sha256(quantized.tobytes()).hexdigest()})
    require(len(proofs) == 9 and {v["module"] for v in proofs} == set(expected)
            and set(floating_names) == set(floating), "Require nine integer projections and three preserved floating matrices")
    nodes = [new for node in graph.graph.node for new in replacements.get(node.name, [node])]
    initializers = [t for t in graph.graph.initializer if t.name not in removed]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*initializers, *additions])
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph, full_check=True)
    integer_graph = copy.deepcopy(graph)
    precise = rewrite(graph)
    properties = {item.key: item.value for item in precise.metadata_props}
    properties.update({"hs_tasnet.runtime_variant": "quadrature-nine-u8u8-precise-v1",
                       "hs_tasnet.source_model_state_sha256": fingerprint,
                       "hs_tasnet.magnitude_projection_quantized": "false",
                       "hs_tasnet.phase_projections_quantized": "false"})
    helper.set_model_props(precise, properties)
    require(state_sha256(native.state_dict()) == fingerprint, "Export or conversion changed the saved model")
    return float_graph, integer_graph, precise, {"projections": proofs, "conv_rewrites": conversion,
                                               "preserved_floating_initializers": floating_names}


class PreciseMagnitudeEncoder(nn.Module):
    def __init__(self, original, projection):
        super().__init__()
        self.original_projection = projection
        self.register_buffer("magnitude_weight", original.magnitude_projection.weight.detach().double().clone())

    def forward(self, packed):
        power = packed.unflatten(-1, (1026, 2)).square().sum(-1)
        magnitude = (power + float(np.float32(1e-12))).sqrt()
        scale = power.mean(-1, keepdim=True).clamp_min(float(np.float32(1e-8))).sqrt()
        return self.original_projection(packed) + F.linear(torch.log1p(magnitude / scale), self.magnitude_weight)


def make_reference(native, integer_graph, proof):
    from onnx import numpy_helper
    from research.direct.latency58_quadrature_onnx import make_export_copy
    from research.direct.latency58_int8_reference import IntegerLinear, IntegerConv, IntegerGRU
    from research.direct.latency58_int8_precise_float import PreciseProjection
    stored = {v.name: numpy_helper.to_array(v) for v in integer_graph.graph.initializer}
    prefixes = {v["module"]: v["initializer"] for v in proof["projections"]}
    independent = []
    def projection(weight, bias, name):
        module = IntegerLinear(weight, bias, stored, prefixes[name])
        independent.append({"module": name, **module.proof})
        return PreciseProjection(module)
    copied = make_export_copy(native).model
    for name in ("conv_encode", "basis_to_embed"):
        original = getattr(copied, name)
        converted = IntegerConv(original, stored, prefixes[name])
        converted.projection = projection(original.weight, original.bias, name)
        setattr(copied, name, converted)
    class Recurrent(IntegerGRU):
        def __init__(self, original):
            nn.Module.__init__(self)
            for attr, kind in (("input_maps", "ih"), ("hidden_maps", "hh")):
                setattr(self, attr, nn.ModuleList([projection(getattr(original, f"weight_{kind}_l{layer}"),
                    getattr(original, f"bias_{kind}_l{layer}"), f"fusion_branch.weight_{kind}_l{layer}")
                    for layer in range(2)]))
    copied.fusion_branch = Recurrent(copied.fusion_branch)
    original = copied.spec_encode
    copied.spec_encode = PreciseMagnitudeEncoder(original, projection(original.weight, original.bias, "spec_encode"))
    for name in ("to_spec_masks", "to_waveform_masks"):
        original = getattr(copied, name)
        setattr(copied, name, projection(original.weight, original.bias, name))
    require(len(independent) == 9, "Every integer projection needs independent reconstruction")
    return QuadraturePreciseReference(copied.double()).eval().requires_grad_(False), independent


class QuadraturePreciseReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, history, hidden, spectral_tail, waveform_tail):
        from research.direct.latency58 import PUBLIC_FUSION_SCALE
        m = self.model
        require(all(v.dtype == torch.float32 and v.device.type == "cpu" for v in
                    (audio, history, hidden, spectral_tail, waveform_tail)), "Require public CPU FP32")
        joined = torch.cat((history.double(), audio.double()), -1)
        spectrum = torch.fft.rfft((joined * m.analysis_window).reshape(2, 1024), n=1024, dim=-1)
        feature = torch.view_as_real(spectrum).reshape(1, 2, 1, 513, 2)
        packed = feature.permute(0, 2, 1, 3, 4).reshape(1, 1, 2052)
        spec = m.spec_encode(packed)
        real, gate = m.conv_encode(joined).chunk(2, dim=1)
        basis = real.relu() * gate.sigmoid()
        wave = m.basis_to_embed(basis).transpose(1, 2)
        fused_input = torch.cat((spec, wave), -1)
        recurrent, next_hidden = m.fusion_branch(fused_input, hidden.double() / PUBLIC_FUSION_SCALE)
        fused_spec, fused_wave = (fused_input + recurrent).chunk(2, dim=-1)
        features = m.spec_norm(fused_spec + spec)
        logits = m.to_spec_masks(features).float().reshape(1, 1, 2, 513, 2, 4)
        masks = m._residual_source_softmax(logits).permute(0, 2, 1, 3, 4, 5)
        # Explicit FP32 phase arithmetic at the graph's precision boundary.
        # Do not call the native phase helper from this independent oracle.
        phase = F.linear(F.linear(features.float(), m.phase_reduce.weight.float()), m.phase_expand.weight.float())
        phase = phase.reshape(1, 1, 2, 511, 4)
        phase = F.pad(phase - phase.mean(-1, keepdim=True), (0, 0, 1, 1)).permute(0, 2, 1, 3, 4)
        carrier = feature.float()
        rotated = torch.stack((-carrier[..., 1], carrier[..., 0]), -1)
        masked = (carrier.unsqueeze(-1) * masks + rotated.unsqueeze(-1) * phase.unsqueeze(-2))
        masked = masked.permute(0, 5, 1, 2, 3, 4).contiguous()
        frames = torch.fft.irfft(torch.view_as_complex(masked), n=1024, dim=-1)[..., -256:]
        frames = frames[..., 0, :] * m.synthesis.spectral_window.float()
        spectral = (frames[..., :128] + spectral_tail) / m.synthesis.spectral_denominator.float()
        logits = m.to_waveform_masks(m.waveform_norm(fused_wave + wave)).float().reshape(1, 1, 4, 1500).transpose(-1, -2)
        masks = m._residual_source_softmax(logits)
        source_basis = (basis.float().transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        decoded = F.linear(source_basis, m.waveform_decoder_weight.float().flatten(1).t())
        decoded = decoded.reshape(1, 4, 1, 2, 256).permute(0, 1, 3, 2, 4).reshape(1, 4, 2, 256)
        decoded = decoded * m.synthesis.window.float()
        wave_audio = decoded[..., :128] + waveform_tail
        raw = (spectral + wave_audio) * m.output_source_scales.float()[None, :, None, None]
        from research.direct.latency58_residual_model import corrected_estimates
        _, deployed = corrected_estimates(raw, history[..., -128:], m.fixed_residual_share.float())
        return (deployed, joined[..., -896:].float().clone(), (next_hidden * PUBLIC_FUSION_SCALE).float(),
                frames[..., 128:].clone(), decoded[..., 128:].clone())

