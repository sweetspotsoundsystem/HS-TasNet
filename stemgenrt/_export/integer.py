"""Ten signed integer projections from the selected saved branch-memory checkpoint.

The independent CPU reference reconstructs all quantization bytes without ORT.
This is a distinct inference variant and requires its own full-panel quality
evaluation; the source checkpoint's FP32 score is not its deployment score.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .helpers import require, sha
from .helpers import state_sha256

VERSION = "branch-memory-ten-u8s8-reduced-precise-v1"


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build(native, fp32_graph):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    from onnxruntime.quantization.onnx_model import ONNXModel
    from onnxruntime.quantization.quant_utils import quantize_data
    from .conv_gemm import convert
    from .integer_precision import rewrite

    fingerprint = state_sha256(native.state_dict())
    graph = copy.deepcopy(fp32_graph)
    require(len(graph.graph.input) == len(graph.graph.output) == 1 + len(native.initial_state(1)),
            "Require the selected branch-memory model's complete state interface")
    float_sha = hashlib.sha256(graph.SerializeToString()).hexdigest()
    properties = {v.key: v.value for v in graph.metadata_props}
    convolution_proof = convert(graph, ["/conv_encode/Conv", "/basis_to_embed/Conv"])
    helper_model = ONNXModel(graph)
    helper_model.replace_gemm_with_matmul()
    graph = helper_model.model
    weights = {v.name: nh.to_array(v) for v in graph.graph.initializer}
    expected = source_matrices(native)
    replacements, removed, additions, proof, preserved = {}, set(), [], [], {}
    for node in graph.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in weights:
            continue
        name, weight = node.input[1], weights[node.input[1]]
        matches = [key for key, value in expected.items() if np.array_equal(weight, value)]
        if not matches:
            preserved[name] = {"shape": list(weight.shape), "sha256": digest(weight)}
            continue
        require(len(matches) == 1 and name not in removed
                and sum(name in n.input for n in graph.graph.node) == 1,
                "Shared, repeated or ambiguous integer matrix")
        module = matches[0]
        channels = [quantize_data(np.ascontiguousarray(column), T.INT8,
                                  symmetric=True, reduce_range=True) for column in weight.T]
        zero = np.asarray([v[0] for v in channels], np.int8).reshape(-1)
        scale = np.asarray([v[1] for v in channels], np.float32).reshape(-1)
        quantized = np.ascontiguousarray(np.stack([v[2] for v in channels], axis=1), dtype=np.int8)
        require(np.all(zero == 0) and np.abs(quantized.astype(np.int16)).max() <= 64
                and weight.shape[0] * 255 * 64 < 2**31,
                "Signed pair or complete centered dot product could overflow")
        for suffix, value in (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero)):
            additions.append(nh.from_array(value, name + suffix))
        kind = "magnitude_u8s8" if module == "spec_encode.magnitude_projection" else "u8s8"
        prefix = node.name + "/" + kind
        quant, act_scale, act_zero = (prefix + v for v in ("/input", "/input_scale", "/input_zero"))
        integer, cast, combined = (prefix + v for v in ("/integer", "/float", "/scale"))
        replacements[node.name] = [
            helper.make_node("DynamicQuantizeLinear", [node.input[0]], [quant, act_scale, act_zero],
                             name=prefix + "/quantize"),
            helper.make_node("MatMulInteger", [quant, name + "_quantized", act_zero, name + "_zero_point"],
                             [integer], name=prefix + "/matmul"),
            helper.make_node("Cast", [integer], [cast], name=prefix + "/cast", to=T.FLOAT),
            helper.make_node("Mul", [act_scale, name + "_scale"], [combined], name=prefix + "/scales"),
            helper.make_node("Mul", [cast, combined], list(node.output), name=prefix + "/dequantize")]
        removed.add(name)
        proof.append({"module": module, "initializer": name, "shape": list(weight.shape),
                      "source_matrix_sha256": digest(weight), "quantized_matrix_sha256": digest(quantized),
                      "scale_sha256": digest(scale), "weight_zero_points_all_zero": True,
                      "weight_precision": "S8 symmetric reduced range [-64,64]",
                      "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64,
                      "worst_centered_dot_absolute_sum": weight.shape[0] * 255 * 64})
    require(len(proof) == 10 and {v["module"] for v in proof} == set(expected),
            "Require exactly ten mapped integer projections")
    preserved_bytes = {v.name: v.SerializeToString() for v in graph.graph.initializer if v.name not in removed}
    nodes = [replacement for n in graph.graph.node for replacement in replacements.get(n.name, [n])]
    initializers = [v for v in graph.graph.initializer if v.name not in removed]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*initializers, *additions])
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph, full_check=True)
    integer_graph = copy.deepcopy(graph)
    precise = rewrite(graph)
    precise_count = {v.key: v.value for v in precise.metadata_props}["stemgenrt.precise_node_count"]
    # Historical source quality may accompany a pinned graph; it is never
    # interpreted as the integer candidate's quality.
    for key in ("full14_sdr_db", "full14_result_sha256"):
        old_key = "stemgenrt." + key
        if old_key in properties:
            properties["stemgenrt.source_checkpoint_" + key] = properties.pop(old_key)
    properties.update({"stemgenrt.runtime_variant": VERSION,
                       "stemgenrt.integer_weight_precision": "Ten S8 symmetric reduced-range matrices [-64,64]",
                       "stemgenrt.activation_quantization": "Dynamic U8 per one-frame projection; FP32 scales and dequantization",
                       "stemgenrt.precise_node_count": precise_count,
                       "stemgenrt.inference_precision": "FP64 quantizer ancestors; integer projections and output decoding FP32; public states FP32",
                       "stemgenrt.floating_additional_layers": "phase factors, fusion refinement, temporal attention and two branch GRUs",
                       "stemgenrt.refinement_silu_implementation": "x / (1 + exp(-x)); FP64 quantizer ancestor",
                       "stemgenrt.integer_exporter_sha256": sha(__file__),
                       "stemgenrt.integer_precision_transform_sha256": sha(rewrite.__code__.co_filename),
                       "stemgenrt.embedded_quality_scope": "Source FP32 checkpoint only; deployment graph quality is recorded separately"})
    helper.set_model_props(precise, properties)
    require(all(v.SerializeToString() == preserved_bytes[v.name] for v in precise.graph.initializer
                if v.name in preserved_bytes) and state_sha256(native.state_dict()) == fingerprint,
            "Unquantized initializer or source model changed")
    onnx.checker.check_model(precise, full_check=True)
    return integer_graph, precise, {"version": VERSION, "source_fp32_graph_sha256": float_sha,
                                   "source_model_state_sha256": fingerprint, "projections": proof,
                                   "convolution_rewrites": convolution_proof,
                                   "preserved_floating_matrices": preserved,
                                   "unquantized_initializers_byte_exact": True,
                                   "quality_measured": False, "native_host_qualified": False}


class SignedProjection(nn.Module):
    """Independently derive signed weights, then use PyTorch int32 dot products."""
    def __init__(self, weight, bias, stored, prefix):
        super().__init__()
        from .integer_reference import PreciseProjection
        raw = weight.detach().float().cpu().reshape(weight.shape[0], -1).numpy()
        maximum = np.max(np.abs(raw), axis=1)
        scale = ((maximum - (-maximum)).astype(np.float64) / 128.).astype(np.float32)
        scale = np.where(scale < np.finfo(np.float32).tiny, np.float32(1), scale)
        quantized = np.clip(np.rint(raw / scale[:, None]), -64, 64).astype(np.int8)
        require(np.array_equal(quantized.T, stored[prefix + "_quantized"])
                and np.array_equal(scale, stored[prefix + "_scale"])
                and np.array_equal(np.zeros(raw.shape[0], np.int8), stored[prefix + "_zero_point"]),
                "Independent signed range reconstruction differs: " + prefix)
        self.register_buffer("weight", torch.from_numpy(np.ascontiguousarray(quantized.T.astype(np.int32))))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float64)))
        self.register_buffer("bias", None if bias is None else bias.detach().double().cpu().clone())
        self.proof = {"initializer": prefix, "source_float_sha256": digest(raw),
                      "quantized_matrix_sha256": digest(quantized.T),
                      "weight_scale_zero_point_bit_exact": True,
                      "reference_arithmetic": "NumPy range reconstruction and PyTorch int32 products; no ORT oracle"}
        self._projection_forward = PreciseProjection.forward

    def forward(self, values):
        return self._projection_forward(self, values)


class OneFrameConv(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, x):
        return self.projection(x.reshape(1, -1)).unsqueeze(-1)


class MagnitudeFeatures(nn.Module):
    def __init__(self, original, magnitude):
        super().__init__()
        self.original_projection, self.magnitude_projection = original, magnitude

    def forward(self, packed):
        power = packed.unflatten(-1, (1026, 2)).square().sum(-1)
        magnitude = (power + float(np.float32(1e-12))).sqrt()
        scale = power.mean(-1, keepdim=True).clamp_min(float(np.float32(1e-8))).sqrt()
        return self.original_projection(packed) + self.magnitude_projection(torch.log1p(magnitude / scale))


def make_reference(native, integer_graph, proof):
    from onnx import numpy_helper as nh
    from .fp32 import make_export_copy
    from .integer_reference import IntegerGRU
    stored = {v.name: nh.to_array(v) for v in integer_graph.graph.initializer}
    prefixes = {v["module"]: v["initializer"] for v in proof["projections"]}
    independent = []
    def projection(weight, bias, name):
        module = SignedProjection(weight, bias, stored, prefixes[name])
        independent.append({"module": name, **module.proof})
        return module
    copied = make_export_copy(native).model
    for name in ("conv_encode", "basis_to_embed"):
        original = getattr(copied, name)
        setattr(copied, name, OneFrameConv(projection(original.weight, original.bias, name)))
    class Recurrent(IntegerGRU):
        def __init__(self, original):
            nn.Module.__init__(self)
            for attr, kind in (("input_maps", "ih"), ("hidden_maps", "hh")):
                setattr(self, attr, nn.ModuleList([projection(getattr(original, f"weight_{kind}_l{layer}"),
                    getattr(original, f"bias_{kind}_l{layer}"), f"fusion_branch.weight_{kind}_l{layer}")
                    for layer in range(2)]))
    copied.fusion_branch = Recurrent(copied.fusion_branch)
    original = copied.spec_encode
    copied.spec_encode = MagnitudeFeatures(projection(original.weight, original.bias, "spec_encode"),
        projection(original.magnitude_projection.weight, None, "spec_encode.magnitude_projection"))
    for name in ("to_spec_masks", "to_waveform_masks"):
        original = getattr(copied, name)
        setattr(copied, name, projection(original.weight, original.bias, name))
    require(len(independent) == len(prefixes) == 10, "Require ten independent projection reconstructions")
    return BranchIntegerReference(copied.double()).eval().requires_grad_(False), independent


class BranchIntegerReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, history, hidden, spectral_tail, waveform_tail, past_keys, past_values,
                spec_hidden, waveform_hidden):
        from ..model import PUBLIC_FUSION_SCALE
        from ..model import corrected_estimates
        m = self.model
        require(all(v.dtype == torch.float32 and v.device.type == "cpu" for v in
                    (audio, history, hidden, spectral_tail, waveform_tail, past_keys, past_values, spec_hidden, waveform_hidden)), 
                "Require CPU public float32 values")
        joined = torch.cat((history.double(), audio.double()), -1)
        spectrum = torch.fft.rfft((joined * m.analysis_window).reshape(2, 1024), n=1024, dim=-1)
        feature = torch.view_as_real(spectrum).reshape(1, 2, 1, 513, 2)
        packed = feature.permute(0, 2, 1, 3, 4).reshape(1, 1, 2052)
        spec = m.spec_encode(packed)
        real, gate = m.conv_encode(joined).chunk(2, dim=1)
        basis = real.relu() * gate.sigmoid()
        wave = m.basis_to_embed(basis).transpose(1, 2)
        fusion_input = torch.cat((spec, wave), -1)
        recurrent, next_hidden = m.fusion_branch(fusion_input, hidden.double() / PUBLIC_FUSION_SCALE)
        fused = fusion_input + recurrent
        refined = fused + F.linear(F.silu(F.linear(fused, m.fusion_refine_reduce.weight)),
                                   m.fusion_refine_expand.weight)
        private_spec, private_wave = refined.chunk(2, -1)
        private_spec = m.spec_norm(private_spec + spec)
        private_wave = m.waveform_norm(private_wave + wave)
        spec_memory, next_spec_hidden = m.spec_memory(private_spec, spec_hidden.double() / PUBLIC_FUSION_SCALE)
        wave_memory, next_wave_hidden = m.waveform_memory(private_wave, waveform_hidden.double() / PUBLIC_FUSION_SCALE)
        spec_correction = F.linear(spec_memory, m.spec_memory_output.weight)
        wave_correction = F.linear(wave_memory, m.waveform_memory_output.weight)
        queries = F.linear(refined, m.temporal_query.weight)
        keys = torch.cat((past_keys.double(), F.linear(refined, m.temporal_key.weight)), 1)
        values = torch.cat((past_values.double(), F.linear(refined, m.temporal_value.weight)), 1)
        logits = (queries.unsqueeze(-2) * keys.unsqueeze(1)).sum(-1) * 0.125
        weights = torch.softmax(logits, dim=-1)
        attended = (weights.unsqueeze(-1) * values.unsqueeze(1)).sum(-2)
        fused_spec, fused_wave = (refined + F.linear(attended, m.temporal_output.weight)).chunk(2, -1)
        features = m.spec_norm(fused_spec + spec) + spec_correction
        logits = m.to_spec_masks(features).float().reshape(1, 1, 2, 513, 2, 4)
        masks = m._residual_source_softmax(logits).permute(0, 2, 1, 3, 4, 5)
        phase = F.linear(F.linear(features.float(), m.phase_reduce.weight.float()), m.phase_expand.weight.float())
        phase = phase.reshape(1, 1, 2, 511, 4)
        phase = F.pad(phase - phase.mean(-1, keepdim=True), (0, 0, 1, 1)).permute(0, 2, 1, 3, 4)
        carrier = feature.float()
        rotated = torch.stack((-carrier[..., 1], carrier[..., 0]), -1)
        masked = carrier.unsqueeze(-1) * masks + rotated.unsqueeze(-1) * phase.unsqueeze(-2)
        masked = masked.permute(0, 5, 1, 2, 3, 4).contiguous()
        frames = torch.fft.irfft(torch.view_as_complex(masked), n=1024, dim=-1)[..., -256:]
        frames = frames[..., 0, :] * m.synthesis.spectral_window.float()
        spectral = (frames[..., :128] + spectral_tail) / m.synthesis.spectral_denominator.float()
        logits = m.to_waveform_masks(m.waveform_norm(fused_wave + wave) + wave_correction).float().reshape(1, 1, 4, 1500).transpose(-1, -2)
        masks = m._residual_source_softmax(logits)
        source_basis = (basis.float().transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        decoded = F.linear(source_basis, m.waveform_decoder_weight.float().flatten(1).t())
        decoded = decoded.reshape(1, 4, 1, 2, 256).permute(0, 1, 3, 2, 4).reshape(1, 4, 2, 256)
        decoded = decoded * m.synthesis.window.float()
        waveform = decoded[..., :128] + waveform_tail
        raw = (spectral + waveform) * m.output_source_scales.float()[None, :, None, None]
        _, deployed = corrected_estimates(raw, history[..., -128:], m.fixed_residual_share.float())
        return (deployed, joined[..., -896:].float().clone(), (next_hidden * PUBLIC_FUSION_SCALE).float(),
                frames[..., 128:].clone(), decoded[..., 128:].clone(),
                keys[:, -31:].float().clone(), values[:, -31:].float().clone(),
                (next_spec_hidden * PUBLIC_FUSION_SCALE).float(),
                (next_wave_hidden * PUBLIC_FUSION_SCALE).float())


def source_matrices(native):
    modules = {name: getattr(native, name) for name in
               ("conv_encode", "basis_to_embed", "spec_encode", "to_spec_masks", "to_waveform_masks")}
    matrices = {name: np.ascontiguousarray(module.weight.detach().cpu().numpy().reshape(module.weight.shape[0], -1).T)
                for name, module in modules.items()}
    matrices.update({f"fusion_branch.weight_{kind}_l{layer}": np.ascontiguousarray(
        getattr(native.fusion_branch, f"weight_{kind}_l{layer}").detach().cpu().numpy().T)
        for layer in range(2) for kind in ("ih", "hh")})
    matrices["spec_encode.magnitude_projection"] = np.ascontiguousarray(
        native.spec_encode.magnitude_projection.weight.detach().cpu().numpy().T)
    require(len(matrices) == 10 and all(v.dtype == np.float32 and v.ndim == 2 for v in matrices.values()),
            "Expected ten FP32 source matrices")
    return matrices
