"""Keep higher precision only on paths feeding dynamic quantization.

Output decoding returns to the original FP32 arithmetic. The graph transform
derives this boundary from data dependencies and inserts explicit casts.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.direct.run_latency58_quality import require


def rewrite(graph):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper

    original = onnx.shape_inference.infer_shapes(copy.deepcopy(graph), strict_mode=True, check_type=True)
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*original.graph.input, *original.graph.output, *original.graph.value_info]}
    types.update({v.name: v.data_type for v in original.graph.initializer})
    require(all(v in types for n in original.graph.node for v in n.output), "Missing inferred value type")
    producers = {v: n for n in original.graph.node for v in n.output}
    consumers = {}
    for node in original.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    def single(value, operation):
        nodes = consumers[value]
        require(len(nodes) == 1 and nodes[0].op_type == operation, "Projection topology changed")
        return nodes[0]
    islands, quantizers = set(), []
    for integer in (n for n in original.graph.node if n.op_type == "MatMulInteger"):
        quantize = producers[integer.input[0]]
        cast = single(integer.output[0], "Cast")
        scaled = single(cast.output[0], "Mul")
        scales = producers[next(v for v in scaled.input if v != cast.output[0])]
        bias = single(scaled.output[0], "Add")
        require(quantize.op_type == "DynamicQuantizeLinear" and scales.op_type == "Mul"
                and quantize.output[1] in scales.input, "Wrong integer block")
        islands.update(n.name for n in (quantize, integer, cast, scaled, scales, bias))
        quantizers.append(quantize)
    require(len(islands) == 54 and len(quantizers) == 9, "Require nine unchanged integer blocks")
    precise = set()
    def visit(value):
        node = producers.get(value)
        if node is None or node.name in islands or node.name in precise:
            return
        precise.add(node.name)
        for parent in node.input:
            visit(parent)
    for quantizer in quantizers:
        visit(quantizer.input[0])
    require(all(n.name not in precise for n in original.graph.node if n.name in islands), "Precision boundary crossed a projection")
    require(next(n for n in original.graph.node if n.op_type == "DFT").name in precise
            and sum(n.op_type == "DFT" and n.name in precise for n in original.graph.node) == 1,
            "Only the feature FFT should need FP64")
    available = {v.name: v.type.tensor_type.elem_type for v in original.graph.input}
    available.update({t.name: t.data_type for t in original.graph.initializer})
    nodes, casts = [], {}
    public_names = {v.name for v in original.graph.output}
    aliases = {name: name + "__internal" for name in public_names}
    def input_as(name, expected):
        actual_name = aliases.get(name, name)
        actual = available[name]
        if actual != T.FLOAT and actual != T.DOUBLE:
            return actual_name
        if actual == expected:
            return actual_name
        key = (name, expected)
        if key not in casts:
            value = name + ("__to64" if expected == T.DOUBLE else "__to32")
            nodes.append(helper.make_node("Cast", [actual_name], [value],
                         name="/precise_core/cast_" + str(len(casts)), to=expected))
            casts[key] = value
        return casts[key]
    for original_node in original.graph.node:
        node = copy.deepcopy(original_node)
        precision = T.DOUBLE if node.name in precise else T.FLOAT
        for index, value in enumerate(node.input):
            if value:
                node.input[index] = input_as(value, precision)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == T.FLOAT and precision == T.DOUBLE:
                attribute.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attribute.t).astype(np.float64), attribute.t.name))
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == T.FLOAT:
                attribute.i = precision
        for index, value in enumerate(node.output):
            available[value] = precision if types[value] == T.FLOAT else types[value]
            node.output[index] = aliases.get(value, value)
        nodes.append(node)
    for output in original.graph.output:
        source = input_as(output.name, T.FLOAT)
        nodes.append(helper.make_node("Identity", [source], [output.name], name="/precise_core/output/" + output.name))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.value_info[:]
    # Every initializer is left byte-exact; constant casts can be folded by ORT.
    require([t.SerializeToString() for t in graph.graph.initializer] ==
            [t.SerializeToString() for t in original.graph.initializer], "Initializer bytes changed")
    metadata = {"source." + item.key: item.value for item in graph.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "c204-hop128-u8u8-fp64-quantizer-ancestors-v1",
                     "hs_tasnet.native_host_qualified": "false",
                     "hs_tasnet.precise_node_count": str(len(precise))})
    helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    return graph


class PreciseCoreReference(nn.Module):
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
        logits = m.to_spec_masks(m.spec_norm(fused_spec + spec)).float().reshape(1, 1, 2, 513, 2, 4)
        masks = m._residual_source_softmax(logits).permute(0, 2, 1, 3, 4, 5)
        masked = (feature.float().unsqueeze(-1) * masks).permute(0, 5, 1, 2, 3, 4).contiguous()
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
        dbv = raw[:, :3]
        deployed = torch.cat((dbv, history[..., -128:].unsqueeze(1) - dbv.sum(1, keepdim=True)), 1)
        return (deployed, joined[..., -896:].float().clone(), (next_hidden * PUBLIC_FUSION_SCALE).float(),
                frames[..., 128:].clone(), decoded[..., 128:].clone())


def make_reference(native, graph):
    from research.direct.latency58_int8_precise_float import make_reference as precise_reference
    reference, proof = precise_reference(native, graph)
    return PreciseCoreReference(reference.internal.model).eval().requires_grad_(False), proof
