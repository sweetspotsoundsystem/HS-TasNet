"""Independent fused-QKV integer reconstruction using no runtime-output oracle."""
import torch
from torch import nn
from torch.nn import functional as F
from research.direct.run_latency58_quality import require


def make_reference(native, ten_graph, ten_conversion, fourteen_graph, fourteen_conversion,
                   sixteen_graph, sixteen_conversion, candidate, conversion):
    from onnx import numpy_helper as nh
    from research.direct.latency58_branch_output_int8_reference import make_reference as parent_reference
    from research.direct.latency58_branch_int8 import SignedProjection
    reference, proof = parent_reference(native, ten_graph, ten_conversion, fourteen_graph,
                                       fourteen_conversion, sixteen_graph, sixteen_conversion)
    names = ('temporal_query', 'temporal_key', 'temporal_value')
    require(all(getattr(native, name).bias is None for name in names), 'QKV bias policy changed')
    weight = torch.cat([getattr(native, name).weight for name in names], dim=0)
    require(tuple(weight.shape) == (256, 1000), 'Combined source QKV shape changed')
    stored = {v.name: nh.to_array(v) for v in candidate.graph.initializer}
    projection = SignedProjection(weight, None, stored, conversion['initializer'])
    reference.model.temporal_qkv = projection
    for name in names:
        delattr(reference.model, name)
    proof.append({'module': 'temporal_qkv', 'source_modules': names, **projection.proof, 'bias': None})
    require(len(proof) == 17, 'Expected seventeen independent integer products')
    return AttentionQKVIntegerReference(reference.model).eval().requires_grad_(False), proof


class AttentionQKVIntegerReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, history, hidden, spectral_tail, waveform_tail, past_keys, past_values,
                spec_hidden, waveform_hidden):
        from research.direct.latency58 import PUBLIC_FUSION_SCALE
        from research.direct.latency58_residual_model import corrected_estimates
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
        spec_correction = m.spec_memory_output(spec_memory)
        wave_correction = m.waveform_memory_output(wave_memory)
        qkv = m.temporal_qkv(refined)
        queries, new_keys, new_values = qkv.split((64, 64, 128), dim=-1)
        queries = queries[:, -1:]
        keys = torch.cat((past_keys.double(), new_keys), 1)
        values = torch.cat((past_values.double(), new_values), 1)
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
