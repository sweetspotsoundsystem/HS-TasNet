"""Faithful FP32 ONNX lowering of the current streaming streaming model."""
from __future__ import annotations
import copy
import torch
from . import helpers as export_helpers
from .helpers import require, model_state_sha256
from ..model import (
    StemgenRT58, FEATURE_HISTORY, HOP, PUBLIC_FUSION_SCALE,
    SOURCE_ORDER, SYNTHESIS_SAMPLES, corrected_estimates,
)

def interface(model):
    require(type(model) is StemgenRT58,
            "Export requires the current streaming StemgenRT58")
    require(model.sample_rate == 44100 and model.hop_samples == 128
            and model.graph_alignment_samples == 128,
            "The current model requires 44100 Hz and a fixed 128-sample hop")
    require(tuple(model.analysis_window.shape) == (1024,)
            and tuple(model.waveform_decoder_weight.shape) == (1500, 2, 256)
            and tuple(model.conv_encode.kernel_size) == (1024,)
            and tuple(model.conv_encode.stride) == (128,),
            "The current model requires fixed 1024/256/128 analysis/synthesis/hop geometry")
    state = model.initial_state(1)
    names = tuple(state._fields)
    shapes = tuple(tuple(value.shape) for value in state)
    require(names == ("audio_history", "fusion_hidden", "spectral_numerator_tail", "waveform_tail",
                      "attention_keys", "attention_values", "spec_memory_hidden", "waveform_memory_hidden")
                      + (("past_carrier_history",) if model.has_past_filter else ())
            and shapes == ((1, 2, 896), (2, 1, 1000), (1, 4, 2, 128), (1, 4, 2, 128),
                           (1, model.attention_window - 1, 64), (1, model.attention_window - 1, 128),
                           (1, 1, 500), (1, 1, 500))
                           + (((1, 2, 2, 513, 2),) if model.has_past_filter else ()),
            "The current streaming interface changed")
    return {"state_names": names, "state_shapes": shapes,
            "input_names": ("audio_chunk", *names),
            "output_names": ("separated_chunk", *("next_" + name for name in names)),
            "input_shapes": ((1, 2, HOP), *shapes),
            "output_shapes": ((1, 4, 2, HOP), *shapes)}


TOLERANCES = {"waveform_max_abs": 1e-4, "stem_callback_rms": 1e-5,
              "state_max_abs_decoded_units": 5e-4, "reconstruction_max_abs": 1e-6}


class _RFFT1024(torch.autograd.Function):
    @staticmethod
    def forward(ctx, audio):
        spectrum = torch.fft.rfft(audio, n=1024, dim=-1)
        return torch.stack((spectrum.real, spectrum.imag), dim=-1)

    @staticmethod
    def symbolic(graph, audio):
        axis = graph.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        values = graph.op("Unsqueeze", audio, axis)
        length = graph.op("Constant", value_t=torch.tensor(1024, dtype=torch.int64))
        result = graph.op("DFT", values, length, axis_i=1, inverse_i=0, onesided_i=1)
        return result.setType(audio.type().with_sizes([audio.type().sizes()[0], 513, 2]))


def one_hop_shared_filter(kernel, features, masks, carrier, history):
    """Batch-one export of both native taps with the same source centering.

    Gate order is channel, past lag, real/imaginary component, source. Current
    frequency masks are reused for both lags. No future carrier is accessed.
    """
    gates = kernel.gate(features).tanh().reshape(1, 2, 2, 1, 2, 4)
    coefficients = masks[:, :, :, 1:-1] * gates
    coefficients = coefficients - coefficients.mean(dim=-1, keepdim=True)
    past = torch.flip(history, dims=(2,))[:, :, :, 1:-1]
    real = past[..., 0, None] * coefficients[..., 0, :] - past[..., 1, None] * coefficients[..., 1, :]
    imag = past[..., 0, None] * coefficients[..., 1, :] + past[..., 1, None] * coefficients[..., 0, :]
    correction = torch.stack((real.sum(dim=2, keepdim=True), imag.sum(dim=2, keepdim=True)), dim=-2)
    correction = torch.nn.functional.pad(correction, (0, 0, 0, 0, 1, 1))
    return correction, torch.cat((history[:, :, 1:], carrier), dim=2)

def make_export_copy(model):
    """Build only a separate export copy; the complete deployed residual is formed once."""
    interface(model)
    if model.has_past_filter:
        require(model.past_filter.lags == (1, 2) and tuple(model.past_filter.gate.weight.shape) == (32, 500)
                and model.past_filter.features == 500 and model.past_filter.bins == 513
                and model.past_filter.channels == 2 and model.past_filter.sources == 4,
                "Fixed-hop past-filter geometry changed")
    require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for value in model.state_dict().values()),
            "Export requires finite FP32 model tensors")
    class IRFFT1024(torch.autograd.Function):
        @staticmethod
        def forward(ctx, spectrum):
            value = torch.complex(spectrum[..., 0], spectrum[..., 1])
            return torch.fft.irfft(value, n=1024, dim=1)

        @staticmethod
        def symbolic(graph, spectrum):
            endpoints = torch.ones(513, 2, dtype=torch.float32)
            endpoints[0, 1] = 0.0
            endpoints[-1, 1] = 0.0
            values = graph.op('Mul', spectrum, graph.op('Constant', value_t=endpoints))
            indices = graph.op('Constant', value_t=torch.arange(511, 0, -1, dtype=torch.int64))
            reflected = graph.op('Gather', values, indices, axis_i=1)
            sign = graph.op('Constant', value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
            reflected = graph.op('Mul', reflected, sign)
            full = graph.op('Concat', values, reflected, axis_i=1)
            length = graph.op('Constant', value_t=torch.tensor(1024, dtype=torch.int64))
            inverse = graph.op('DFT', full, length, axis_i=1, inverse_i=1, onesided_i=0)
            real_index = graph.op('Constant', value_t=torch.tensor(0, dtype=torch.int64))
            result = graph.op('Gather', inverse, real_index, axis_i=2)
            return result.setType(spectrum.type().with_sizes([spectrum.type().sizes()[0], 1024]))

    class Latency58StreamingWrapper(torch.nn.Module):
        def __init__(self, copied):
            super().__init__()
            self.model = copied

        def forward(self, audio_chunk, *states):
            audio_history, fusion_hidden, spectral_numerator_tail, waveform_tail = states[:4]
            copied = self.model
            joined = torch.cat((audio_history, audio_chunk), dim=-1)
            feature_ri = _RFFT1024.apply((joined * copied.analysis_window).reshape(2, 1024))
            feature_ri = feature_ri.reshape(1, 2, 1, 513, 2)
            packed = feature_ri.permute(0, 2, 1, 3, 4).reshape(1, 1, 2052)
            spec = copied.spec_encode(packed)
            to_relu, to_sigmoid = copied.conv_encode(joined).chunk(2, dim=1)
            basis = to_relu.relu() * to_sigmoid.sigmoid()
            waveform = copied.basis_to_embed(basis).transpose(1, 2)
            fusion_input = torch.cat((spec, waveform), dim=-1)
            recurrent, next_physical_hidden = copied.fusion_branch(
                fusion_input, fusion_hidden / PUBLIC_FUSION_SCALE)
            fused = fusion_input + recurrent
            fused = fused + copied.refinement(fused)
            private_spec, private_waveform = fused.chunk(2, dim=-1)
            private_spec = copied.spec_norm(private_spec + spec)
            private_waveform = copied.waveform_norm(private_waveform + waveform)
            spec_memory, next_spec_hidden = copied.spec_memory(
                private_spec, states[6] / PUBLIC_FUSION_SCALE)
            waveform_memory, next_waveform_hidden = copied.waveform_memory(
                private_waveform, states[7] / PUBLIC_FUSION_SCALE)
            spec_correction = copied.spec_memory_output(spec_memory)
            waveform_correction = copied.waveform_memory_output(waveform_memory)
            correction, keys, values = copied.attention(
                fused, states[4], states[5], tail_only=True)
            fused = fused + correction
            extra_states = (keys, values)
            fused_spec, fused_waveform = fused.chunk(2, dim=-1)
            spec = fused_spec + spec
            waveform = fused_waveform + waveform

            spec_features = copied.spec_norm(spec) + spec_correction
            logits = copied.to_spec_masks(spec_features).reshape(1, 1, 2, 513, 2, 4)
            masks = copied._residual_source_softmax(logits).permute(0, 2, 1, 3, 4, 5)
            phase = copied.phase_coefficients(spec_features)
            quadrature = torch.stack((-feature_ri[..., 1], feature_ri[..., 0]), -1)
            masked_ri = feature_ri.unsqueeze(-1) * masks + quadrature.unsqueeze(-1) * phase.unsqueeze(-2)
            filter_state = ()
            if copied.has_past_filter:
                carrier_residual, next_history = one_hop_shared_filter(
                    copied.past_filter, spec_features, masks, feature_ri, states[8])
                masked_ri = masked_ri + carrier_residual
                filter_state = (next_history,)
            source_ri = masked_ri.permute(0, 5, 1, 2, 3, 4).contiguous().reshape(8, 513, 2)
            frames = IRFFT1024.apply(source_ri).reshape(1, 4, 2, 1024)[..., 1024 - SYNTHESIS_SAMPLES:1024]
            frames = frames * copied.synthesis.spectral_window
            spectral = (frames[..., :HOP] + spectral_numerator_tail) / copied.synthesis.spectral_denominator
            next_spectral_tail = frames[..., HOP:].clone()

            waveform_logits = copied.to_waveform_masks(copied.waveform_norm(waveform) + waveform_correction)
            waveform_logits = waveform_logits.reshape(1, 1, 4, 1500).transpose(-1, -2)
            waveform_masks = copied._residual_source_softmax(waveform_logits)
            source_basis = basis.transpose(1, 2).unsqueeze(-1) * waveform_masks
            source_basis = source_basis.permute(0, 3, 1, 2)
            decoded = torch.nn.functional.linear(source_basis, copied.waveform_decoder_weight.flatten(1).t(), bias=None)
            decoded = decoded.reshape(1, 4, 1, 2, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
            windowed = (decoded * copied.synthesis.window).reshape(1, 4, 2, SYNTHESIS_SAMPLES)
            waveform_audio = windowed[..., :HOP] + waveform_tail
            next_waveform_tail = windowed[..., HOP:].clone()

            raw = (spectral + waveform_audio) * copied.output_source_scales[None, :, None, None]
            delayed_mixture = audio_history[..., -HOP:]
            _, deployed = corrected_estimates(raw, delayed_mixture, copied.fixed_residual_share)
            return (deployed, joined[..., -FEATURE_HISTORY:].clone(),
                    next_physical_hidden * PUBLIC_FUSION_SCALE,
                    next_spectral_tail, next_waveform_tail, *extra_states,
                    next_spec_hidden * PUBLIC_FUSION_SCALE, next_waveform_hidden * PUBLIC_FUSION_SCALE, *filter_state)

    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        copied = copy.deepcopy(model).cpu().eval()
        copied.fusion_branch = export_helpers.OneFrameGRUForONNX(copied.fusion_branch)
        copied.spec_memory = export_helpers.OneFrameGRUForONNX(copied.spec_memory)
        copied.waveform_memory = export_helpers.OneFrameGRUForONNX(copied.waveform_memory)
        export_helpers.replace_rmsnorm_layers(copied)
        wrapper = Latency58StreamingWrapper(copied).eval()
    require(model_state_sha256(copied) == model_state_sha256(model), 'Export copy changed checkpoint tensor bytes')
    return wrapper
