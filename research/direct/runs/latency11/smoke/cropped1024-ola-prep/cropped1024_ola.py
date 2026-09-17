"""Isolated full513-mask Hann1024 carrier with cropped Hann512/hop256 OLA.

No model is constructed on import. The authenticated C91 factory is explicit.
The common four-state/output containers describe the structural callback API;
this family's spectral state and checkpoint identity differ from old OLA512.
No export, native-host qualification, optimizer or training loop is provided.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from research.direct.latency_ola512 import (
    OLA512Model, OLA512Output, OLA512State, OLA512Synthesis, overlap_frames,
    EXPECTED_C91_CONFIG, REFINED_C91, REFINED_C91_SHA256, file_sha256,
    HOP, FEATURE_HISTORY, FEATURE_SAMPLES, SYNTHESIS_SAMPLES,
    CHANNELS, SOURCES, BASIS, EMBED, PUBLIC_FUSION_SCALE, SOURCE_ORDER,
)


VERSION = 'ola-cropped1024-hann512-hop256-v1'
CARRIER_SAMPLES = 1024
MASK_BINS = 513
CROP_START = 512
INITIALIZATION = 'authenticated_c91_full513_right_baked512_cropped1024_ola'
PARAMETER_SHAPES = {
    'waveform_decoder_weight': (1500, 2, 512),
    'spec_encode.weight': (500, 2052), 'spec_encode.bias': (500,),
    'conv_encode.weight': (3000, 2, 1024), 'conv_encode.bias': (3000,),
    'basis_to_embed.weight': (500, 1500, 1), 'basis_to_embed.bias': (500,),
    **{f'fusion_branch.{name}_l{layer}': shape for layer in (0, 1) for name, shape in (
        ('weight_ih', (3000, 1000)), ('weight_hh', (3000, 1000)),
        ('bias_ih', (3000,)), ('bias_hh', (3000,)))},
    'spec_norm.weight': (500,), 'to_spec_masks.weight': (8208, 500), 'to_spec_masks.bias': (8208,),
    'waveform_norm.weight': (500,), 'to_waveform_masks.weight': (6000, 500), 'to_waveform_masks.bias': (6000,),
}
BUFFER_SHAPES = {'analysis_window': (1024,), 'output_source_scales': (4,),
                 'synthesis.window': (512,), 'synthesis.spectral_denominator': (256,),
                 'synthesis.waveform_window_sum': (256,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Cropped1024Synthesis(OLA512Synthesis):
    """Full1024 inverse, last512 crop and independent numerator normalization.

The inherited waveform() is used unchanged. The carrier window is supplied by
the owning model/caller so it is not registered as a duplicate persistent buffer.
"""

    def __init__(self, analysis_window: Tensor):
        require(isinstance(analysis_window, Tensor) and analysis_window.shape == (CARRIER_SAMPLES,)
                and analysis_window.dtype in (torch.float32, torch.float64),
                'Expected one FP32/FP64 Hann1024 analysis window')
        super().__init__(dtype=analysis_window.dtype)
        self.rebuild_denominator(analysis_window)

    def rebuild_denominator(self, analysis_window: Tensor):
        """Derive the buffer from the actual copied analysis/synthesis values."""
        require(analysis_window.shape == (CARRIER_SAMPLES,)
                and analysis_window.dtype == self.window.dtype and analysis_window.device == self.window.device
                and bool(torch.isfinite(analysis_window).all()), 'Analysis/carrier window contract differs')
        denominator = (analysis_window[512:768] * self.window[:HOP]
                       + analysis_window[768:1024] * self.window[HOP:])
        require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
                'Cropped spectral denominator must be finite and strictly positive')
        with torch.no_grad():
            self.spectral_denominator.copy_(denominator)

    def carrier(self, audio_with_past: Tensor, analysis_window: Tensor) -> Tensor:
        """Trailing Hann1024 frames from [...,768+T], with a literal256 hop."""
        require(audio_with_past.ndim >= 2 and audio_with_past.shape[-1] >= CARRIER_SAMPLES
                and (audio_with_past.shape[-1] - FEATURE_HISTORY) % HOP == 0
                and audio_with_past.dtype == self.window.dtype
                and audio_with_past.device == self.window.device
                and analysis_window.shape == (CARRIER_SAMPLES,)
                and analysis_window.dtype == self.window.dtype and analysis_window.device == self.window.device,
                'Carrier expects matching [...,768+T] physical audio and Hann1024')
        frames = audio_with_past.unfold(-1, CARRIER_SAMPLES, HOP)
        return torch.fft.rfft(frames * analysis_window, n=CARRIER_SAMPLES, dim=-1)

    def spectral(self, spectrum: Tensor, previous_numerator_tail: Tensor) -> tuple[Tensor, Tensor]:
        require(spectrum.ndim >= 2 and spectrum.is_complex() and spectrum.shape[-1] == MASK_BINS
                and spectrum.real.dtype == self.window.dtype and spectrum.device == self.window.device,
                'Cropped spectral synthesis expects matching complex [...,F,513]')
        frames = torch.fft.irfft(spectrum, n=CARRIER_SAMPLES, dim=-1)[..., CROP_START:]
        frames = frames * self.window
        numerator, next_tail = overlap_frames(frames, previous_numerator_tail)
        denominator = self.spectral_denominator.repeat(spectrum.shape[-2])
        return numerator / denominator, next_tail


class Cropped1024OLAModel(OLA512Model):
    """Own constructor/render/identity; immutable OLA structural helpers reused."""

    synthesis_samples = SYNTHESIS_SAMPLES
    carrier_samples = CARRIER_SAMPLES
    spectral_mask_bins = MASK_BINS

    def __init__(self):
        # Deliberately do not construct the old257-bin model and replace its head.
        nn.Module.__init__(self)
        self.spec_encode = nn.Linear(CHANNELS * MASK_BINS * 2, EMBED)
        self.conv_encode = nn.Conv1d(CHANNELS, BASIS * 2, FEATURE_SAMPLES, stride=HOP)
        self.basis_to_embed = nn.Conv1d(BASIS, EMBED, 1)
        self.fusion_branch = nn.GRU(2 * EMBED, 2 * EMBED, num_layers=2, batch_first=True)
        self.spec_norm = nn.RMSNorm(EMBED)
        self.to_spec_masks = nn.Linear(EMBED, CHANNELS * MASK_BINS * 2 * SOURCES)
        self.waveform_norm = nn.RMSNorm(EMBED)
        self.to_waveform_masks = nn.Linear(EMBED, SOURCES * BASIS)
        self.waveform_decoder_weight = nn.Parameter(torch.zeros(BASIS, CHANNELS, SYNTHESIS_SAMPLES))
        self.register_buffer('analysis_window', torch.hann_window(FEATURE_SAMPLES, periodic=True))
        self.register_buffer('output_source_scales', torch.ones(SOURCES, dtype=torch.float32))
        self.synthesis = Cropped1024Synthesis(self.analysis_window)
        self.provenance = {'version': VERSION, 'initialization': 'uninitialized_schema_only'}

    @property
    def architecture_metadata(self) -> dict:
        return {
            'version': VERSION, 'source_order': list(SOURCE_ORDER), 'sample_rate': 44100,
            'feature_n_fft': FEATURE_SAMPLES, 'carrier_n_fft': CARRIER_SAMPLES,
            'synthesis_n_fft': CARRIER_SAMPLES, 'spectral_mask_bins': MASK_BINS,
            'spectral_output_crop': [512, 1024], 'synthesis_frame_samples': SYNTHESIS_SAMPLES,
            'waveform_decoder_samples': SYNTHESIS_SAMPLES, 'hop_samples': HOP,
            'feature_history_samples': FEATURE_HISTORY,
            'carrier_window': 'authenticated stored periodic Hann1024',
            'synthesis_window': 'periodic Hann512',
            'spectral_denominator_rule': 'h1024[512+p]*h512[p]+h1024[768+p]*h512[256+p], p=0..255',
            'graph_alignment_samples': HOP, 'host_queue_samples': HOP,
            'intended_total_latency_samples': 2 * HOP, 'host_queue_implemented_in_this_module': False,
            'future_context_from_output_samples': HOP, 'future_callbacks_beyond_received_input': 0,
            'samplewise_latest_input_minus_output': '511-p for emitted callback sample p=0..255',
            'flush_hops': 1, 'public_fusion_state_scale': PUBLIC_FUSION_SCALE,
            'state_names': list(OLA512State._fields), 'state_family': VERSION,
            'state_interchangeable_with_old_ola512': False,
            'precision': 'float32', 'optional_2048_analysis': False,
            'graph_qualified': False, 'native_host_qualified': False,
        }

    def _render_fp32(self, audio: Tensor, state: OLA512State) -> OLA512Output:
        batch, _, samples = audio.shape
        frame_count = samples // HOP
        joined = torch.cat((state.audio_history, audio), dim=-1)
        feature_frames = joined.unfold(-1, FEATURE_SAMPLES, HOP)
        feature_spec = torch.fft.rfft(feature_frames * self.analysis_window, n=FEATURE_SAMPLES, dim=-1)
        feature_ri = torch.view_as_real(feature_spec)
        packed = feature_ri.permute(0, 2, 1, 3, 4).reshape(batch, frame_count, CHANNELS * MASK_BINS * 2)
        spec = self.spec_encode(packed)
        to_relu, to_sigmoid = self.conv_encode(joined).chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        waveform = self.basis_to_embed(basis).transpose(1, 2)
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        recurrent, next_physical_hidden = self.fusion_branch(fusion_input, physical_hidden)
        fused = fusion_input + recurrent
        fused_spec, fused_waveform = fused.chunk(2, dim=-1)
        spec = fused_spec + spec
        waveform = fused_waveform + waveform

        spec_logits = self.to_spec_masks(self.spec_norm(spec))
        spec_logits = spec_logits.reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        spec_masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        # Reuse the complete complex feature carrier; no separate FFT512 exists.
        masked_ri = feature_ri.unsqueeze(-1) * spec_masks
        source_spectrum = torch.view_as_complex(masked_ri.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, next_spectral_tail = self.synthesis.spectral(source_spectrum, state.spectral_numerator_tail)

        waveform_logits = self.to_waveform_masks(self.waveform_norm(waveform))
        waveform_logits = waveform_logits.reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        waveform_masks = self._residual_source_softmax(waveform_logits)
        source_basis = basis.transpose(1, 2).unsqueeze(-1) * waveform_masks
        source_basis = source_basis.permute(0, 3, 1, 2)
        decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
        waveform_audio, next_waveform_tail = self.synthesis.waveform(decoded, state.waveform_tail)
        scales = self.output_source_scales[None, :, None, None]
        raw = (spectral + waveform_audio) * scales
        delayed_mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        retained = raw[:, :3]
        deployed = torch.cat((retained, delayed_mixture.unsqueeze(1) - retained.sum(dim=1, keepdim=True)), dim=1)
        next_state = OLA512State(joined[..., -FEATURE_HISTORY:].clone(),
            next_physical_hidden * PUBLIC_FUSION_SCALE, next_spectral_tail, next_waveform_tail)
        return OLA512Output(raw, deployed, spectral * scales, waveform_audio * scales, delayed_mixture, next_state)

    @classmethod
    def from_refined_c91(cls, checkpoint: Path = REFINED_C91) -> 'Cropped1024OLAModel':
        checkpoint = Path(checkpoint).resolve()
        require(file_sha256(checkpoint) == REFINED_C91_SHA256, 'Authenticated refined C91 bytes differ before loading')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(isinstance(payload, dict) and set(payload) == {'model', 'config'}, 'C91 payload inventory differs')
        config = pickle.loads(payload['config']) if isinstance(payload['config'], bytes) else payload['config']
        require(config == EXPECTED_C91_CONFIG, 'Exact C91 source configuration differs')
        source = payload['model']
        require(isinstance(source, dict), 'C91 state must be a tensor mapping')
        copies = {
            'analysis_window': 'stft.window', 'output_source_scales': 'output_source_scales',
            'spec_encode.weight': 'spec_encode.1.weight', 'spec_encode.bias': 'spec_encode.1.bias',
            'conv_encode.weight': 'conv_encode.weight', 'conv_encode.bias': 'conv_encode.bias',
            'basis_to_embed.weight': 'basis_to_embed.0.weight', 'basis_to_embed.bias': 'basis_to_embed.0.bias',
            'spec_norm.weight': 'to_spec_masks.0.weight',
            'to_spec_masks.weight': 'to_spec_masks.1.weight', 'to_spec_masks.bias': 'to_spec_masks.1.bias',
            'waveform_norm.weight': 'to_waveform_masks.0.weight',
            'to_waveform_masks.weight': 'to_waveform_masks.1.weight', 'to_waveform_masks.bias': 'to_waveform_masks.1.bias',
            **{f'fusion_branch.{name}_l{layer}': f'fusion_branch.{name}_l{layer}'
               for layer in (0, 1) for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')},
        }
        unused_shapes = {'stft.streaming_envelope': (1024,), 'conv_decode.window': (1024,), 'conv_decode.bias': (2,)}
        require(set(source) == set(copies.values()) | {'conv_decode.weight'} | set(unused_shapes)
                and len(source) == 26, 'Exact C91 source tensor inventory differs')
        require(all(isinstance(value, Tensor) and value.dtype == torch.float32 and value.device.type == 'cpu'
                    and bool(torch.isfinite(value).all()) for value in source.values()), 'C91 tensors must be finite CPU FP32')
        require(source['conv_decode.weight'].shape == (BASIS, CHANNELS, FEATURE_SAMPLES)
                and all(tuple(source[name].shape) == shape for name, shape in unused_shapes.items()),
                'C91 decoder or unused tensor shape differs')
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            model = cls()
        require({name: tuple(value.shape) for name, value in model.named_parameters()} == PARAMETER_SHAPES
                and {name: tuple(value.shape) for name, value in model.named_buffers()} == BUFFER_SHAPES,
                'New family must have exactly21 parameters and five buffers')
        state = model.state_dict()
        with torch.no_grad():
            for destination, original in copies.items():
                require(state[destination].shape == source[original].shape, f'Transfer shape differs: {original}')
                state[destination].copy_(source[original])
            state['waveform_decoder_weight'].copy_(source['conv_decode.weight'][..., 512:1024])
        model.load_state_dict(state, strict=True)
        model.synthesis.rebuild_denominator(model.analysis_window)
        require(file_sha256(checkpoint) == REFINED_C91_SHA256, 'Authenticated C91 changed during initialization')
        model.provenance = {
            'version': VERSION, 'initialization': INITIALIZATION,
            'reference_path': str(checkpoint), 'reference_sha256': REFINED_C91_SHA256,
            'copied_tensors': copies,
            'spectral_projection_rule': 'Copy all513 bins in original stereo/bin/real-imaginary/source order',
            'waveform_parameter_rule': 'new[...,j] = old_baked[...,512+j]',
            'waveform_effective_rule': 'new_parameter[...,j] * periodic_Hann512[j]',
            'spectral_denominator_from_actual_copied_window': True,
            'unused_reference_tensors': sorted(unused_shapes),
            'unused_decoder_bias_reason': 'C91 custom forward omits the stored ConvTranspose bias',
            'training_updates': 0, 'reference_model_modified': False, 'trained_ola_state_used': False,
            'equivalence_claimed': False, 'graph_qualified': False, 'native_host_qualified': False,
        }
        return model.eval()


__all__ = ['Cropped1024OLAModel', 'Cropped1024Synthesis', 'VERSION', 'INITIALIZATION',
           'PARAMETER_SHAPES', 'BUFFER_SHAPES']
