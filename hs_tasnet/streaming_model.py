"""Trainable PyTorch definition of the released stereo streaming architecture.

The model emits audio with 128 samples of delay. Playback queueing belongs to
its host. ``render`` exposes both raw four-stem predictions and the deployed
mixture-consistent outputs; ``forward`` returns deployed audio and all states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION = "hs-tasnet-stereo-hop128-v1"
HOP = 128
FEATURE_SAMPLES = 1024
FEATURE_HISTORY = 896
SYNTHESIS_SAMPLES = 256
CROP_START = 768
MASK_BINS = 513
CHANNELS, SOURCES, BASIS, EMBED = 2, 4, 1500, 500
PUBLIC_FUSION_SCALE = 2.0**-18
SOURCE_ORDER = ("drums", "bass", "vocals", "other")


def require(condition, message):
    if not condition:
        raise ValueError(message)



def asymmetric_windows(*, dtype=torch.float32):
    """K=1024, M=128, d=0 instance of Wang et al. arXiv:2106.11794.

    The spectral synthesis window is separate from the unchanged waveform
    decoder's Hann256. Their analysis/synthesis product is Hann256.
    """
    require(dtype in (torch.float32, torch.float64), "Use FP32 or FP64 windows")
    n = torch.arange(1024, dtype=torch.float64, device="cpu")
    analysis = torch.where(n < 896, torch.sin(torch.pi * n / 1792),
                           torch.cos(torch.pi * (n - 896) / 256))
    prototype = 0.5 * (1 - torch.cos(torch.pi * torch.arange(256, dtype=torch.float64) / 128))
    synthesis = torch.cat((prototype[:128] / analysis[768:896], analysis[896:]))
    return analysis.to(dtype), synthesis.to(dtype)


def overlap_frames(frames: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
    require(frames.ndim >= 2 and frames.shape[-1] == SYNTHESIS_SAMPLES
            and frames.shape[-2] >= 1
            and previous_tail.shape == (*frames.shape[:-2], HOP)
            and previous_tail.dtype == frames.dtype
            and previous_tail.device == frames.device,
            "Expected [...,F,256] frames and matching [...,128] tail")
    left, right = frames[..., :HOP], frames[..., HOP:]
    prior = torch.cat((previous_tail.unsqueeze(-2), right[..., :-1, :]), dim=-2)
    emitted = (left + prior).reshape(*frames.shape[:-2], frames.shape[-2] * HOP)
    return emitted, right[..., -1, :].clone()


class _Synthesis(nn.Module):
    def __init__(self, analysis_window, spectral_window):
        torch.nn.Module.__init__(self)
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and analysis_window.dtype in (torch.float32, torch.float64), "Expected FP32/FP64 analysis")
        waveform = torch.hann_window(SYNTHESIS_SAMPLES, periodic=True,
                                     dtype=analysis_window.dtype, device=analysis_window.device)
        self.register_buffer("window", waveform)
        self.register_buffer("spectral_window", spectral_window.clone())
        self.register_buffer("spectral_denominator", torch.zeros_like(waveform[:HOP]))
        self.register_buffer("waveform_window_sum", waveform[:HOP] + waveform[HOP:])
        self.rebuild_denominator(analysis_window)

    def rebuild_denominator(self, analysis_window):
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and self.spectral_window.shape == (SYNTHESIS_SAMPLES,)
                and all(t.dtype == self.window.dtype and t.device == self.window.device
                        and bool(torch.isfinite(t).all()) for t in (analysis_window, self.spectral_window)),
                "Analysis and spectral synthesis window contracts differ")
        product = analysis_window[CROP_START:] * self.spectral_window
        denominator = product[:HOP] + product[HOP:]
        require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
                "Spectral overlap divisor must be finite and positive")
        with torch.no_grad():
            self.spectral_denominator.copy_(denominator)

    def spectral(self, spectrum, previous_tail):
        require(spectrum.ndim >= 2 and spectrum.is_complex() and spectrum.shape[-1] == MASK_BINS
                and spectrum.real.dtype == self.window.dtype and spectrum.device == self.window.device,
                "Expected matching complex [...,F,513] spectrum")
        frames = torch.fft.irfft(spectrum, n=FEATURE_SAMPLES, dim=-1)[..., CROP_START:]
        numerator, tail = overlap_frames(frames * self.spectral_window, previous_tail)
        return numerator / self.spectral_denominator.repeat(spectrum.shape[-2]), tail

    def waveform(self, frames: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
        require(frames.dtype == self.window.dtype and frames.device == self.window.device,
                "Waveform frames and synthesis window must share dtype/device")
        return overlap_frames(frames * self.window, previous_tail)


class StreamingState(NamedTuple):
    audio_history: Tensor             # [B,2,896]
    fusion_hidden: Tensor             # [2,B,1000], physical hidden * 2**-18
    spectral_numerator_tail: Tensor   # [B,4,2,128], before denominator and gains
    waveform_tail: Tensor             # [B,4,2,128], after Hann, before gains

    def detached(self) -> "StreamingState":
        return StreamingState(*(value.detach() for value in self))


@dataclass(frozen=True)
class StreamingOutput:
    raw: Tensor
    deployed: Tensor
    spectral: Tensor
    waveform: Tensor
    delayed_mixture: Tensor
    state: StreamingState


class StreamingHSTasNet(nn.Module):
    """Stereo, four-source streaming network with FP32 audio and explicit state.

    Constructing this class initializes untrained weights. Use ``from_checkpoint``
    to load trained weights. Set ``training_precision = "bf16"`` for the CUDA
    training policy; learned kernels use BF16 while parameters, FFT, synthesis,
    nonlinear masks and recurrent state remain FP32. Evaluation always uses FP32.
    """
    sample_rate = 44100
    hop_samples = HOP
    graph_alignment_samples = HOP
    algorithmic_latency_samples = HOP
    source_order = SOURCE_ORDER
    training_precision = "fp32"

    def __init__(self):
        super().__init__()
        require(torch.get_default_dtype() == torch.float32, "Construct the streaming model with the FP32 default dtype")
        self.spec_encode = nn.Linear(CHANNELS * MASK_BINS * 2, EMBED)
        self.conv_encode = nn.Conv1d(CHANNELS, BASIS * 2, FEATURE_SAMPLES, stride=HOP)
        self.basis_to_embed = nn.Conv1d(BASIS, EMBED, 1)
        self.fusion_branch = nn.GRU(2 * EMBED, 2 * EMBED, num_layers=2, batch_first=True)
        self.spec_norm = nn.RMSNorm(EMBED)
        self.to_spec_masks = nn.Linear(EMBED, CHANNELS * MASK_BINS * 2 * SOURCES)
        self.waveform_norm = nn.RMSNorm(EMBED)
        self.to_waveform_masks = nn.Linear(EMBED, SOURCES * BASIS)
        self.waveform_decoder_weight = nn.Parameter(torch.zeros(BASIS, CHANNELS, SYNTHESIS_SAMPLES))
        analysis, spectral = asymmetric_windows()
        self.register_buffer("analysis_window", analysis)
        self.register_buffer("output_source_scales", torch.ones(SOURCES, dtype=torch.float32))
        self.synthesis = _Synthesis(self.analysis_window, spectral)
        self.training_precision = "fp32"

    def initial_state(self, batch_size: int, *, device=None) -> StreamingState:
        require(type(batch_size) is int and batch_size > 0, "Expected positive batch size")
        device = self.output_source_scales.device if device is None else torch.device(device)
        require(device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "State uses model device and FP32")
        shapes = ((batch_size, CHANNELS, FEATURE_HISTORY), (2, batch_size, 2 * EMBED),
                  (batch_size, SOURCES, CHANNELS, HOP), (batch_size, SOURCES, CHANNELS, HOP))
        return StreamingState(*(torch.zeros(shape, dtype=torch.float32, device=device) for shape in shapes))

    def _validate(self, audio: Tensor, state: StreamingState | None) -> StreamingState:
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32,
                "Expected FP32 [B,2,T] audio, T a positive multiple of 128, on model device")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(isinstance(state, StreamingState), "Use the distinct StreamingState family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes)), "State shape/dtype/device differs")
        return state

    @staticmethod
    def _residual_source_softmax(logits: Tensor) -> Tensor:
        return logits.add(torch.softmax(logits, dim=-1), alpha=float(SOURCES))

    def _render_fp32(self, audio: Tensor, state: StreamingState) -> StreamingOutput:
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
        recurrent, hidden = self.fusion_branch(fusion_input, state.fusion_hidden / PUBLIC_FUSION_SCALE)
        fused_spec, fused_waveform = (fusion_input + recurrent).chunk(2, dim=-1)
        spec, waveform = fused_spec + spec, fused_waveform + waveform

        spec_logits = self.to_spec_masks(self.spec_norm(spec))
        spec_logits = spec_logits.reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        spec_masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        masked_ri = feature_ri.unsqueeze(-1) * spec_masks
        spectrum = torch.view_as_complex(masked_ri.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spectral_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)

        wave_logits = self.to_waveform_masks(self.waveform_norm(waveform))
        wave_logits = wave_logits.reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        source_basis = basis.transpose(1, 2).unsqueeze(-1) * self._residual_source_softmax(wave_logits)
        source_basis = source_basis.permute(0, 3, 1, 2)
        decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t())
        decoded = decoded.reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES)
        waveform_audio, waveform_tail = self.synthesis.waveform(decoded.permute(0, 1, 3, 2, 4), state.waveform_tail)

        scales = self.output_source_scales[None, :, None, None]
        raw = (spectral + waveform_audio) * scales
        delayed = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        retained = raw[:, :3]
        deployed = torch.cat((retained, delayed.unsqueeze(1) - retained.sum(dim=1, keepdim=True)), dim=1)
        next_state = StreamingState(joined[..., -FEATURE_HISTORY:].clone(),
                                    hidden * PUBLIC_FUSION_SCALE, spectral_tail, waveform_tail)
        return StreamingOutput(raw, deployed, spectral * scales, waveform_audio * scales, delayed, next_state)

    def forward(self, audio: Tensor, state: StreamingState | None = None, *, return_raw=False):
        output = self.render(audio, state)
        return (output.raw if return_raw else output.deployed), output.state

    def forward_chunk(self, audio: Tensor, state: StreamingState | None = None, *, return_raw=False):
        require(audio.ndim == 3 and audio.shape[-1] == HOP, "Literal input requires exactly 128 samples")
        return self.forward(audio, state, return_raw=return_raw)

    def flush(self, state: StreamingState, *, return_raw=False):
        require(isinstance(state, StreamingState), "Flush requires StreamingState")
        zeros = state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP))
        return self.forward_chunk(zeros, state, return_raw=return_raw)

    def _render_bf16(self, audio, state):
        if self.training_precision != "bf16" or audio.device.type != "cuda":
            raise ValueError("BF16 OLA training requires CUDA; no CPU training fallback")
        # Base render() already disabled autocast. Enable only learned dense
        # kernels below; FFT, nonlinear masks and explicit states stay FP32.
        batch, _, samples = audio.shape
        frame_count = samples // HOP
        joined = torch.cat((state.audio_history, audio), dim=-1)
        frames = joined.unfold(-1, FEATURE_SAMPLES, HOP)
        feature = torch.fft.rfft(frames * self.analysis_window, n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spec = self.spec_encode(packed)
            convolution = self.conv_encode(joined)
        to_relu, to_sigmoid = convolution.float().chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            waveform = self.basis_to_embed(basis).transpose(1, 2)
        spec, waveform = spec.float(), waveform.float()
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recurrent, next_hidden = self.fusion_branch(
                fusion_input.to(torch.bfloat16), physical_hidden.to(torch.bfloat16))
        fused_spec, fused_waveform = (fusion_input + recurrent.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec)
        waveform = self.waveform_norm(fused_waveform + waveform)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spec_logits = self.to_spec_masks(spec)
            waveform_logits = self.to_waveform_masks(waveform)
        spec_logits = spec_logits.float().reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        carrier = feature
        masked = torch.view_as_real(carrier).unsqueeze(-1) * masks
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spec_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)
        waveform_logits = waveform_logits.float().reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        masks = self._residual_source_softmax(waveform_logits)
        source_basis = (basis.transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.float().reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
        waveform_audio, wave_tail = self.synthesis.waveform(decoded, state.waveform_tail)
        scales = self.output_source_scales[None, :, None, None]
        raw = (spectral + waveform_audio) * scales
        mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        dbv = raw[:, :3]
        deployed = torch.cat((dbv, mixture.unsqueeze(1) - dbv.sum(dim=1, keepdim=True)), dim=1)
        next_state = StreamingState(joined[..., -FEATURE_HISTORY:].clone(),
                                 next_hidden.float() * PUBLIC_FUSION_SCALE, spec_tail, wave_tail)
        return StreamingOutput(raw, deployed, spectral * scales, waveform_audio * scales, mixture, next_state)

    def render(self, audio: Tensor, state: StreamingState | None = None) -> StreamingOutput:
        state = self._validate(audio, state)
        if not torch.isfinite(audio).all() or any(not torch.isfinite(value).all() for value in state):
            raise ValueError("Audio and state must be finite")
        with torch.autocast(audio.device.type, enabled=False):
            if self.training and self.training_precision != "fp32":
                return self._render_bf16(audio, state)
            return self._render_fp32(audio, state)

    @torch.no_grad()
    def separate(self, audio: Tensor) -> Tensor:
        """Separate [B,2,T] from zero state, returning aligned [B,4,2,T]."""
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and bool(torch.isfinite(audio).all()), "Expected finite FP32 [B,2,T] audio on the model device")
        count = audio.shape[-1]
        if count == 0:
            return audio.new_empty((audio.shape[0], SOURCES, CHANNELS, 0))
        padding = (-count) % HOP
        output = self.render(F.pad(audio, (0, padding + HOP)))
        return output.deployed[..., HOP:HOP + count]

    @classmethod
    def from_checkpoint(cls, path):
        """Load a portable PyTorch checkpoint on CPU, ready for evaluation."""
        from .streaming_checkpoint import load_streaming_checkpoint
        return load_streaming_checkpoint(path)
