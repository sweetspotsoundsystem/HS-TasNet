"""Research-only full1024 carrier with cropped256 synthesis and hop128.

This distinct state family has 128 samples of graph delay. A future 128-sample
worker queue would give 256 total samples. No host, training, export, or model
loading runs on import. The accepted 11.6 ms model is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VERSION = "cropped1024-hann256-hop128-v1"
HOP = 128
FEATURE_SAMPLES = 1024
FEATURE_HISTORY = FEATURE_SAMPLES - HOP
SYNTHESIS_SAMPLES = 2 * HOP
CROP_START = FEATURE_SAMPLES - SYNTHESIS_SAMPLES
MASK_BINS = 513
CHANNELS, SOURCES, BASIS, EMBED = 2, 4, 1500, 500
PUBLIC_FUSION_SCALE = 2.0**-18
SOURCE_ORDER = ("drums", "bass", "vocals", "other")
ACCEPTED_CHECKPOINT = Path(__file__).resolve().parent / (
    "runs/latency11/cropped1024-matched-raw4_control-b4-bf16-lr3e-5/"
    "checkpoints/step-000250/model.pt"
)
ACCEPTED_SHA256 = "ac46729e5e4d379b09914a6e40ae927e09089b43fd4eef219ae7e034f355da65"


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


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


class Cropped256Synthesis(nn.Module):
    def __init__(self, analysis_window: Tensor):
        super().__init__()
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and analysis_window.dtype in (torch.float32, torch.float64),
                "Expected FP32/FP64 Hann1024")
        window = torch.hann_window(SYNTHESIS_SAMPLES, periodic=True,
                                   dtype=analysis_window.dtype, device=analysis_window.device)
        self.register_buffer("window", window)
        self.register_buffer("spectral_denominator", torch.zeros_like(window[:HOP]))
        self.register_buffer("waveform_window_sum", window[:HOP] + window[HOP:])
        self.rebuild_denominator(analysis_window)

    def rebuild_denominator(self, analysis_window: Tensor):
        require(analysis_window.shape == (FEATURE_SAMPLES,)
                and analysis_window.dtype == self.window.dtype
                and analysis_window.device == self.window.device,
                "Analysis and synthesis window contracts differ")
        denominator = (analysis_window[CROP_START:CROP_START + HOP] * self.window[:HOP]
                       + analysis_window[CROP_START + HOP:] * self.window[HOP:])
        require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
                "The cropped spectral divisor must be finite and positive")
        with torch.no_grad():
            self.spectral_denominator.copy_(denominator)

    def spectral(self, spectrum: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
        require(spectrum.ndim >= 2 and spectrum.is_complex()
                and spectrum.shape[-1] == MASK_BINS
                and spectrum.real.dtype == self.window.dtype
                and spectrum.device == self.window.device,
                "Expected matching complex [...,F,513] spectrum")
        frames = torch.fft.irfft(spectrum, n=FEATURE_SAMPLES, dim=-1)[..., CROP_START:]
        numerator, next_tail = overlap_frames(frames * self.window, previous_tail)
        return numerator / self.spectral_denominator.repeat(spectrum.shape[-2]), next_tail

    def waveform(self, frames: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
        require(frames.dtype == self.window.dtype and frames.device == self.window.device,
                "Waveform frames and synthesis window must share dtype/device")
        return overlap_frames(frames * self.window, previous_tail)


class Latency58State(NamedTuple):
    audio_history: Tensor             # [B,2,896]
    fusion_hidden: Tensor             # [2,B,1000], physical hidden * 2**-18
    spectral_numerator_tail: Tensor   # [B,4,2,128], before denominator and gains
    waveform_tail: Tensor             # [B,4,2,128], after Hann, before gains

    def detached(self) -> "Latency58State":
        return Latency58State(*(value.detach() for value in self))


@dataclass(frozen=True)
class Latency58Output:
    raw: Tensor
    deployed: Tensor
    spectral: Tensor
    waveform: Tensor
    delayed_mixture: Tensor
    state: Latency58State


class Latency58Model(nn.Module):
    sample_rate = 44100
    hop_samples = HOP
    synthesis_samples = SYNTHESIS_SAMPLES
    feature_samples = FEATURE_SAMPLES
    graph_alignment_samples = HOP
    alignment_samples = HOP
    host_queue_samples = HOP
    algorithmic_latency_samples = 2 * HOP
    host_visible_pdc_samples = 2 * HOP
    public_fusion_state_scale = PUBLIC_FUSION_SCALE
    future_context_samples = HOP
    future_callbacks_beyond_received_input = 0
    flush_required = True
    flush_hops = 1

    def __init__(self):
        super().__init__()
        self.spec_encode = nn.Linear(CHANNELS * MASK_BINS * 2, EMBED)
        self.conv_encode = nn.Conv1d(CHANNELS, BASIS * 2, FEATURE_SAMPLES, stride=HOP)
        self.basis_to_embed = nn.Conv1d(BASIS, EMBED, 1)
        self.fusion_branch = nn.GRU(2 * EMBED, 2 * EMBED, num_layers=2, batch_first=True)
        self.spec_norm = nn.RMSNorm(EMBED)
        self.to_spec_masks = nn.Linear(EMBED, CHANNELS * MASK_BINS * 2 * SOURCES)
        self.waveform_norm = nn.RMSNorm(EMBED)
        self.to_waveform_masks = nn.Linear(EMBED, SOURCES * BASIS)
        self.waveform_decoder_weight = nn.Parameter(torch.zeros(BASIS, CHANNELS, SYNTHESIS_SAMPLES))
        self.register_buffer("analysis_window", torch.hann_window(FEATURE_SAMPLES, periodic=True))
        self.register_buffer("output_source_scales", torch.ones(SOURCES, dtype=torch.float32))
        self.synthesis = Cropped256Synthesis(self.analysis_window)
        self.provenance = {"version": VERSION, "initialization": "uninitialized_schema_only"}

    @property
    def architecture_metadata(self) -> dict:
        return {
            "version": VERSION, "state_family": VERSION, "sample_rate": self.sample_rate,
            "source_order": list(SOURCE_ORDER), "feature_n_fft": FEATURE_SAMPLES,
            "carrier_n_fft": FEATURE_SAMPLES, "spectral_mask_bins": MASK_BINS,
            "spectral_output_crop": [CROP_START, FEATURE_SAMPLES],
            "synthesis_frame_samples": SYNTHESIS_SAMPLES, "hop_samples": HOP,
            "feature_history_samples": FEATURE_HISTORY, "graph_alignment_samples": HOP,
            "host_queue_samples": HOP, "intended_total_latency_samples": 2 * HOP,
            "host_queue_implemented_in_this_module": False,
            "future_callbacks_beyond_received_input": 0,
            "samplewise_latest_input_minus_output": "255-p for output callback sample p=0..127",
            "state_names": list(Latency58State._fields), "flush_hops": 1,
            "public_fusion_state_scale": PUBLIC_FUSION_SCALE, "precision": "float32",
            "native_host_qualified": False,
        }

    def initial_state(self, batch_size: int, *, device=None) -> Latency58State:
        require(type(batch_size) is int and batch_size > 0, "Expected positive batch size")
        device = self.output_source_scales.device if device is None else torch.device(device)
        require(device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "State uses model device and FP32")
        shapes = ((batch_size, CHANNELS, FEATURE_HISTORY), (2, batch_size, 2 * EMBED),
                  (batch_size, SOURCES, CHANNELS, HOP), (batch_size, SOURCES, CHANNELS, HOP))
        return Latency58State(*(torch.zeros(shape, dtype=torch.float32, device=device) for shape in shapes))

    def _validate(self, audio: Tensor, state: Latency58State | None) -> Latency58State:
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32,
                "Expected FP32 [B,2,T] audio, T a positive multiple of 128, on model device")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(isinstance(state, Latency58State), "Use the distinct Latency58State family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "State shape/dtype/device differs")
        return state

    @staticmethod
    def _residual_source_softmax(logits: Tensor) -> Tensor:
        return logits.add(torch.softmax(logits, dim=-1), alpha=float(SOURCES))

    def render(self, audio: Tensor, state: Latency58State | None = None) -> Latency58Output:
        state = self._validate(audio, state)
        with torch.autocast(audio.device.type, enabled=False):
            return self._render_fp32(audio, state)

    def _render_fp32(self, audio: Tensor, state: Latency58State) -> Latency58Output:
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
        next_state = Latency58State(joined[..., -FEATURE_HISTORY:].clone(),
                                    hidden * PUBLIC_FUSION_SCALE, spectral_tail, waveform_tail)
        return Latency58Output(raw, deployed, spectral * scales, waveform_audio * scales, delayed, next_state)

    def forward(self, audio: Tensor, state: Latency58State | None = None, *, return_raw=False):
        output = self.render(audio, state)
        return (output.raw if return_raw else output.deployed), output.state

    def forward_chunk(self, audio: Tensor, state: Latency58State | None = None, *, return_raw=False):
        require(audio.ndim == 3 and audio.shape[-1] == HOP, "Literal input requires exactly 128 samples")
        return self.forward(audio, state, return_raw=return_raw)

    def flush(self, state: Latency58State, *, return_raw=False):
        require(isinstance(state, Latency58State), "Flush requires Latency58State")
        zeros = state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP))
        return self.forward_chunk(zeros, state, return_raw=return_raw)

    @classmethod
    def from_accepted(cls, checkpoint: Path = ACCEPTED_CHECKPOINT) -> "Latency58Model":
        """Initialize new CPU weights from authenticated accepted deployment bytes."""
        require(torch.get_default_dtype() == torch.float32, "Initializer requires the FP32 default")
        checkpoint = Path(checkpoint).resolve()
        require(file_sha256(checkpoint) == ACCEPTED_SHA256, "Accepted checkpoint identity differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict) and payload.get("schema") == "cropped1024-ola-inference-v1"
                and payload.get("step") == 250
                and payload.get("architecture", {}).get("version") == "ola-cropped1024-hann512-hop256-v1"
                and payload.get("provenance", {}).get("training_updates") == 2250,
                "Accepted checkpoint schema/family/update count differs")
        source = payload["model"]
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        state = model.state_dict()
        require(isinstance(source, dict) and set(source) == set(state), "Accepted tensor inventory differs")
        with torch.no_grad():
            for name, destination in state.items():
                original = source[name]
                require(isinstance(original, Tensor) and original.dtype == torch.float32
                        and original.device.type == "cpu" and bool(torch.isfinite(original).all()),
                        "Accepted tensors must be finite CPU FP32")
                if name.startswith("synthesis."):
                    continue
                if name == "waveform_decoder_weight":
                    require(original.shape == (BASIS, CHANNELS, 512), "Accepted decoder shape differs")
                    destination.copy_(original[..., -SYNTHESIS_SAMPLES:])
                else:
                    require(original.shape == destination.shape, "Transfer shape differs: " + name)
                    destination.copy_(original)
        model.load_state_dict(state, strict=True)
        model.synthesis.rebuild_denominator(model.analysis_window)
        require(file_sha256(checkpoint) == ACCEPTED_SHA256, "Accepted checkpoint changed during loading")
        model.provenance = {
            "version": VERSION, "initialization": "accepted_raw4_2250_last256_decoder",
            "parent_checkpoint": str(checkpoint), "parent_checkpoint_sha256": ACCEPTED_SHA256,
            "parent_model_state_sha256": payload["model_state_sha256"],
            "parent_training_updates": 2250, "pilot_updates": 0,
            "equivalence_claimed": False,
        }
        return model
