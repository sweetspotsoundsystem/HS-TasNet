"""Isolated OLA512/hop256 functional prototype; no common loader integration.

The separator sees trailing 1024-sample feature frames.  Local 512-sample
Hann carriers and a Hann-windowed waveform decoder overlap at hop 256.
Outputs have 256 samples of graph delay.  A separately implemented 256-sample
host queue would make the intended total latency 512; this module does not
implement or qualify that host queue.

Nothing is loaded or executed on import.  ``from_refined_c91`` authenticates
the preserved reference before deserializing it and constructs a NEW model.
The initializer is deliberately not equivalent to C91 or native C191.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VERSION = "ola512-hop256-feature1024-folded-wave-v1"
SAMPLE_RATE = 44_100
HOP = 256
SYNTHESIS_SAMPLES = 512
FEATURE_SAMPLES = 1024
FEATURE_HISTORY = FEATURE_SAMPLES - HOP
SOURCES = 4
CHANNELS = 2
BASIS = 1500
EMBED = 500
FEATURE_BINS = FEATURE_SAMPLES // 2 + 1
SYNTHESIS_BINS = SYNTHESIS_SAMPLES // 2 + 1
PUBLIC_FUSION_SCALE = 2.0**-18
SOURCE_ORDER = ("drums", "bass", "vocals", "other")
REFINED_C91 = Path(__file__).resolve().parent / "runs/c91-refined-v1/model.pt"
REFINED_C91_SHA256 = "86ead5c164c3a49a5fb91f8d02a8db01e7c684109e485021620b33b51a51387a"

EXPECTED_C91_CONFIG = {
    "dim": EMBED, "small": False, "stereo": True, "num_basis": BASIS,
    "segment_len": FEATURE_SAMPLES, "overlap_len": 512,
    "n_fft": FEATURE_SAMPLES, "sample_rate": SAMPLE_RATE,
    "num_sources": SOURCES, "torch_compile": False, "use_gru": True,
    "use_branch_rnns": False, "residual_source_softmax": True,
    "decoder_hann_baked": True, "rnn_klass": None,
    "spec_branch_use_phase": True, "norm_before_mask_estimate": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_frames(frames: Tensor, previous_tail: Tensor) -> tuple[Tensor, Tensor]:
    """Overlap real adjacent frames, without a denominator or source gain.

    ``frames`` is ``[..., frame_count, 512]`` and ``previous_tail`` is
    ``[..., 256]``.  The first output hop uses the supplied tail; subsequent
    hops use the preceding frame's right half.  Returned state retains its
    autograd connection.  A caller doing truncated training must detach it.
    """
    if (frames.ndim < 2 or frames.shape[-1] != SYNTHESIS_SAMPLES
            or frames.shape[-2] < 1
            or previous_tail.shape != (*frames.shape[:-2], HOP)
            or previous_tail.dtype != frames.dtype
            or previous_tail.device != frames.device):
        raise ValueError("OLA expects [...,F,512] frames and matching [...,256] tail")
    left, right = frames[..., :HOP], frames[..., HOP:]
    prior = torch.cat((previous_tail.unsqueeze(-2), right[..., :-1, :]), dim=-2)
    emitted = (left + prior).reshape(*frames.shape[:-2], frames.shape[-2] * HOP)
    return emitted, right[..., -1, :].clone()


class OLA512Synthesis(nn.Module):
    """Parameter-free local transform and explicit spectral/waveform overlap.

    Kept independent of the separator so a unity-transform oracle can exercise
    timing and normalization without loading any model checkpoint.
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("The independent transform supports float32 or float64")
        window = torch.hann_window(SYNTHESIS_SAMPLES, periodic=True, dtype=dtype)
        self.register_buffer("window", window)
        self.register_buffer("spectral_denominator", window[:HOP].square() + window[HOP:].square())
        self.register_buffer("waveform_window_sum", window[:HOP] + window[HOP:])

    def carrier(self, audio_with_past: Tensor) -> Tensor:
        """Hann512 rFFT frames from ``[...,256+T]``, T a positive hop multiple."""
        if (audio_with_past.ndim < 2 or audio_with_past.shape[-1] < SYNTHESIS_SAMPLES
                or (audio_with_past.shape[-1] - HOP) % HOP
                or audio_with_past.dtype != self.window.dtype
                or audio_with_past.device != self.window.device):
            raise ValueError("Carrier expects matching [...,256+T] real audio")
        frames = audio_with_past.unfold(-1, SYNTHESIS_SAMPLES, HOP)
        return torch.fft.rfft(frames * self.window, n=SYNTHESIS_SAMPLES, dim=-1)

    def spectral(self, spectrum: Tensor, previous_numerator_tail: Tensor) -> tuple[Tensor, Tensor]:
        """IFFT, synthesis Hann, overlap numerator, then squared-Hann division."""
        if (spectrum.ndim < 2 or not spectrum.is_complex() or spectrum.shape[-1] != SYNTHESIS_BINS
                or spectrum.real.dtype != self.window.dtype
                or spectrum.device != self.window.device):
            raise ValueError("Spectral synthesis expects matching complex [...,F,257]")
        frames = torch.fft.irfft(spectrum, n=SYNTHESIS_SAMPLES, dim=-1)
        frames = frames * self.window
        numerator, next_tail = overlap_frames(frames, previous_numerator_tail)
        frame_count = spectrum.shape[-2]
        denominator = self.spectral_denominator.repeat(frame_count)
        return numerator / denominator, next_tail

    def waveform(self, unwindowed_frames: Tensor, previous_windowed_tail: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the new Hann once, then overlap; no squared-window division."""
        if (unwindowed_frames.dtype != self.window.dtype
                or unwindowed_frames.device != self.window.device):
            raise ValueError("Waveform synthesis dtype/device must match the Hann buffer")
        return overlap_frames(unwindowed_frames * self.window, previous_windowed_tail)


class OLA512State(NamedTuple):
    audio_history: Tensor                 # [B,2,768], physical input samples
    fusion_hidden: Tensor                 # [2,B,1000], physical GRU state * 2**-18
    spectral_numerator_tail: Tensor        # [B,4,2,256], before denominator/gains
    waveform_tail: Tensor                  # [B,4,2,256], after Hann, before gains

    def detached(self) -> "OLA512State":
        return OLA512State(*(value.detach() for value in self))


@dataclass(frozen=True)
class OLA512Output:
    raw: Tensor                          # all four learned source heads, graph aligned
    deployed: Tensor                     # unchanged DBV, residual Other
    spectral: Tensor                     # normalized/overlapped, source gains applied
    waveform: Tensor                     # overlapped, source gains applied
    delayed_mixture: Tensor              # [B,2,T], exactly the emitted physical interval
    state: OLA512State


class OLA512Model(nn.Module):
    """Fixed C91-sized separator with long features and short real-frame OLA.

    The plain constructor provides the schema for strict state loading; use
    ``from_refined_c91`` for the chosen authenticated initializer.  No current
    C191 correction module is attached, and no existing model is modified.
    """

    sample_rate = SAMPLE_RATE
    hop_samples = HOP
    synthesis_samples = SYNTHESIS_SAMPLES
    feature_samples = FEATURE_SAMPLES
    graph_alignment_samples = HOP
    alignment_samples = HOP
    host_queue_samples = HOP
    algorithmic_latency_samples = 2 * HOP
    host_visible_pdc_samples = 2 * HOP
    # Output-relative lookahead is explicit; there is no read beyond this call.
    future_context_samples = HOP
    future_callbacks_beyond_received_input = 0
    public_fusion_state_scale = PUBLIC_FUSION_SCALE
    flush_required = True
    flush_hops = 1

    def __init__(self):
        super().__init__()
        # Keep the copied C91 modules recognizable, without importing a legacy
        # model that would construct an unused 1024-sample synthesis decoder.
        self.spec_encode = nn.Linear(CHANNELS * FEATURE_BINS * 2, EMBED)
        self.conv_encode = nn.Conv1d(CHANNELS, BASIS * 2, FEATURE_SAMPLES, stride=HOP)
        self.basis_to_embed = nn.Conv1d(BASIS, EMBED, 1)
        self.fusion_branch = nn.GRU(2 * EMBED, 2 * EMBED, num_layers=2, batch_first=True)
        self.spec_norm = nn.RMSNorm(EMBED)
        self.to_spec_masks = nn.Linear(EMBED, CHANNELS * SYNTHESIS_BINS * 2 * SOURCES)
        self.waveform_norm = nn.RMSNorm(EMBED)
        self.to_waveform_masks = nn.Linear(EMBED, SOURCES * BASIS)
        # This parameter is UNWINDOWED in the new model.  The seed happens to
        # contain old baked-window values; the new Hann is applied in waveform().
        self.waveform_decoder_weight = nn.Parameter(torch.zeros(BASIS, CHANNELS, SYNTHESIS_SAMPLES))
        self.register_buffer("analysis_window", torch.hann_window(FEATURE_SAMPLES, periodic=True))
        self.register_buffer("output_source_scales", torch.ones(SOURCES, dtype=torch.float32))
        self.synthesis = OLA512Synthesis()
        self.provenance: dict = {"version": VERSION, "initialization": "uninitialized_schema_only"}

    @property
    def architecture_metadata(self) -> dict:
        return {
            "version": VERSION, "source_order": list(SOURCE_ORDER),
            "sample_rate": SAMPLE_RATE, "feature_n_fft": FEATURE_SAMPLES,
            "synthesis_n_fft": SYNTHESIS_SAMPLES, "hop_samples": HOP,
            "feature_history_samples": FEATURE_HISTORY,
            "graph_alignment_samples": HOP, "host_queue_samples": HOP,
            "intended_total_latency_samples": 2 * HOP,
            "host_queue_implemented_in_this_module": False,
            "future_context_from_output_samples": HOP,
            "future_callbacks_beyond_received_input": 0,
            "flush_hops": 1, "public_fusion_state_scale": PUBLIC_FUSION_SCALE,
            "state_names": list(OLA512State._fields),
            "precision": "float32", "optional_2048_analysis": False,
        }

    def initial_state(self, batch_size: int, *, device=None) -> OLA512State:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        device = self.output_source_scales.device if device is None else torch.device(device)
        if device != self.output_source_scales.device or self.output_source_scales.dtype != torch.float32:
            raise ValueError("State must use the model's device and the prototype requires float32")
        shapes = ((batch_size, CHANNELS, FEATURE_HISTORY), (2, batch_size, 2 * EMBED),
                  (batch_size, SOURCES, CHANNELS, HOP), (batch_size, SOURCES, CHANNELS, HOP))
        return OLA512State(*(torch.zeros(shape, device=device, dtype=torch.float32) for shape in shapes))

    def _validate(self, audio: Tensor, state: OLA512State | None) -> OLA512State:
        if (audio.ndim != 3 or audio.shape[0] < 1 or audio.shape[1] != CHANNELS
                or audio.shape[-1] < HOP or audio.shape[-1] % HOP
                or audio.dtype != torch.float32
                or audio.device != self.output_source_scales.device
                or self.output_source_scales.dtype != torch.float32):
            raise ValueError("Audio must be float32 [B,2,T], T a positive multiple of 256, on the model device")
        if state is None:
            return self.initial_state(audio.shape[0])
        if not isinstance(state, OLA512State):
            raise ValueError("Use the explicit four-tensor OLA512State ABI")
        expected = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                    (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP))
        for value, shape in zip(state, expected, strict=True):
            if tuple(value.shape) != shape or value.dtype != torch.float32 or value.device != audio.device:
                raise ValueError("OLA state shape/dtype/device differs from the declared ABI")
        return state

    @staticmethod
    def _residual_source_softmax(logits: Tensor) -> Tensor:
        # C91 uses residual logits + 4*competitive, not a standalone softmax.
        return logits.add(torch.softmax(logits, dim=-1), alpha=float(SOURCES))

    def render(self, audio: Tensor, state: OLA512State | None = None) -> OLA512Output:
        """Grouped differentiable render; returned samples include graph pre-roll.

        For F input hops starting at t, emitted time is [t-256,t-256+F*256).
        State is never detached implicitly.  Autocast is disabled because this
        prototype's feature and explicit-state ABI is FP32 only.
        """
        state = self._validate(audio, state)
        with torch.autocast(audio.device.type, enabled=False):
            return self._render_fp32(audio, state)

    def _render_fp32(self, audio: Tensor, state: OLA512State) -> OLA512Output:
        batch, _, samples = audio.shape
        frame_count = samples // HOP
        joined = torch.cat((state.audio_history, audio), dim=-1)

        # Same feature window, FFT normalization and packing as C91's STFT and
        # Rearrange('(b s) f n c -> b n (s f c)').  Only frame stride changes.
        feature_frames = joined.unfold(-1, FEATURE_SAMPLES, HOP)
        feature_spec = torch.fft.rfft(feature_frames * self.analysis_window,
                                      n=FEATURE_SAMPLES, dim=-1)
        feature_ri = torch.view_as_real(feature_spec)
        packed = feature_ri.permute(0, 2, 1, 3, 4).reshape(batch, frame_count, CHANNELS * FEATURE_BINS * 2)
        spec = self.spec_encode(packed)

        to_relu, to_sigmoid = self.conv_encode(joined).chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        waveform = self.basis_to_embed(basis).transpose(1, 2)
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        recurrent, next_physical_hidden = self.fusion_branch(fusion_input, physical_hidden)
        # Preserve BOTH C91 residual additions and their order.
        fused = fusion_input + recurrent
        fused_spec, fused_waveform = fused.chunk(2, dim=-1)
        spec = fused_spec + spec
        waveform = fused_waveform + waveform

        spec_logits = self.to_spec_masks(self.spec_norm(spec))
        spec_logits = spec_logits.reshape(batch, frame_count, CHANNELS, SYNTHESIS_BINS, 2, SOURCES)
        spec_masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        # Drop the first 512 history samples: the remaining local carrier has
        # exactly the previous 256 observed samples plus this call's input.
        carrier = self.synthesis.carrier(joined[..., FEATURE_HISTORY - HOP:])
        masked_ri = torch.view_as_real(carrier).unsqueeze(-1) * spec_masks
        # [B,C,F,Q,RI,S] -> [B,S,C,F,Q,RI] before view_as_complex.
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
        # Match C91's sum-then-scale order for raw heads.  The diagnostic branch
        # values are scaled separately and may differ in their sum by roundoff.
        raw = (spectral + waveform_audio) * scales
        delayed_mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        retained = raw[:, :3]
        deployed = torch.cat((retained, delayed_mixture.unsqueeze(1) - retained.sum(dim=1, keepdim=True)), dim=1)
        next_state = OLA512State(
            joined[..., -FEATURE_HISTORY:].clone(),
            next_physical_hidden * PUBLIC_FUSION_SCALE,
            next_spectral_tail,
            next_waveform_tail,
        )
        return OLA512Output(raw, deployed, spectral * scales, waveform_audio * scales,
                            delayed_mixture, next_state)

    def forward(self, audio: Tensor, state: OLA512State | None = None, *, return_raw: bool = False):
        result = self.render(audio, state)
        return (result.raw if return_raw else result.deployed), result.state

    def forward_raw(self, audio: Tensor, state: OLA512State | None = None):
        return self.forward(audio, state, return_raw=True)

    def forward_chunk(self, audio: Tensor, state: OLA512State, *, return_raw: bool = False):
        if audio.ndim != 3 or audio.shape[-1] != HOP:
            raise ValueError("Literal callback input must contain exactly 256 samples")
        return self.forward(audio, state, return_raw=return_raw)

    def flush(self, state: OLA512State, *, return_raw: bool = False):
        """Feed ONE zero input hop and return the final graph-delayed real hop.

        This advances the state.  Further calls represent further zero audio;
        they are not an idempotent flush and are not needed for tail recovery.
        """
        if not isinstance(state, OLA512State):
            raise ValueError("Flush requires OLA512State")
        zeros = state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP))
        return self.forward_chunk(zeros, state, return_raw=return_raw)

    @classmethod
    def from_refined_c91(cls, checkpoint: Path = REFINED_C91) -> "OLA512Model":
        """Authenticate and copy the chosen C91 initializer onto CPU only.

        Hash verification precedes the trusted local config deserialization.
        No existing loader/trainer is imported; this does not create an export,
        checkpoint file, CUDA context, or training job.
        """
        checkpoint = Path(checkpoint).resolve()
        if file_sha256(checkpoint) != REFINED_C91_SHA256:
            raise ValueError("The initializer requires the authenticated refined C91 checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != {"model", "config"}:
            raise ValueError("Refined C91 deployment payload schema changed")
        config = pickle.loads(payload["config"]) if isinstance(payload["config"], bytes) else payload["config"]
        if config != EXPECTED_C91_CONFIG:
            raise ValueError("Refined C91 architecture differs from this fixed prototype")
        source = payload["model"]
        if not isinstance(source, dict):
            raise ValueError("Refined C91 model state is not a mapping")
        # Preserve the caller's CPU initialization RNG.  No CUDA RNG is queried.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()

        copies = {
            "analysis_window": "stft.window",
            "output_source_scales": "output_source_scales",
            "spec_encode.weight": "spec_encode.1.weight",
            "spec_encode.bias": "spec_encode.1.bias",
            "conv_encode.weight": "conv_encode.weight",
            "conv_encode.bias": "conv_encode.bias",
            "basis_to_embed.weight": "basis_to_embed.0.weight",
            "basis_to_embed.bias": "basis_to_embed.0.bias",
            "spec_norm.weight": "to_spec_masks.0.weight",
            "waveform_norm.weight": "to_waveform_masks.0.weight",
            "to_waveform_masks.weight": "to_waveform_masks.1.weight",
            "to_waveform_masks.bias": "to_waveform_masks.1.bias",
        }
        copies.update({f"fusion_branch.{name}": f"fusion_branch.{name}"
                       for name, _ in model.fusion_branch.named_parameters()})
        transformed = {"to_spec_masks.1.weight", "to_spec_masks.1.bias", "conv_decode.weight"}
        unused = {"stft.streaming_envelope", "conv_decode.window", "conv_decode.bias"}
        if set(source) != set(copies.values()) | transformed | unused:
            raise ValueError("Refined C91 source tensor inventory changed")
        if any(not isinstance(value, Tensor) or value.dtype != torch.float32
               or not bool(torch.isfinite(value).all()) for value in source.values()):
            raise ValueError("Refined C91 tensors must be finite float32")
        state = model.state_dict()
        with torch.no_grad():
            for destination, original in copies.items():
                if state[destination].shape != source[original].shape:
                    raise ValueError(f"Copied tensor shape differs: {original}")
                state[destination].copy_(source[original])
            weight = source["to_spec_masks.1.weight"]
            bias = source["to_spec_masks.1.bias"]
            if weight.shape != (CHANNELS * FEATURE_BINS * 2 * SOURCES, EMBED) or bias.shape != (weight.shape[0],):
                raise ValueError("Original spectral projection layout changed")
            selected_weight = weight.reshape(CHANNELS, FEATURE_BINS, 2, SOURCES, EMBED)[:, ::2]
            selected_bias = bias.reshape(CHANNELS, FEATURE_BINS, 2, SOURCES)[:, ::2]
            state["to_spec_masks.weight"].copy_(selected_weight.reshape_as(state["to_spec_masks.weight"]))
            state["to_spec_masks.bias"].copy_(selected_bias.reshape_as(state["to_spec_masks.bias"]))
            decoder = source["conv_decode.weight"]
            if decoder.shape != (BASIS, CHANNELS, FEATURE_SAMPLES):
                raise ValueError("Original baked waveform decoder shape changed")
            folded = decoder[..., :SYNTHESIS_SAMPLES] + decoder[..., SYNTHESIS_SAMPLES:]
            state["waveform_decoder_weight"].copy_(folded)
        model.load_state_dict(state, strict=True)
        if file_sha256(checkpoint) != REFINED_C91_SHA256:
            raise ValueError("Reference checkpoint changed while the initializer was constructed")
        model.provenance = {
            "version": VERSION, "initialization": "authenticated_c91_folded512_then_new_hann",
            "reference_path": str(checkpoint), "reference_sha256": REFINED_C91_SHA256,
            "copied_tensors": copies,
            "spectral_projection_rule": "new[s,q,ri,source] = old[s,2*q,ri,source]",
            "waveform_parameter_rule": "new[...,j] = old_baked[...,j] + old_baked[...,512+j]",
            "waveform_effective_rule": "new_parameter[...,j] * periodic_Hann512[j]",
            "unused_reference_tensors": sorted(unused),
            "unused_decoder_bias_reason": "C91's custom forward omits the stored ConvTranspose bias",
            "reference_model_modified": False, "equivalence_claimed": False,
        }
        return model.eval()


__all__ = [
    "OLA512Model", "OLA512Output", "OLA512State", "OLA512Synthesis",
    "overlap_frames", "VERSION", "HOP", "PUBLIC_FUSION_SCALE",
    "REFINED_C91", "REFINED_C91_SHA256",
]
