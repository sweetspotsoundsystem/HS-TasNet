"""Trainable, vectorized C191 full correction with current-chunk alignment.

The historical constructor is used only to import the original weights. Saved
state dictionaries load without invoking the historical checkpoint chain.
The core's C130 source supplies the existing multi-frame current-chunk path;
the small correction modules below have no frozen-parent or controller code.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


STATE_ROOT = Path("/home/axel/autoresearch/codex/HS-TasNet-latency11-v1-state")
HISTORICAL_ROOT = STATE_ROOT / "workspaces/c191-c188-continuous-output-tcn-activation-preparation-v1/HS-TasNet"
CORE_SOURCE = Path(__file__).with_name("causal_core.py")
FULL_CHECKPOINT = STATE_ROOT / "runs/c191-c188-continuous-output-tcn-v1/checkpoints/payloads/step-000128-a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b.pt"
FULL_RUNTIME_SHA256 = "6315b6670f0313ad554e8e3088ced42be6c0cbc285bccb8f5f553c4d8fa367f2"
HOP = 512
FUSION_SCALE = 2.0**-18
HEAD_PHASE_FEATURES_VERSION = "phase12-v1"
TAP_OFFSETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048)
DEFAULT_CORE_CONFIG = dict(
    dim=500, small=False, stereo=True, num_basis=1500, segment_len=1024,
    overlap_len=512, n_fft=1024, sample_rate=44100, num_sources=4,
    torch_compile=False, use_gru=True, use_branch_rnns=False,
    residual_source_softmax=True, decoder_hann_baked=True, rnn_klass=None,
    spec_branch_use_phase=True, norm_before_mask_estimate=True,
    causal_current_chunk=True,
)


def _core_class():
    from research.direct.causal_core import HSTasNet
    return HSTasNet


def load_historical_runtime():
    """Read the exact full-strength U128 native runtime on CPU, without a run."""
    source = str(HISTORICAL_ROOT)
    sys.path.insert(0, source)
    try:
        import c191_runtime

        runtime, _ = c191_runtime.build_exact_runtime(device="cpu")
        payload = torch.load(FULL_CHECKPOINT, map_location="cpu", weights_only=True)
        c191_runtime.restore_head_state(runtime, payload["head_state"])
        runtime.eval()
        actual = c191_runtime.tensor_state_sha256(runtime.state_dict())
        if actual != FULL_RUNTIME_SHA256:
            raise ValueError(f"Historical C191 runtime identity changed: {actual}")
        return runtime
    finally:
        sys.path.remove(source)


class _ExactAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, correction):
        return torch.where(correction == 0, base, base + correction)

    @staticmethod
    def backward(ctx, gradient):
        # The old C184 helper deliberately blocked gradients into its parent.
        return gradient, gradient


class _C140Block(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.delay = 2 * dilation
        self.depthwise = nn.Conv1d(32, 32, 3, dilation=dilation, groups=32)
        self.pointwise = nn.Conv1d(32, 32, 1)

    def forward(self, x):
        return x[..., self.delay:] + self.pointwise(F.silu(self.depthwise(x)))


class _C140(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(20, 32, 1)
        self.blocks = nn.ModuleList(_C140Block(d) for d in (1, 1, 2, 4, 8, 16, 32))
        self.output_projection = nn.Conv1d(32, 6, 1)


class _Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(96, 16, 1)
        self.output_projection = nn.Conv1d(16, 4, 1)


class _Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(56, 16, 1, groups=2, bias=False)
        self.output_projection = nn.Conv1d(16, 4, 1, groups=2, bias=False)
        self.extension_output_projection = nn.Conv1d(16, 4, 1, groups=2, bias=False)
        self.register_buffer("correction_taper", torch.zeros(1, 1, 32))
        self.register_buffer("extension_taper", torch.zeros(1, 4, 128))


class _Correction(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(56, 16, 1, groups=2, bias=False)
        self.edge_drums_projection = nn.Conv1d(8, 4, 1, bias=False)
        self.edge_bass_projection = nn.Conv1d(8, 4, 1, bias=False)
        self.bulk_drums_projection = nn.Conv1d(8, 2, 1, bias=False)
        self.bulk_bass_projection = nn.Conv1d(8, 2, 1, bias=False)
        self.register_buffer("edge_basis", torch.zeros(1, 2, 1, 64))
        self.register_buffer("bulk_window", torch.zeros(1, 1, 256))


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.bass_residual_weight = nn.Parameter(torch.zeros(2, 16, 1))
        self.register_buffer("envelope", torch.zeros(1, 1, 512))


class _HeadBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(16, 16, 2, dilation=dilation, groups=16, bias=False)
        self.source_mixing = nn.Conv1d(16, 16, 1, groups=2, bias=False)

    def forward(self, x):
        return x + F.silu(self.source_mixing(self.depthwise(F.pad(x, (self.dilation, 0)))))


class _Head(nn.Module):
    def __init__(self, *, phase_features=False):
        super().__init__()
        self.input_projection = nn.Conv1d(4, 16, 1, groups=2, bias=False)
        self.blocks = nn.ModuleList(_HeadBlock(2**i) for i in range(11))
        self.output_projection = nn.Conv1d(16, 4, 1, groups=2, bias=False)
        self.phase_features = False
        if phase_features:
            self.enable_phase_features()

    def enable_phase_features(self):
        """Add zero-initialized audio/phase columns before creating an optimizer.

        Each source keeps its original L/R columns, followed by L*cos, R*cos,
        L*sin and R*sin. Expanding the existing parameter uses no random draws.
        """
        if self.phase_features:
            return self
        projection = self.input_projection
        weight = projection.weight
        if projection.in_channels != 4 or tuple(weight.shape) != (16, 2, 1):
            raise ValueError("Phase expansion requires the native four-channel head")
        expanded = weight.detach().new_zeros((16, 6, 1))
        expanded[:, :2].copy_(weight.detach())
        projection.in_channels = 12
        projection.weight = nn.Parameter(expanded, requires_grad=weight.requires_grad)
        self.phase_features = True
        return self

    def forward(self, dense_db, samples):
        if self.phase_features:
            batch, _, length = dense_db.shape
            positions = (torch.arange(length, device=dense_db.device) - 2048).remainder(HOP)
            phase = positions.to(torch.float32) * (2.0 * torch.pi / HOP)
            source_audio = dense_db.reshape(batch, 2, 2, length)
            inputs = torch.cat((source_audio, source_audio * phase.cos(),
                                source_audio * phase.sin()), dim=2).reshape(batch, 12, length)
        else:
            inputs = dense_db
        hidden = F.silu(self.input_projection(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden[..., -samples:])


class C191Model(nn.Module):
    """Full 28.6M-parameter C191; all learned modules are differentiable.

    Output has zero graph delay. The host's asynchronous queue supplies the
    512-sample (11.61 ms) scheduling delay. State is compatible with native
    C191, including its public GRU scale of 2**-18.

    ``return_raw=True`` retains final corrected D/B/V and the core's original
    fourth head, before residual Other replacement. That fourth head was not
    emitted by historical C191; it is exposed for full-core supervision.
    """

    sample_rate = 44100
    hop_samples = HOP
    host_visible_pdc_samples = HOP
    algorithmic_latency_samples = HOP
    alignment_samples = 0
    future_context_samples = 0
    flush_required = False

    def __init__(self, core_config=None, *, corrections_fp32=False, head_phase_features=False):
        super().__init__()
        # Training precision only: no new learned state or deployment behavior.
        self.corrections_fp32 = bool(corrections_fp32)
        self.core_config = dict(DEFAULT_CORE_CONFIG if core_config is None else core_config)
        self.core = _core_class()(**self.core_config)
        self.c140 = _C140()
        self.temporal = _Temporal()
        self.refiner = _Refiner()
        self.correction = _Correction()
        self.router = _Router()
        self.head = _Head(phase_features=head_phase_features)

    @classmethod
    def from_historical(cls, runtime=None):
        """Materialize all original weights, then use ordinary state_dict I/O."""
        runtime = load_historical_runtime() if runtime is None else runtime
        model = cls(pickle.loads(runtime.parent.c157.parent._config))
        mapping = {
            "parent.c157.parent.": "core.",
            "parent.c157.view.frozen_c140.": "c140.",
            "parent.c157.view.temporal_head.": "temporal.",
            "parent.refiner.": "refiner.",
            "parent.correction.": "correction.",
            "parent.router.": "router.",
            "head.": "head.",
        }
        state = {}
        for name, value in runtime.state_dict().items():
            for old, new in mapping.items():
                if name.startswith(old):
                    state[new + name[len(old):]] = value
                    break
            else:
                raise ValueError(f"Unknown historical C191 state: {name}")
        model.load_state_dict(state, strict=True)
        return model.eval()

    def initial_state(self, batch_size, device=None, dtype=torch.float32):
        device = self.core.output_source_scales.device if device is None else device
        shapes = ((batch_size, 2, 512), (2, batch_size, 1000),
                  (batch_size, 20, 128), (batch_size, 32, 512),
                  (batch_size, 1), (batch_size, 4, 2048), (batch_size, 4, 2048))
        return tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in shapes)

    @staticmethod
    def _frames(x):
        batch, channels, samples = x.shape
        return x.reshape(batch, channels, samples // HOP, HOP).permute(0, 2, 1, 3).reshape(-1, channels, HOP)

    @staticmethod
    def _sequence(x, batch):
        frames, channels, _ = x.shape
        return x.reshape(batch, frames // batch, channels, HOP).permute(0, 2, 1, 3).reshape(batch, channels, -1)

    @staticmethod
    def _sparse_taps(raw, history, support):
        batch, _, samples = raw.shape
        joined = torch.cat((history, raw), dim=-1).reshape(batch, 2, 2, -1)
        starts = torch.arange(samples // HOP, device=raw.device) * HOP + 2048
        offsets = torch.tensor(TAP_OFFSETS, device=raw.device)
        positions = starts[:, None, None] - offsets[None, :, None] + torch.arange(support, device=raw.device)
        # [B, source, stereo, frame, tap, support] -> source/tap/stereo channels.
        taps = joined[..., positions].permute(0, 3, 1, 4, 2, 5)
        return taps.reshape(-1, 56, support)

    @staticmethod
    def _hidden_tiles(projection, packed):
        # The native artifact uses 32-sample tiles for both sparse tap heads.
        return torch.cat(tuple(F.silu(projection(part)) for part in packed.split(32, dim=-1)), dim=-1)

    def forward_chunk(self, audio, state=None, *, return_raw=False):
        if audio.shape[-1] != HOP:
            raise ValueError("C191 forward_chunk requires exactly 512 samples")
        return self.forward(audio, state, return_raw=return_raw)

    def forward_raw(self, audio, state=None):
        return self.forward(audio, state, return_raw=True)

    def forward(self, audio, state=None, *, return_raw=False):
        if audio.ndim != 3 or audio.shape[1] != 2 or audio.shape[-1] < HOP or audio.shape[-1] % HOP:
            raise ValueError("C191 audio must have shape [B,2,T], T a positive multiple of 512")
        batch = audio.shape[0]
        if state is None:
            state = self.initial_state(batch, device=audio.device)
        past, fusion = state[:2]
        raw, hiddens, components = self.core(
            torch.cat((past, audio), dim=-1),
            hiddens=(None, None, fusion / FUSION_SCALE, None, None),
            auto_causal_pad=False, auto_curtail_length_to_multiple=False,
            is_streaming=True, return_streaming_components=True,
        )
        if self.corrections_fp32:
            with torch.autocast(audio.device.type, enabled=False):
                correction_state = (*state[:2], *(value.float() for value in state[2:]))
                return self._forward_corrections(
                    audio.float(), correction_state, raw.float(), hiddens,
                    tuple(value.float() for value in components), return_raw=return_raw,
                )
        return self._forward_corrections(audio, state, raw, hiddens, components, return_raw=return_raw)

    def _forward_corrections(self, audio, state, raw, hiddens, components, *, return_raw):
        """Apply the retained correction stack in the current precision context."""
        batch, _, samples = audio.shape
        frames = samples // HOP
        _, _, feature_history, hidden_history, valid, raw_history, emitted_history = state
        features = torch.cat((audio, *(x[:, :3].reshape(batch, 6, samples) for x in (raw, *components))), dim=1)
        joined_features = torch.cat((feature_history, features), dim=-1)
        hidden = self.c140.input_projection(joined_features)
        for block in self.c140.blocks:
            hidden = block(hidden)
        c140_delta = self.c140.output_projection(hidden).reshape(batch, 3, 2, samples)
        joined_hidden = torch.cat((hidden_history, hidden), dim=-1)
        temporal_taps = torch.cat((joined_hidden[..., 512:], joined_hidden[..., 511:-1], joined_hidden[..., :samples]), dim=1)
        projected = F.silu(self.temporal.input_projection(temporal_taps))
        temporal_delta = self.temporal.output_projection(projected)
        combined_delta = torch.cat((temporal_delta, torch.zeros_like(temporal_delta[:, :2])), dim=1).reshape(batch, 3, 2, samples)
        dbv = raw[:, :3] + c140_delta
        dbv = dbv + combined_delta
        frame_valid = torch.cat((valid[:, None, :], torch.ones(batch, frames - 1, 1, device=audio.device, dtype=valid.dtype)), dim=1).reshape(batch * frames, 1, 1)
        sample_valid = frame_valid.reshape(batch, frames, 1).repeat_interleave(HOP, dim=1).transpose(1, 2)
        dbv = torch.where((sample_valid > 0.5)[:, None], dbv, raw[:, :3])
        raw_db = dbv[:, :2].reshape(batch, 4, samples)
        raw_frames = self._frames(raw_db)

        # C166: fixed support of 31 incumbent + 127 extension samples per hop.
        refiner_hidden = self._hidden_tiles(self.refiner.input_projection, self._sparse_taps(raw_db, raw_history, 128))
        incumbent = self.refiner.output_projection(refiner_hidden[..., :32]) * self.refiner.correction_taper
        extension = self.refiner.extension_output_projection(refiner_hidden) * self.refiner.extension_taper
        prefix = torch.cat((raw_frames[..., :31] + incumbent[..., :31] + extension[..., :31],
                            raw_frames[..., 31:127] + extension[..., 31:127]), dim=-1)
        prefix = torch.where(frame_valid == 1.0, prefix, raw_frames[..., :127])
        db_frames = torch.cat((prefix, raw_frames[..., 127:]), dim=-1)

        # C168: phase-local edge and bulk corrections use the same *raw* C157
        # history as C166, not the corrected outputs or their own history.
        correction_hidden = self._hidden_tiles(self.correction.input_projection, self._sparse_taps(raw_db, raw_history, 256))
        corrections = []
        for source, source_hidden in zip(("drums", "bass"), correction_hidden.split(8, dim=1)):
            coefficients = getattr(self.correction, f"edge_{source}_projection")(source_hidden[..., :1]).reshape(batch * frames, 2, 2, 1)
            edge = coefficients[:, 0] * self.correction.edge_basis[:, 0] + coefficients[:, 1] * self.correction.edge_basis[:, 1]
            bulk = getattr(self.correction, f"bulk_{source}_projection")(source_hidden) * self.correction.bulk_window
            corrections.append(F.pad(edge, (0, HOP - 64)) + F.pad(bulk, (0, HOP - 256)))
        correction = torch.cat(corrections, dim=1) * (frame_valid == 1.0)
        db_frames = _ExactAdd.apply(db_frames, correction)

        # C184: the Bass router shares C157 temporal features.
        bass_residual = F.conv1d(self._frames(projected), self.router.bass_residual_weight)
        bass_residual = bass_residual * self.router.envelope * (frame_valid > 0.5)
        db_frames = torch.cat((db_frames[:, :2], _ExactAdd.apply(db_frames[:, 2:], bass_residual)), dim=1)
        parent_db = self._sequence(db_frames, batch)

        # C191: continuous causal TCN over the uncorrected C188 D/B history.
        dense_db = torch.cat((emitted_history, parent_db), dim=-1)
        head_delta = self.head(dense_db, samples)
        final_db = _ExactAdd.apply(parent_db, head_delta).reshape(batch, 2, 2, samples)
        final_dbv = torch.cat((final_db, dbv[:, 2:3]), dim=1)
        other = raw[:, 3:4] if return_raw else audio[:, None] - final_dbv.sum(dim=1, keepdim=True)
        stems = torch.cat((final_dbv, other), dim=1)
        next_state = (
            audio[..., -HOP:].clone(), hiddens[2] * FUSION_SCALE,
            joined_features[..., -128:].clone(), joined_hidden[..., -512:].clone(),
            torch.ones_like(valid), torch.cat((raw_history, raw_db), dim=-1)[..., -2048:].clone(),
            dense_db[..., -2048:].clone(),
        )
        return stems, next_state
