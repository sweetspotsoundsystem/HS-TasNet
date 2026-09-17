"""Trainable 512-sample-latency adapters for retained C126 and full C191."""

from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn


HISTORY_ROOT = Path("/home/axel/autoresearch/codex/HS-TasNet-latency11-v1-state")
ORIGINAL_CORE_SOURCE = HISTORY_ROOT / "workspaces/c130-branch-aware-tcn-v1/HS-TasNet/hs_tasnet/hs_tasnet.py"
ORIGINAL_CORE_SHA256 = "fc02c718121ef615c3924b6975aab3045bc862dec6965398d2f6c85872d45b30"
CORE_SOURCE = Path(__file__).with_name("causal_core.py")
C126_CHECKPOINT = HISTORY_ROOT / "runs/c126-causal-synthesis-production-horizon-v5/gates-v5/step-100000-v5/c126-step-100000-deploy-v5.pt"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def core_class():
    """Use the exact retained causal core vendored with these adapters."""
    from research.direct.causal_core import HSTasNet
    return HSTasNet


class C126Model(nn.Module):
    def __init__(self, core_config):
        super().__init__()
        self.core_config = dict(core_config)
        self.core = core_class()(**self.core_config)
        if not self.core.causal_current_chunk or self.core.overlap_len != 512:
            raise ValueError("Expected the retained current-chunk C126 core")
        if not self.core.conv_decode.hann_window_baked:
            raise ValueError("Keep the effective 512-tap decoder baked")

    def initial_state(self, batch_size, device=None):
        reference = next(self.parameters())
        device = reference.device if device is None else device
        return (
            torch.zeros(batch_size, 2, 512, device=device, dtype=reference.dtype),
            torch.zeros(2, batch_size, 1000, device=device, dtype=reference.dtype),
        )

    def forward(self, audio, state=None, *, return_raw=False):
        if audio.ndim != 3 or audio.shape[1] != 2 or audio.shape[-1] % 512 or not audio.shape[-1]:
            raise ValueError("Expected audio [batch, 2, positive multiple of 512]")
        if state is None:
            state = self.initial_state(audio.shape[0], device=audio.device)
        history, hidden = state
        raw, hiddens = self.core(
            torch.cat((history, audio), dim=-1),
            hiddens=(None, None, hidden, None, None),
            auto_causal_pad=False, auto_curtail_length_to_multiple=False,
            is_streaming=True,
        )
        raw = raw.float()
        output = raw if return_raw else torch.cat((
            raw[:, :3], audio.float().unsqueeze(1) - raw[:, :3].sum(dim=1, keepdim=True),
        ), dim=1)
        return output, (audio[..., -512:].clone(), hiddens[2])

    def forward_chunk(self, audio, state, *, return_raw=False):
        if audio.shape[-1] != 512:
            raise ValueError("A callback contains exactly 512 samples")
        return self(audio, state, return_raw=return_raw)


class Latency11Model(nn.Module):
    """Common streaming/crop API; outputs refer to the current input samples.

    ``forward(audio, state=None)`` processes a multiple of 512 samples.
    ``forward_chunk(audio, state)`` processes one callback. Both return
    ``(stems, next_state)``. Default outputs include residual Other;
    ``forward_raw`` retains the native core Other head for training.
    """

    overlap_len = 512
    segment_len = 1024
    sample_rate = 44_100
    audio_channels = 2
    num_sources = 4
    causal_current_chunk = True
    algorithmic_latency_samples = 512
    alignment_samples = 0

    def __init__(self, engine, kind):
        super().__init__()
        self.engine = engine
        self.kind = kind
        self.vocal_gain = None
        self._vocal_output_ratio = 1.0

    @property
    def core(self):
        return self.engine.core

    @property
    def output_source_scales(self):
        scales = self.core.output_source_scales
        if self._vocal_output_ratio == 1.0:
            return scales
        scales = scales.clone()
        scales[2] *= self._vocal_output_ratio
        return scales

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(self, batch_size, device=None):
        return self.engine.initial_state(batch_size, device=device)

    def forward(self, audio, state=None, *, return_raw=False):
        stems, next_state = self.engine(audio, state, return_raw=return_raw)
        return self._calibrate(audio, stems, return_raw=return_raw), next_state

    def forward_raw(self, audio, state=None):
        return self(audio, state, return_raw=True)

    def forward_chunk(self, audio, state, *, return_raw=False):
        stems, next_state = self.engine.forward_chunk(audio, state, return_raw=return_raw)
        return self._calibrate(audio, stems, return_raw=return_raw), next_state

    def _calibrate(self, audio, stems, *, return_raw):
        if self._vocal_output_ratio == 1.0:
            return stems
        stems = stems.float()
        dbv = torch.cat((stems[:, :2], stems[:, 2:3] * self._vocal_output_ratio), dim=1)
        other = stems[:, 3:4] if return_raw else audio.float()[:, None] - dbv.sum(dim=1, keepdim=True)
        return torch.cat((dbv, other), dim=1)

    @torch.no_grad()
    def set_vocal_gain(self, gain):
        """Calibrate the final vocal output without altering correction features."""
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("Vocal gain must be finite and positive")
        self.vocal_gain = float(gain)
        self._vocal_output_ratio = gain / (2.0 * float(self.core.output_source_scales[2]))
        if abs(self._vocal_output_ratio - 1.0) < 1e-7:
            self._vocal_output_ratio = 1.0

    def init_stateful_transform_fn(self, device=None, return_reduced_sources=None):
        self.eval()
        device = self.device if device is None else torch.device(device)
        state = self.initial_state(1, device=device)

        @torch.inference_mode()
        def transform(audio_chunk):
            nonlocal state
            is_numpy = isinstance(audio_chunk, np.ndarray)
            audio = torch.as_tensor(audio_chunk, device=device, dtype=torch.float32)
            output, state = self.forward_chunk(audio.unsqueeze(0), state)
            output = output[0]
            if return_reduced_sources is not None:
                output = output[return_reduced_sources].sum(dim=0)
            return output.cpu().numpy() if is_numpy else output

        return transform


def _validate_long_analysis_metadata(value):
    from research.direct.latency_long_analysis import LONG_ANALYSIS_METADATA, LONG_ANALYSIS_VERSION
    expected = {**LONG_ANALYSIS_METADATA, "version": LONG_ANALYSIS_VERSION}
    if (type(value) is not dict or value.keys() != expected.keys()
            or any(type(value[key]) is not type(item) or value[key] != item
                   for key, item in expected.items())):
        raise ValueError("Unsupported or inconsistent C191 long-analysis metadata")
    return expected


def long_analysis_metadata(model):
    """Return the opt-in architecture descriptor; native models retain no key."""
    if not hasattr(model.engine, "long_analysis"):
        return None
    from research.direct.latency_long_analysis import C191LongAnalysisModel, LONG_ANALYSIS_VERSION
    if model.kind != "c191" or not isinstance(model.engine, C191LongAnalysisModel):
        raise ValueError("Long-analysis state requires the explicit C191 variant")
    return _validate_long_analysis_metadata({
        **model.engine.long_analysis_metadata, "version": LONG_ANALYSIS_VERSION,
    })


def load_model(kind="c126", checkpoint=None, *, device="cpu", vocal_gain=None):
    """Load a retained model or a direct snapshot, retaining native decoder/gains."""
    if kind not in ("c126", "c191"):
        raise ValueError("Model kind must be c126 or c191")
    provenance = {}
    if checkpoint is not None or kind == "c126":
        checkpoint_path = Path(checkpoint or C126_CHECKPOINT).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if vocal_gain is None:
            vocal_gain = payload.get("vocal_gain")
        provenance.update(payload.get("provenance", {}))
        saved_kind = payload.get("latency11_kind", "c126")
        if saved_kind != kind:
            raise ValueError(f"Checkpoint kind {saved_kind} differs from requested {kind}")
        head_phase_features = False
        if "c191_head_phase_features" in payload:
            from research.direct.latency11_c191 import HEAD_PHASE_FEATURES_VERSION
            if kind != "c191" or payload["c191_head_phase_features"] != HEAD_PHASE_FEATURES_VERSION:
                raise ValueError("Unsupported explicit C191 head phase-feature metadata")
            head_phase_features = True
        analysis_metadata = None
        if "c191_long_analysis" in payload:
            if kind != "c191" or head_phase_features:
                raise ValueError("Long analysis requires native-head C191")
            analysis_metadata = _validate_long_analysis_metadata(payload["c191_long_analysis"])
        if "c191_long_analysis" in provenance and provenance["c191_long_analysis"] != analysis_metadata:
            raise ValueError("C191 long-analysis provenance differs from its architecture metadata")
        if "run_config" in payload and payload["run_config"].get("c191_long_analysis") != analysis_metadata:
            raise ValueError("C191 long-analysis resume configuration differs from its architecture metadata")
        config = payload["config"]
        config = pickle.loads(config) if isinstance(config, bytes) else config
        if kind == "c126":
            engine = C126Model(config)
            (engine if "latency11_kind" in payload else engine.core).load_state_dict(payload["model"], strict=True)
        else:
            if analysis_metadata is None:
                from research.direct.latency11_c191 import C191Model
                engine = C191Model(core_config=config, head_phase_features=head_phase_features)
            else:
                from research.direct.latency_long_analysis import C191LongAnalysisModel
                engine = C191LongAnalysisModel(core_config=config, head_phase_features=False)
            engine.load_state_dict(payload["model"], strict=True)
            if analysis_metadata is not None and not torch.equal(
                engine.long_analysis.window,
                torch.hann_window(2048, periodic=True, dtype=torch.float32),
            ):
                raise ValueError("C191 long-analysis window differs from its fixed Hann definition")
    else:
        from research.direct.latency11_c191 import C191Model, FULL_CHECKPOINT, FULL_RUNTIME_SHA256
        engine = C191Model.from_historical()
        checkpoint_path = FULL_CHECKPOINT
        provenance["original_runtime_state_sha256"] = FULL_RUNTIME_SHA256
    model = Latency11Model(engine, kind).to(device).eval()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    provenance.setdefault("original_payload_path", str(checkpoint_path))
    provenance.setdefault("original_payload_sha256", checkpoint_sha256)
    provenance.update({
        "kind": kind,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "n_params": model.num_parameters,
        "core_source": str(CORE_SOURCE),
        "core_source_sha256": file_sha256(CORE_SOURCE),
        "original_core_source": str(ORIGINAL_CORE_SOURCE),
        "original_core_source_sha256": ORIGINAL_CORE_SHA256,
        "algorithmic_latency_samples": model.algorithmic_latency_samples,
        "alignment_samples": model.alignment_samples,
    })
    if kind == "c191" and model.engine.head.phase_features:
        from research.direct.latency11_c191 import HEAD_PHASE_FEATURES_VERSION
        provenance["c191_head_phase_features"] = HEAD_PHASE_FEATURES_VERSION
    model.provenance = provenance
    analysis_metadata = long_analysis_metadata(model)
    if analysis_metadata is not None:
        analysis_source = Path(__file__).with_name("latency_long_analysis.py")
        model.provenance.update({
            "c191_long_analysis": analysis_metadata,
            "long_analysis_source": str(analysis_source),
            "long_analysis_source_sha256": file_sha256(analysis_source),
        })
    if vocal_gain is not None:
        model.set_vocal_gain(vocal_gain)
    return model


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "latency11_kind": model.kind,
        "config": model.engine.core_config,
        "provenance": getattr(model, "provenance", {}),
        "vocal_gain": model.vocal_gain,
        "model": {key: value.detach().cpu() for key, value in model.engine.state_dict().items()},
    }
    if model.kind == "c191" and model.engine.head.phase_features:
        from research.direct.latency11_c191 import HEAD_PHASE_FEATURES_VERSION
        payload["c191_head_phase_features"] = HEAD_PHASE_FEATURES_VERSION
        payload["provenance"] = dict(payload["provenance"],
                                     c191_head_phase_features=HEAD_PHASE_FEATURES_VERSION,
                                     n_params=model.num_parameters)
    analysis_metadata = long_analysis_metadata(model)
    if analysis_metadata is not None:
        payload["c191_long_analysis"] = analysis_metadata
        payload["provenance"] = dict(payload["provenance"],
                                     c191_long_analysis=analysis_metadata,
                                     n_params=model.num_parameters)
    torch.save(payload, temporary)
    temporary.replace(path)
