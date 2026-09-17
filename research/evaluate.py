"""Deterministic stateful-streaming evaluator for HS-TasNet checkpoints.

This is intentionally separate from training.  It loads one checkpoint once,
streams every selected track from beginning to end, resets state only at stream
boundaries, and scores only predeclared excerpts with one fixed alignment.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# cuBLAS reads this at CUDA initialization. Refuse an externally conflicting
# value rather than silently changing the evaluator backend.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG) != _CUBLAS_WORKSPACE_CONFIG:
    raise RuntimeError(
        f"CUBLAS_WORKSPACE_CONFIG must be {_CUBLAS_WORKSPACE_CONFIG!r}"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hs-tasnet-matplotlib")

import numpy as np
import soundfile as sf
import torch

# Running this file directly makes ``research/`` sys.path[0].  Put the checkout
# first so evaluation can never silently import a separately installed wheel.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from research.metrics import (
        SOURCE_ORDER,
        MetricConfig,
        band_sdr,
        fft_bandpass,
        mean_or_none,
        mixture_consistency,
        projection_metrics,
        rms_dbfs,
        single_source_probe_metrics,
        windowed_sdr,
    )
except ModuleNotFoundError:  # ``python research/evaluate.py``
    from metrics import (  # type: ignore[no-redef]
        SOURCE_ORDER,
        MetricConfig,
        band_sdr,
        fft_bandpass,
        mean_or_none,
        mixture_consistency,
        projection_metrics,
        rms_dbfs,
        single_source_probe_metrics,
        windowed_sdr,
    )

from research.evaluation_schema import (
    SOURCE_ORDER as DECISION_SOURCE_ORDER,
    validate_decision_inputs,
)

if tuple(SOURCE_ORDER) != DECISION_SOURCE_ORDER:
    raise RuntimeError("metric and decision-input source orders disagree")


def _sha256_file(path: Path, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model: Any) -> str:
    """Hash ordered tensor names, metadata, and bytes from a loaded state dict."""

    digest = hashlib.sha256()

    def add(blob: bytes) -> None:
        digest.update(len(blob).to_bytes(8, byteorder="big", signed=False))
        digest.update(blob)

    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"non-tensor model state is unsupported: {name}")
        cpu_value = value.detach().to(device="cpu").contiguous()
        add(name.encode("utf-8"))
        add(str(cpu_value.dtype).encode("ascii"))
        add(json.dumps(list(cpu_value.shape), separators=(",", ":")).encode("ascii"))
        add(cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, expected_file_hash: str | None) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    file_hash = _sha256_file(path)
    if expected_file_hash is not None and file_hash.lower() != expected_file_hash.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_file_hash}, got {file_hash}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value, file_hash


def _validate_manifest(
    manifest: dict[str, Any], *, allow_sealed_test: bool = False
) -> None:
    recorded_hash = manifest.get("content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("content_sha256", None)
    actual_hash = _canonical_hash(unhashed)
    if recorded_hash != actual_hash:
        raise RuntimeError(
            f"manifest content hash mismatch: recorded={recorded_hash}, actual={actual_hash}"
        )
    required = {
        "schema_version": 1,
        "dataset": "MUSDB18-HQ",
        "musdb_is_wav": True,
        "sample_rate": 44_100,
        "channels": 2,
        "source_order": list(SOURCE_ORDER),
    }
    if allow_sealed_test:
        required.update({
            "split": "test",
            "disk_subset": "test",
            "track_count": 46,
            "expected_local_track_count": 46,
            "official_track_count": 50,
            "missing_from_official_count": 4,
        })
    else:
        required.update({
            "split": "valid",
            "disk_subset": "train",
            "track_count": 14,
            "expected_local_track_count": 14,
            "official_track_count": 14,
            "missing_from_official_count": 0,
        })
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"manifest {key!r} must be {expected!r}, got {manifest.get(key)!r}"
            )
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("manifest tracks must be a non-empty list")
    if manifest.get("track_count") != len(tracks):
        raise ValueError("manifest track_count does not match tracks")
    names = [track.get("name") for track in tracks]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("manifest tracks must have unique sorted names")
    for track in tracks:
        if list(track.get("stems", {}).keys()) != list(SOURCE_ORDER):
            raise ValueError(f"bad stem order for track {track.get('name')!r}")
        if track.get("sample_rate") != 44_100 or track.get("channels") != 2:
            raise ValueError(f"bad audio metadata for track {track.get('name')!r}")
        mixture_check = track.get("mixture_sum", {})
        if mixture_check.get("checked") is not True:
            raise ValueError(
                f"unverified mixture/stem sum for track {track.get('name')!r}"
            )
        if (
            mixture_check.get("max_abs_error", math.inf)
            > mixture_check.get("max_abs_tolerance", -math.inf)
            or mixture_check.get("rms_error", math.inf)
            > mixture_check.get("rms_tolerance", -math.inf)
        ):
            raise ValueError(
                f"failed mixture/stem sum for track {track.get('name')!r}"
            )


def _safe_dataset_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid manifest path {relative!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"manifest path escapes dataset root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_config(config: dict[str, Any], manifest: dict[str, Any]) -> MetricConfig:
    if "checkpoint" in config or "checkpoints" in config:
        raise ValueError("checkpoint belongs only on the evaluator command line")
    expected = {
        "schema_version": 1,
        "sample_rate": 44_100,
        "source_order": list(SOURCE_ORDER),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config {key!r} must be {value!r}")
    if config.get("deterministic_algorithms") is not True:
        raise ValueError("config deterministic_algorithms must be true")
    alignment = config.get("alignment_samples")
    if not isinstance(alignment, int) or isinstance(alignment, bool):
        raise ValueError("alignment_samples must be an integer")
    if abs(alignment) > 1024:
        raise ValueError("alignment magnitude may not exceed the latency ceiling")
    excerpts = config.get("excerpts")
    defaults = config.get("default_excerpts")
    if not isinstance(excerpts, dict) and not isinstance(defaults, list):
        raise ValueError("config requires excerpts by track or default_excerpts")
    if isinstance(excerpts, dict):
        manifest_names = {track["name"] for track in manifest["tracks"]}
        extras = sorted(set(excerpts) - manifest_names)
        if extras:
            raise ValueError(f"config excerpts contain unknown tracks: {extras}")
    _configured_probe_tracks(config, manifest)
    metric_config = MetricConfig.from_mapping(config.get("metrics"))
    if metric_config.sample_rate != 44_100:
        raise ValueError("metric sample_rate must be 44100")
    return metric_config


def _configured_probe_tracks(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the frozen sorted validation subset used for isolated-source probes.

    ``single_source_probes`` remains accepted for compatibility: true means all
    validation tracks and false means none.  The explicit track list takes
    precedence when present and is the preferred frozen contract form.
    """

    manifest_names = tuple(track["name"] for track in manifest["tracks"])
    available = set(manifest_names)
    legacy = config.get("single_source_probes", True)
    if not isinstance(legacy, bool):
        raise ValueError("single_source_probes must be boolean when present")

    if "single_source_probe_tracks" not in config:
        return manifest_names if legacy else ()

    configured = config["single_source_probe_tracks"]
    if not isinstance(configured, list) or any(
        not isinstance(name, str) or not name for name in configured
    ):
        raise ValueError("single_source_probe_tracks must be a list of track names")
    if configured != sorted(configured) or len(configured) != len(set(configured)):
        raise ValueError("single_source_probe_tracks must be sorted and unique")
    unknown = sorted(set(configured) - available)
    if unknown:
        raise ValueError(f"unknown single-source probe tracks: {unknown}")
    if not legacy and configured:
        raise ValueError(
            "single_source_probes=false conflicts with a non-empty probe track list"
        )
    return tuple(configured)


def _sample_value(spec: Mapping[str, Any], samples_key: str, seconds_key: str,
                  sample_rate: int) -> int:
    has_samples = samples_key in spec
    has_seconds = seconds_key in spec
    if has_samples == has_seconds:
        raise ValueError(f"specify exactly one of {samples_key} and {seconds_key}")
    if has_samples:
        value = spec[samples_key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{samples_key} must be an integer")
        return value
    value = spec[seconds_key]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{seconds_key} must be finite")
    return int(round(float(value) * sample_rate))


def _reference_intervals(
    track: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, int | str]]:
    track_specs = config.get("excerpts", {}).get(track["name"])
    specs = track_specs if track_specs is not None else config.get("default_excerpts")
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"no fixed excerpts configured for {track['name']!r}")
    sample_rate = int(config["sample_rate"])
    alignment = int(config["alignment_samples"])
    frames = int(track["frames"])
    intervals: list[dict[str, int | str]] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"invalid excerpt for {track['name']!r}")
        if spec.get("all") is True:
            if len(spec) != 1:
                raise ValueError("an all-track excerpt cannot have other fields")
            ref_start = max(0, -alignment)
            ref_end = min(frames, frames - alignment)
        else:
            ref_start = _sample_value(spec, "start_sample", "start_seconds", sample_rate)
            length = _sample_value(spec, "num_samples", "duration_seconds", sample_rate)
            ref_end = ref_start + length
        estimate_start = ref_start + alignment
        estimate_end = ref_end + alignment
        if ref_start < 0 or ref_end <= ref_start or ref_end > frames:
            raise ValueError(
                f"reference excerpt {index} is outside {track['name']!r}"
            )
        if estimate_start < 0 or estimate_end > frames:
            raise ValueError(
                f"aligned estimate excerpt {index} is outside {track['name']!r}"
            )
        intervals.append({
            "id": str(spec.get("id", index)),
            "reference_start": ref_start,
            "reference_end": ref_end,
            "estimate_start": estimate_start,
            "estimate_end": estimate_end,
        })
    sorted_intervals = sorted(intervals, key=lambda item: int(item["reference_start"]))
    for previous, current in zip(sorted_intervals, sorted_intervals[1:]):
        if int(current["reference_start"]) < int(previous["reference_end"]):
            raise ValueError(f"overlapping excerpts for {track['name']!r}")
    return intervals


class _Capture:
    def __init__(self, intervals: Sequence[tuple[int, int]], output_shape: tuple[int, ...]):
        self.intervals = list(intervals)
        self.audio = [
            np.empty((*output_shape, end - start), dtype=np.float32)
            for start, end in intervals
        ]
        self.written = [np.zeros(end - start, dtype=np.bool_) for start, end in intervals]

    def add(self, chunk_start: int, chunk: np.ndarray) -> None:
        chunk_end = chunk_start + chunk.shape[-1]
        for index, (start, end) in enumerate(self.intervals):
            overlap_start = max(start, chunk_start)
            overlap_end = min(end, chunk_end)
            if overlap_start >= overlap_end:
                continue
            source_start = overlap_start - chunk_start
            target_start = overlap_start - start
            length = overlap_end - overlap_start
            self.audio[index][..., target_start:target_start + length] = (
                chunk[..., source_start:source_start + length]
            )
            self.written[index][target_start:target_start + length] = True

    def overlap_bounds(self, chunk_start: int, chunk_end: int) -> tuple[int, int] | None:
        """Smallest slice of a chunk containing all requested overlaps."""

        overlaps = [
            (max(start, chunk_start), min(end, chunk_end))
            for start, end in self.intervals
            if max(start, chunk_start) < min(end, chunk_end)
        ]
        if not overlaps:
            return None
        return min(start for start, _ in overlaps), max(end for _, end in overlaps)

    def finish(self) -> list[np.ndarray]:
        if not all(mask.all() for mask in self.written):
            raise RuntimeError("stream ended before all requested excerpts were captured")
        return self.audio


_BATCHED_IO_BLOCK_HOPS = 64
_MAX_EVALUATION_BATCH_STREAMS = 16
_PARITY_CALLBACKS = 64
_PARITY_ATOL = 1e-6
_PARITY_RTOL = 1e-5


class _BlockedAudioReader:
    """Read exact float32 WAV samples in larger blocks, then serve callback hops."""

    def __init__(
        self,
        audio_path: Path,
        *,
        expected_frames: int,
        hop: int,
        block_hops: int,
    ) -> None:
        if block_hops <= 0:
            raise ValueError("block_hops must be positive")
        self.audio_path = audio_path
        self.hop = hop
        self.block_frames = hop * block_hops
        self.file = sf.SoundFile(str(audio_path))
        if (
            self.file.samplerate != 44_100
            or self.file.channels != 2
            or self.file.frames != expected_frames
        ):
            self.file.close()
            raise RuntimeError(f"unexpected audio metadata: {audio_path}")
        self.buffer = np.empty((0, 2), dtype=np.float32)
        self.cursor = 0

    def read_hop(self) -> np.ndarray:
        if self.cursor == self.buffer.shape[0]:
            self.buffer = self.file.read(
                self.block_frames, dtype="float32", always_2d=True
            )
            self.cursor = 0
        end = min(self.cursor + self.hop, self.buffer.shape[0])
        block = self.buffer[self.cursor:end]
        self.cursor = end
        return block

    def close(self) -> None:
        self.file.close()


def _open_blocked_readers(
    audio_paths: Sequence[Path],
    expected_frames: Sequence[int],
    *,
    hop: int,
    block_hops: int,
) -> list[_BlockedAudioReader]:
    """Open readers incrementally so a later open failure closes earlier files."""

    if len(audio_paths) != len(expected_frames):
        raise ValueError("reader paths/frame counts must have equal lengths")
    readers: list[_BlockedAudioReader] = []
    try:
        for path, frames in zip(audio_paths, expected_frames):
            readers.append(_BlockedAudioReader(
                path,
                expected_frames=frames,
                hop=hop,
                block_hops=block_hops,
            ))
    except Exception:
        for reader in readers:
            reader.close()
        raise
    return readers


def _init_batched_stateful_transform(
    model: Any,
    *,
    batch_size: int,
    device: torch.device,
):
    """Evaluator-private batched equivalent of the production stream closure.

    Every batch item owns independent past audio, recurrent hidden state, and
    overlap-add state.  Keeping this helper here leaves the deployed B=1 API and
    its latency benchmark completely untouched.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    hop = int(model.overlap_len)
    channels = int(model.audio_channels)
    sources = int(model.num_sources)
    parameter = next(model.parameters(), None)
    model_dtype = parameter.dtype if parameter is not None else torch.float32
    model.eval()
    past_audio = torch.zeros(
        (batch_size, channels, hop), device=device, dtype=model_dtype
    )
    hiddens = None
    overlap_add_buffer = None

    @torch.inference_mode()
    def transform(audio_chunk: torch.Tensor) -> torch.Tensor:
        nonlocal past_audio, hiddens, overlap_add_buffer
        expected_input = (batch_size, channels, hop)
        if tuple(audio_chunk.shape) != expected_input:
            raise RuntimeError(
                f"batched streaming input shape {tuple(audio_chunk.shape)}, "
                f"expected {expected_input}"
            )
        if audio_chunk.device != device or audio_chunk.dtype != model_dtype:
            audio_chunk = audio_chunk.to(device=device, dtype=model_dtype)
        full_chunk = torch.cat((past_audio, audio_chunk), dim=-1)
        transformed, hiddens = model.forward(
            full_chunk,
            hiddens=hiddens,
            return_reduced_sources=None,
            is_streaming=True,
        )
        expected_output = (batch_size, sources, channels, hop * 2)
        if tuple(transformed.shape) != expected_output:
            raise RuntimeError(
                f"batched model output shape {tuple(transformed.shape)}, "
                f"expected {expected_output}"
            )
        if overlap_add_buffer is None:
            overlap_add_buffer = torch.zeros_like(transformed)
        overlap_add_buffer += transformed
        output = overlap_add_buffer[..., :hop].clone()
        overlap_add_buffer = torch.cat(
            (overlap_add_buffer[..., hop:], torch.zeros_like(output)), dim=-1
        )
        # CUDA input is a fresh host-to-device allocation on every callback.
        # On CPU, ``Tensor.to`` may alias the reusable NumPy staging buffer, so
        # retain an explicit copy of the one-hop causal state.
        past_audio = (
            audio_chunk.clone() if audio_chunk.device.type == "cpu" else audio_chunk
        )
        return output

    return transform


def _stream_audio_batch(
    model: Any,
    audio_paths: Sequence[Path],
    capture_intervals: Sequence[Sequence[tuple[int, int]]],
    *,
    device: torch.device,
    expected_frames: Sequence[int],
    io_block_hops: int = _BATCHED_IO_BLOCK_HOPS,
) -> tuple[list[list[np.ndarray]], dict[str, int]]:
    """Stream independent files concurrently without sharing causal state."""

    batch_size = len(audio_paths)
    if batch_size == 0:
        raise ValueError("at least one audio stream is required")
    if len(capture_intervals) != batch_size or len(expected_frames) != batch_size:
        raise ValueError("batched stream arguments must have equal lengths")
    hop = int(model.overlap_len)
    stream_ends: list[int] = []
    captures: list[_Capture] = []
    for intervals, frames in zip(capture_intervals, expected_frames):
        if not intervals:
            raise ValueError("every stream requires at least one capture interval")
        if any(start < 0 or end <= start or end > frames for start, end in intervals):
            raise ValueError("capture interval is outside the declared audio length")
        stream_ends.append(max(end for _, end in intervals))
        captures.append(_Capture(intervals, (len(SOURCE_ORDER), 2)))

    readers: list[_BlockedAudioReader] = []
    try:
        readers = _open_blocked_readers(
            audio_paths,
            expected_frames,
            hop=hop,
            block_hops=io_block_hops,
        )
        transform = _init_batched_stateful_transform(
            model, batch_size=batch_size, device=device
        )
        parameter = next(model.parameters(), None)
        model_dtype = parameter.dtype if parameter is not None else torch.float32
        input_block = np.empty((batch_size, 2, hop), dtype=np.float32)
        max_stream_end = max(stream_ends)
        callback_count = 0
        position = 0
        while position < max_stream_end:
            input_block.fill(0.0)
            valid_frames = [0] * batch_size
            for index, (reader, stream_end) in enumerate(zip(readers, stream_ends)):
                if position >= stream_end:
                    continue
                block = reader.read_hop()
                valid = int(block.shape[0])
                required = min(hop, stream_end - position)
                if valid < required:
                    raise RuntimeError(f"audio ended early: {audio_paths[index]}")
                input_block[index, :, :valid] = block.T
                valid_frames[index] = valid

            tensor = torch.from_numpy(input_block).to(
                device=device, dtype=model_dtype
            )
            output = transform(tensor)
            expected_shape = (batch_size, len(SOURCE_ORDER), 2, hop)
            if tuple(output.shape) != expected_shape:
                raise RuntimeError(
                    f"batched streaming output shape {tuple(output.shape)}, "
                    f"expected {expected_shape}"
                )

            overlaps = [
                capture.overlap_bounds(position, position + valid)
                if valid > 0 else None
                for capture, valid in zip(captures, valid_frames)
            ]
            if any(overlap is not None for overlap in overlaps):
                cpu_output = (
                    output.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
                for index, overlap in enumerate(overlaps):
                    if overlap is None:
                        continue
                    overlap_start, overlap_end = overlap
                    source_start = overlap_start - position
                    source_end = overlap_end - position
                    captured_output = cpu_output[index, ..., source_start:source_end]
                    if not np.isfinite(captured_output).all():
                        raise RuntimeError(
                            f"non-finite streaming output from {audio_paths[index]}"
                        )
                    captures[index].add(overlap_start, captured_output)
            position += hop
            callback_count += 1
    finally:
        for reader in readers:
            reader.close()

    scalar_callback_count = sum(
        (stream_end + hop - 1) // hop for stream_end in stream_ends
    )
    return [capture.finish() for capture in captures], {
        "batch_size": batch_size,
        "callback_count": callback_count,
        "scalar_equivalent_callback_count": scalar_callback_count,
        "io_block_hops": io_block_hops,
    }


def _capture_batched_parity_prefixes(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    callback_count: int = _PARITY_CALLBACKS,
) -> list[np.ndarray]:
    """Capture real-prefix output from fresh evaluator-private batch closures."""

    if callback_count <= 0:
        raise ValueError("parity callback_count must be positive")
    hop = int(model.overlap_len)
    captured_batches: list[np.ndarray] = []
    for batch in batches:
        audio_paths = batch["audio_paths"]
        expected_frames = batch["expected_frames"]
        batch_size = len(audio_paths)
        if len(expected_frames) != batch_size or batch_size <= 0:
            raise ValueError("invalid parity batch inputs")
        readers: list[_BlockedAudioReader] = []
        try:
            readers = _open_blocked_readers(
                audio_paths,
                expected_frames,
                hop=hop,
                block_hops=callback_count,
            )
            transform = _init_batched_stateful_transform(
                model, batch_size=batch_size, device=device
            )
            parameter = next(model.parameters(), None)
            model_dtype = parameter.dtype if parameter is not None else torch.float32
            outputs = np.empty(
                (
                    callback_count,
                    batch_size,
                    len(SOURCE_ORDER),
                    int(model.audio_channels),
                    hop,
                ),
                dtype=np.float32,
            )
            input_block = np.empty(
                (batch_size, int(model.audio_channels), hop), dtype=np.float32
            )
            for callback_index in range(callback_count):
                for stream_index, reader in enumerate(readers):
                    block = reader.read_hop()
                    if tuple(block.shape) != (hop, int(model.audio_channels)):
                        raise RuntimeError(
                            "real parity prefix ended before callback "
                            f"{callback_index + 1}: {audio_paths[stream_index]}"
                        )
                    input_block[stream_index] = block.T
                tensor = torch.from_numpy(input_block).to(
                    device=device, dtype=model_dtype
                )
                output = transform(tensor)
                expected_shape = (
                    batch_size,
                    len(SOURCE_ORDER),
                    int(model.audio_channels),
                    hop,
                )
                if tuple(output.shape) != expected_shape:
                    raise RuntimeError(
                        f"batched parity output shape {tuple(output.shape)}, "
                        f"expected {expected_shape}"
                    )
                cpu_output = (
                    output.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
                if not np.isfinite(cpu_output).all():
                    raise RuntimeError("non-finite batched parity output")
                outputs[callback_index] = cpu_output
            captured_batches.append(outputs)
        finally:
            for reader in readers:
                reader.close()
    return captured_batches


def _compare_production_parity_prefixes(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    batched_outputs: Sequence[np.ndarray],
    *,
    device: torch.device,
    callback_count: int = _PARITY_CALLBACKS,
    atol: float = _PARITY_ATOL,
    rtol: float = _PARITY_RTOL,
) -> list[dict[str, Any]]:
    """Compare independent production B=1 closures with stored batch output."""

    if len(batches) != len(batched_outputs):
        raise ValueError("parity batch/output count differs")
    if callback_count <= 0 or atol < 0.0 or rtol < 0.0:
        raise ValueError("invalid parity comparison configuration")
    hop = int(model.overlap_len)
    records: list[dict[str, Any]] = []
    for batch, expected in zip(batches, batched_outputs):
        audio_paths = batch["audio_paths"]
        expected_frames = batch["expected_frames"]
        labels = batch["stream_labels"]
        batch_size = len(audio_paths)
        expected_shape = (
            callback_count,
            batch_size,
            len(SOURCE_ORDER),
            int(model.audio_channels),
            hop,
        )
        if tuple(expected.shape) != expected_shape:
            raise RuntimeError(
                f"stored batched parity shape {tuple(expected.shape)}, "
                f"expected {expected_shape}"
            )
        transforms = [
            model.init_stateful_transform_fn(
                device=device,
                return_reduced_sources=None,
                auto_convert_to_stereo=False,
            )
            for _ in range(batch_size)
        ]
        readers: list[_BlockedAudioReader] = []
        max_abs_error = 0.0
        max_normalized_error = 0.0
        try:
            readers = _open_blocked_readers(
                audio_paths,
                expected_frames,
                hop=hop,
                block_hops=callback_count,
            )
            for callback_index in range(callback_count):
                for stream_index, (reader, transform) in enumerate(
                    zip(readers, transforms)
                ):
                    block = reader.read_hop()
                    if tuple(block.shape) != (hop, int(model.audio_channels)):
                        raise RuntimeError(
                            "production parity prefix ended before callback "
                            f"{callback_index + 1}: {audio_paths[stream_index]}"
                        )
                    output = transform(np.ascontiguousarray(block.T))
                    if isinstance(output, torch.Tensor):
                        output = (
                            output.detach()
                            .to(device="cpu", dtype=torch.float32)
                            .numpy()
                        )
                    actual = np.asarray(output)
                    production_shape = (
                        len(SOURCE_ORDER), int(model.audio_channels), hop
                    )
                    if tuple(actual.shape) != production_shape:
                        raise RuntimeError(
                            f"production parity output shape {tuple(actual.shape)}, "
                            f"expected {production_shape}"
                        )
                    if actual.dtype != np.float32 or not np.isfinite(actual).all():
                        raise RuntimeError(
                            "production parity output must be finite float32"
                        )
                    reference = actual.astype(np.float64, copy=False)
                    proposed = expected[callback_index, stream_index].astype(
                        np.float64, copy=False
                    )
                    difference = np.abs(proposed - reference)
                    tolerance = atol + rtol * np.abs(reference)
                    normalized = difference / tolerance
                    local_abs = float(np.max(difference))
                    local_normalized = float(np.max(normalized))
                    max_abs_error = max(max_abs_error, local_abs)
                    max_normalized_error = max(
                        max_normalized_error, local_normalized
                    )
                    if not np.all(difference <= tolerance):
                        raise RuntimeError(
                            "batched/production streaming parity failed for "
                            f"{batch['kind']} stream {labels[stream_index]!r} at "
                            f"callback {callback_index + 1}: max_abs={local_abs:.9g}, "
                            f"max_normalized={local_normalized:.9g}"
                        )
        finally:
            for reader in readers:
                reader.close()
        records.append({
            "kind": batch["kind"],
            "batch_index": batch["batch_index"],
            "stream_labels": list(labels),
            "batch_size": batch_size,
            "callback_count": callback_count,
            "prefix_samples": callback_count * hop,
            "batched_model_callback_count": callback_count,
            "production_b1_model_callback_count": callback_count * batch_size,
            "max_abs_error": max_abs_error,
            "max_normalized_error": max_normalized_error,
            "pass": True,
        })
    return records


def _stream_audio(
    model: Any,
    audio_path: Path,
    capture_intervals: Sequence[tuple[int, int]],
    *,
    device: torch.device,
    expected_frames: int,
) -> list[np.ndarray]:
    """Stream the continuous prefix through the final requested output sample.

    State advances through every callback from sample zero, including callbacks
    between excerpts.  The pass stops after the callback containing the maximum
    capture end.  Output remains on its inference device for callbacks which do
    not intersect a capture interval.
    """

    hop = int(model.overlap_len)
    if not capture_intervals:
        raise ValueError("at least one capture interval is required")
    if any(start < 0 or end <= start or end > expected_frames
           for start, end in capture_intervals):
        raise ValueError("capture interval is outside the declared audio length")
    stream_end = max(end for _, end in capture_intervals)
    transform = model.init_stateful_transform_fn(
        device=device,
        return_reduced_sources=None,
        auto_convert_to_stereo=False,
    )
    capture = _Capture(capture_intervals, (len(SOURCE_ORDER), 2))
    parameter = next(model.parameters(), None)
    model_dtype = parameter.dtype if parameter is not None else torch.float32
    position = 0
    with sf.SoundFile(str(audio_path)) as file:
        if file.samplerate != 44_100 or file.channels != 2 or file.frames != expected_frames:
            raise RuntimeError(f"unexpected audio metadata: {audio_path}")
        while position < stream_end:
            block = file.read(hop, dtype="float32", always_2d=True)
            valid = block.shape[0]
            if valid == 0:
                raise RuntimeError(f"audio ended early: {audio_path}")
            if valid < hop:
                block = np.pad(block, ((0, hop - valid), (0, 0)))
            tensor = torch.from_numpy(np.ascontiguousarray(block.T)).to(
                device=device, dtype=model_dtype
            )
            output = transform(tensor)
            if not isinstance(output, torch.Tensor):
                output = torch.as_tensor(output)
            expected_shape = (len(SOURCE_ORDER), 2, hop)
            if output.shape != expected_shape:
                raise RuntimeError(
                    f"streaming output shape {output.shape}, expected {expected_shape}"
                )
            overlap = capture.overlap_bounds(position, position + valid)
            if overlap is not None:
                overlap_start, overlap_end = overlap
                source_start = overlap_start - position
                source_end = overlap_end - position
                captured_output = (
                    output[..., source_start:source_end]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                )
                if not np.isfinite(captured_output).all():
                    raise RuntimeError(
                        f"non-finite streaming output from {audio_path}"
                    )
                capture.add(overlap_start, captured_output)
            position += valid
    return capture.finish()


def _read_excerpt(
    path: Path, start: int, end: int, *, expected_frames: int
) -> np.ndarray:
    with sf.SoundFile(str(path)) as file:
        if (
            file.samplerate != 44_100
            or file.channels != 2
            or file.frames != expected_frames
        ):
            raise RuntimeError(f"unexpected reference audio metadata: {path}")
        file.seek(start)
        audio = file.read(end - start, dtype="float64", always_2d=True)
    if audio.shape != (end - start, 2):
        raise RuntimeError(f"short excerpt read from {path}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"non-finite reference audio in {path}")
    return np.ascontiguousarray(audio.T)


def _weighted(results: Iterable[Mapping[str, Any]], value: str, weight: str) -> float | None:
    pairs = [
        (float(item[value]), int(item[weight]))
        for item in results
        if item.get(value) is not None and int(item.get(weight, 0)) > 0
    ]
    total = sum(item_weight for _, item_weight in pairs)
    if total == 0:
        return None
    return float(sum(item_value * item_weight for item_value, item_weight in pairs) / total)


def _mean_matrix(matrices: Sequence[Sequence[Sequence[float | None]]]) -> list[list[float | None]]:
    if not matrices:
        return []
    rows, columns = len(matrices[0]), len(matrices[0][0])
    return [
        [mean_or_none(matrix[row][column] for matrix in matrices)
         for column in range(columns)]
        for row in range(rows)
    ]


def _score_track(
    name: str,
    intervals: Sequence[Mapping[str, int | str]],
    mixtures: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    estimates: Sequence[np.ndarray],
    metric_config: MetricConfig,
) -> dict[str, Any]:
    full = [[] for _ in SOURCE_ORDER]
    bands = {
        band_name: [[] for _ in SOURCE_ORDER]
        for band_name in metric_config.bands_hz
    }
    projections = []
    consistencies = []
    for mixture, refs, ests in zip(mixtures, references, estimates):
        for source in range(len(SOURCE_ORDER)):
            full[source].append(windowed_sdr(refs[source], ests[source], metric_config))
            for band_name, limits in metric_config.bands_hz.items():
                bands[band_name][source].append(
                    band_sdr(refs[source], ests[source], limits, metric_config)
                )
        projections.append(projection_metrics(refs, ests, metric_config))
        consistencies.append(mixture_consistency(mixture, ests, metric_config))

    per_stem: dict[str, Any] = {}
    for source, source_name in enumerate(SOURCE_ORDER):
        projection_items = [item["per_stem"][source] for item in projections]
        per_stem[source_name] = {
            "full_sdr_db": _weighted(full[source], "db", "active_windows"),
            "band_sdr_db": {
                band_name: _weighted(bands[band_name][source], "db", "active_windows")
                for band_name in metric_config.bands_hz
            },
            "sir_db": _weighted(projection_items, "sir_db", "active_windows"),
            "absent_fp_dbfs": _weighted(
                projection_items, "absent_fp_dbfs", "absent_windows"
            ),
            "absent_fp_ratio_db": _weighted(
                projection_items, "absent_fp_ratio_db", "absent_windows"
            ),
            "active_windows": sum(item["active_windows"] for item in full[source]),
            "absent_windows": sum(item["absent_windows"] for item in projection_items),
        }

    return {
        "name": name,
        "excerpts": [dict(interval) for interval in intervals],
        "full_sdr_db": mean_or_none(
            per_stem[source]["full_sdr_db"] for source in SOURCE_ORDER
        ),
        "low_sdr_db": mean_or_none(
            per_stem[source]["band_sdr_db"]["low_20_250"]
            for source in SOURCE_ORDER
        ),
        "band_sdr_db": {
            band_name: mean_or_none(
                per_stem[source]["band_sdr_db"][band_name]
                for source in SOURCE_ORDER
            )
            for band_name in metric_config.bands_hz
        },
        "bleed_sir_db": mean_or_none(
            per_stem[source]["sir_db"] for source in SOURCE_ORDER
        ),
        "per_stem": per_stem,
        "projection_attribution_db": _mean_matrix(
            [item["projection_attribution_db"] for item in projections]
        ),
        "mixture_consistency_db": mean_or_none(
            item["db"] for item in consistencies
        ),
        "mixture_consistency_error_rms": mean_or_none(
            item["error_rms"] for item in consistencies
        ),
    }


def _score_single_source_track(
    probe_outputs: Sequence[Sequence[np.ndarray]],
    references: Sequence[np.ndarray],
    metric_config: MetricConfig,
) -> dict[str, Any]:
    """Aggregate isolated-input probes without treating silence as a ratio.

    Energy ratios and desired-head retention use only excerpts whose isolated
    input is above the frozen activity threshold. Inactive excerpts instead
    report absolute output RMS dBFS, avoiding the undefined zero/zero ratio.
    Low-band ratios use the same rule after the frozen 20-250 Hz filter.
    """

    rows = len(SOURCE_ORDER)
    output_matrix: list[list[float | None]] = []
    low_matrix: list[list[float | None]] = []
    silent_matrix: list[list[float | None]] = []
    low_silent_matrix: list[list[float | None]] = []
    desired: dict[str, Any] = {}
    activity: dict[str, Any] = {}
    low_band = metric_config.bands_hz["low_20_250"]
    for input_source in range(rows):
        active_metrics = []
        low_active_metrics = []
        silent_outputs: list[list[float]] = []
        low_silent_outputs: list[list[float]] = []
        for refs, outputs in zip(references, probe_outputs[input_source]):
            reference = refs[input_source]
            input_active = (
                rms_dbfs(reference, metric_config.epsilon)
                > metric_config.activity_dbfs
            )
            low_reference = fft_bandpass(
                reference, metric_config.sample_rate, *low_band
            )
            low_input_active = (
                rms_dbfs(low_reference, metric_config.epsilon)
                > metric_config.activity_dbfs
            )
            metrics = None
            if input_active or low_input_active:
                metrics = single_source_probe_metrics(
                    reference, outputs, input_source, metric_config
                )
            if input_active:
                active_metrics.append(metrics)
            else:
                silent_outputs.append([
                    rms_dbfs(output, metric_config.epsilon) for output in outputs
                ])
            if low_input_active:
                low_active_metrics.append(metrics)
            else:
                low_silent_outputs.append([
                    rms_dbfs(
                        fft_bandpass(
                            output, metric_config.sample_rate, *low_band
                        ),
                        metric_config.epsilon,
                    )
                    for output in outputs
                ])

        output_matrix.append([
            mean_or_none(item["output_to_input_db"][head] for item in active_metrics)
            for head in range(rows)
        ])
        low_matrix.append([
            mean_or_none(
                item["low_output_to_input_db"][head] for item in low_active_metrics
            )
            for head in range(rows)
        ])
        silent_matrix.append([
            mean_or_none(item[head] for item in silent_outputs)
            for head in range(rows)
        ])
        low_silent_matrix.append([
            mean_or_none(item[head] for item in low_silent_outputs)
            for head in range(rows)
        ])
        desired[SOURCE_ORDER[input_source]] = {
            "sdr_db": mean_or_none(item["desired_sdr_db"] for item in active_metrics),
            "gain_db": mean_or_none(item["desired_gain_db"] for item in active_metrics),
            "low_sdr_db": mean_or_none(
                item["desired_low_sdr_db"] for item in low_active_metrics
            ),
        }
        activity[SOURCE_ORDER[input_source]] = {
            "active_excerpts": len(active_metrics),
            "silent_excerpts": len(silent_outputs),
            "low_active_excerpts": len(low_active_metrics),
            "low_silent_excerpts": len(low_silent_outputs),
        }
    off_target = [
        output_matrix[row][column]
        for row in range(rows) for column in range(rows) if row != column
    ]
    low_off_target = [
        low_matrix[row][column]
        for row in range(rows) for column in range(rows) if row != column
    ]
    return {
        "desired_head_retention": desired,
        "input_activity": activity,
        "output_to_input_db": output_matrix,
        "low_output_to_input_db": low_matrix,
        "silent_input_fp_dbfs": silent_matrix,
        "low_silent_input_fp_dbfs": low_silent_matrix,
        "matrix_rows": "isolated_input_source",
        "matrix_columns": "estimated_head",
        "activity_threshold_dbfs": metric_config.activity_dbfs,
        "off_target_leakage_db": mean_or_none(off_target),
        "low_off_target_leakage_db": mean_or_none(low_off_target),
    }


def _aggregate_tracks(tracks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_stem: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        per_stem[source] = {
            "full_sdr_db": mean_or_none(
                track["per_stem"][source]["full_sdr_db"] for track in tracks
            ),
            "band_sdr_db": {
                band: mean_or_none(
                    track["per_stem"][source]["band_sdr_db"][band]
                    for track in tracks
                )
                for band in tracks[0]["per_stem"][source]["band_sdr_db"]
            },
            "sir_db": mean_or_none(
                track["per_stem"][source]["sir_db"] for track in tracks
            ),
            "absent_fp_dbfs": mean_or_none(
                track["per_stem"][source]["absent_fp_dbfs"] for track in tracks
            ),
            "absent_fp_ratio_db": mean_or_none(
                track["per_stem"][source]["absent_fp_ratio_db"] for track in tracks
            ),
        }
    primary_values = [
        value
        for source in SOURCE_ORDER
        for value in (
            per_stem[source]["full_sdr_db"],
            per_stem[source]["band_sdr_db"]["low_20_250"],
            per_stem[source]["sir_db"],
        )
    ]
    if any(value is None for value in primary_values):
        raise RuntimeError(
            "insufficient active validation windows for every primary stem metric"
        )
    full_sdr = float(np.mean([
        per_stem[source]["full_sdr_db"] for source in SOURCE_ORDER
    ]))
    low_sdr = float(np.mean([
        per_stem[source]["band_sdr_db"]["low_20_250"] for source in SOURCE_ORDER
    ]))
    bleed_sir = float(np.mean([
        per_stem[source]["sir_db"] for source in SOURCE_ORDER
    ]))
    aggregate: dict[str, Any] = {
        "val_score": 0.50 * full_sdr + 0.25 * low_sdr + 0.25 * bleed_sir,
        "full_sdr_db": full_sdr,
        "low_sdr_db": low_sdr,
        "band_sdr_db": {
            band: mean_or_none(
                per_stem[source]["band_sdr_db"][band] for source in SOURCE_ORDER
            )
            for band in tracks[0]["band_sdr_db"]
        },
        "bleed_sir_db": bleed_sir,
        "per_stem": per_stem,
        "projection_attribution_db": _mean_matrix(
            [track["projection_attribution_db"] for track in tracks]
        ),
        "mixture_consistency_db": mean_or_none(
            track["mixture_consistency_db"] for track in tracks
        ),
        "mixture_consistency_error_rms": mean_or_none(
            track["mixture_consistency_error_rms"] for track in tracks
        ),
    }
    probe_tracks = [track for track in tracks if "single_source" in track]
    if probe_tracks:
        output_matrix = _mean_matrix([
            track["single_source"]["output_to_input_db"]
            for track in probe_tracks
        ])
        low_output_matrix = _mean_matrix([
            track["single_source"]["low_output_to_input_db"]
            for track in probe_tracks
        ])
        silent_matrix = _mean_matrix([
            track["single_source"]["silent_input_fp_dbfs"]
            for track in probe_tracks
        ])
        low_silent_matrix = _mean_matrix([
            track["single_source"]["low_silent_input_fp_dbfs"]
            for track in probe_tracks
        ])
        off_target = [
            output_matrix[row][column]
            for row in range(len(SOURCE_ORDER))
            for column in range(len(SOURCE_ORDER))
            if row != column
        ]
        low_off_target = [
            low_output_matrix[row][column]
            for row in range(len(SOURCE_ORDER))
            for column in range(len(SOURCE_ORDER))
            if row != column
        ]
        aggregate["single_source"] = {
            "track_names": [track["name"] for track in probe_tracks],
            "track_count": len(probe_tracks),
            "desired_head_retention": {
                source: {
                    key: mean_or_none(
                        track["single_source"]["desired_head_retention"][source][key]
                        for track in probe_tracks
                    )
                    for key in ("sdr_db", "gain_db", "low_sdr_db")
                }
                for source in SOURCE_ORDER
            },
            "input_activity": {
                source: {
                    key: sum(
                        track["single_source"]["input_activity"][source][key]
                        for track in probe_tracks
                    )
                    for key in (
                        "active_excerpts",
                        "silent_excerpts",
                        "low_active_excerpts",
                        "low_silent_excerpts",
                    )
                }
                for source in SOURCE_ORDER
            },
            "output_to_input_db": output_matrix,
            "low_output_to_input_db": low_output_matrix,
            "silent_input_fp_dbfs": silent_matrix,
            "low_silent_input_fp_dbfs": low_silent_matrix,
            "silent_input_fp_dbfs_by_output": [
                mean_or_none(silent_matrix[row][column]
                             for row in range(len(SOURCE_ORDER)))
                for column in range(len(SOURCE_ORDER))
            ],
            "low_silent_input_fp_dbfs_by_output": [
                mean_or_none(low_silent_matrix[row][column]
                             for row in range(len(SOURCE_ORDER)))
                for column in range(len(SOURCE_ORDER))
            ],
            "matrix_rows": "isolated_input_source",
            "matrix_columns": "estimated_head",
            "activity_threshold_dbfs": probe_tracks[0]["single_source"][
                "activity_threshold_dbfs"
            ],
            "off_target_leakage_db": mean_or_none(off_target),
            "low_off_target_leakage_db": mean_or_none(low_off_target),
        }
    return aggregate


def _finite_metric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _probe_metrics_complete(
    aggregate: Mapping[str, Any], expected_track_names: Sequence[str]
) -> bool:
    """Check that every active-input guardrail and silent-FP head is defined."""

    single = aggregate.get("single_source")
    if not isinstance(single, dict):
        return False
    if single.get("track_names") != list(expected_track_names):
        return False
    if single.get("track_count") != len(expected_track_names):
        return False
    for key in ("output_to_input_db", "low_output_to_input_db"):
        matrix = single.get(key)
        if not isinstance(matrix, list) or len(matrix) != len(SOURCE_ORDER):
            return False
        if any(
            not isinstance(row, list)
            or len(row) != len(SOURCE_ORDER)
            or not all(_finite_metric(value) for value in row)
            for row in matrix
        ):
            return False
    desired = single.get("desired_head_retention")
    activity = single.get("input_activity")
    if not isinstance(desired, dict) or not isinstance(activity, dict):
        return False
    for source in SOURCE_ORDER:
        if not all(
            _finite_metric(desired.get(source, {}).get(key))
            for key in ("sdr_db", "gain_db", "low_sdr_db")
        ):
            return False
        if activity.get(source, {}).get("active_excerpts", 0) <= 0:
            return False
        if activity.get(source, {}).get("low_active_excerpts", 0) <= 0:
            return False
    silent_count = sum(
        int(activity[source].get("silent_excerpts", 0)) for source in SOURCE_ORDER
    )
    if silent_count <= 0:
        return False
    silent_by_output = single.get("silent_input_fp_dbfs_by_output")
    return (
        isinstance(silent_by_output, list)
        and len(silent_by_output) == len(SOURCE_ORDER)
        and all(_finite_metric(value) for value in silent_by_output)
    )


def _contract_is_complete(
    *,
    sealed_test: bool,
    selected_track_count: int,
    manifest_track_count: int,
    max_tracks: int | None,
    probes_disabled: bool,
    configured_probe_tracks: Sequence[str],
    probed_tracks: Sequence[str],
    probe_metrics_complete: bool | None,
) -> bool:
    common = (
        max_tracks is None
        and selected_track_count == manifest_track_count
        and not probes_disabled
    )
    if sealed_test:
        return (
            common
            and manifest_track_count == 46
            and not configured_probe_tracks
            and not probed_tracks
        )
    return (
        common
        and manifest_track_count == 14
        and bool(configured_probe_tracks)
        and tuple(probed_tracks) == tuple(configured_probe_tracks)
        and probe_metrics_complete is True
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = _sha256_file(checkpoint)
    if args.checkpoint_sha256 and checkpoint_hash.lower() != args.checkpoint_sha256.lower():
        raise RuntimeError("checkpoint SHA-256 mismatch")

    manifest, manifest_file_hash = _load_json(
        args.manifest.expanduser().resolve(), args.manifest_sha256
    )
    _validate_manifest(manifest, allow_sealed_test=args.allow_sealed_test)
    sealed_test = manifest["split"] == "test"
    config, config_file_hash = _load_json(
        args.config.expanduser().resolve(), args.config_sha256
    )
    metric_config = _validate_config(config, manifest)
    root = Path(manifest["root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_root = Path("/home/axel/HS-TasNet/data/musdb18hq").resolve()
    if root != expected_root:
        raise RuntimeError(
            f"validation manifest root changed: expected {expected_root}, got {root}"
        )

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device_name = args.device or config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Parity uses two independently loaded checkpoint instances, and scoring
    # uses a third fresh instance so no closure or model-local parity state can
    # enter the authoritative metric pass.
    from hs_tasnet.hs_tasnet import HSTasNet
    required_deployment = {
        "sample_rate": 44_100,
        "audio_channels": 2,
        "num_sources": 4,
        "segment_len": 1024,
        "overlap_len": 512,
        "n_fft": 1024,
    }

    def load_model() -> tuple[Any, dict[str, int], str]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        loaded = HSTasNet.init_and_load_from(checkpoint)
        loaded.to(device).eval()
        loaded_deployment = {
            "sample_rate": loaded.sample_rate,
            "audio_channels": loaded.audio_channels,
            "num_sources": loaded.num_sources,
            "segment_len": loaded.segment_len,
            "overlap_len": loaded.overlap_len,
            "n_fft": loaded.n_fft,
        }
        if loaded_deployment != required_deployment:
            raise RuntimeError(
                "checkpoint violates frozen deployment shape: "
                f"{loaded_deployment}"
            )
        return loaded, loaded_deployment, _model_state_sha256(loaded)

    model, deployment, batched_model_state_sha256 = load_model()

    selected_tracks = list(manifest["tracks"])
    if args.max_tracks is not None:
        if args.max_tracks <= 0:
            raise ValueError("max_tracks must be positive")
        selected_tracks = selected_tracks[:args.max_tracks]
    configured_probe_tracks = _configured_probe_tracks(config, manifest)
    if sealed_test and configured_probe_tracks:
        raise ValueError("sealed test evaluation forbids single-source probes")
    probe_track_set = (
        set() if args.disable_single_source_probes else set(configured_probe_tracks)
    )

    started = time.monotonic()
    contexts: list[dict[str, Any]] = []
    for track in selected_tracks:
        name = track["name"]
        intervals = _reference_intervals(track, config)
        estimate_intervals = [
            (int(item["estimate_start"]), int(item["estimate_end"]))
            for item in intervals
        ]
        reference_intervals = [
            (int(item["reference_start"]), int(item["reference_end"]))
            for item in intervals
        ]
        mixture_path = _safe_dataset_path(root, track["mixture"])
        stem_paths = [
            _safe_dataset_path(root, track["stems"][source]) for source in SOURCE_ORDER
        ]
        contexts.append({
            "name": name,
            "intervals": intervals,
            "estimate_intervals": estimate_intervals,
            "reference_intervals": reference_intervals,
            "mixture_path": mixture_path,
            "stem_paths": stem_paths,
            "frames": int(track["frames"]),
        })

    track_results: list[dict[str, Any]] = []
    execution_batches: list[dict[str, Any]] = []
    legacy_unbatched = bool(getattr(args, "legacy_unbatched", False))

    if legacy_unbatched:
        streaming_parity: dict[str, Any] = {
            "schema_version": 1,
            "required": False,
            "pass": None,
            "real_audio_prefix": True,
            "comparison": "batched_direct_forward_vs_independent_production_b1",
            "callbacks_per_stream": _PARITY_CALLBACKS,
            "prefix_samples": _PARITY_CALLBACKS * int(model.overlap_len),
            "atol": _PARITY_ATOL,
            "rtol": _PARITY_RTOL,
            "independent_checkpoint_model_instances": False,
            "checkpoint_sha256": checkpoint_hash,
            "model_state_sha256": batched_model_state_sha256,
            "model_state_identity_pass": True,
            "constructor_seed_reset_per_load": True,
            "model_load_count": 1,
            "parity_models_discarded_before_scoring": None,
            "state_reused_for_scoring": None,
            "batched_model_callback_count": 0,
            "production_b1_model_callback_count": 0,
            "batches": [],
        }
    else:
        parity_batches: list[dict[str, Any]] = []
        for group_start in range(0, len(contexts), _MAX_EVALUATION_BATCH_STREAMS):
            group = contexts[
                group_start:group_start + _MAX_EVALUATION_BATCH_STREAMS
            ]
            parity_batches.append({
                "kind": "mixture",
                "batch_index": group_start // _MAX_EVALUATION_BATCH_STREAMS + 1,
                "stream_labels": [context["name"] for context in group],
                "audio_paths": [context["mixture_path"] for context in group],
                "expected_frames": [context["frames"] for context in group],
            })
        parity_probe_contexts = [
            context for context in contexts if context["name"] in probe_track_set
        ]
        parity_probe_tracks_per_batch = max(
            1, _MAX_EVALUATION_BATCH_STREAMS // len(SOURCE_ORDER)
        )
        for group_start in range(
            0, len(parity_probe_contexts), parity_probe_tracks_per_batch
        ):
            group = parity_probe_contexts[
                group_start:group_start + parity_probe_tracks_per_batch
            ]
            parity_batches.append({
                "kind": "isolated_source_probe",
                "batch_index": group_start // parity_probe_tracks_per_batch + 1,
                "stream_labels": [
                    f"{context['name']}::{SOURCE_ORDER[source]}"
                    for context in group
                    for source in range(len(SOURCE_ORDER))
                ],
                "audio_paths": [
                    context["stem_paths"][source]
                    for context in group
                    for source in range(len(SOURCE_ORDER))
                ],
                "expected_frames": [
                    context["frames"]
                    for context in group
                    for _ in SOURCE_ORDER
                ],
            })

        print("capturing batched real-prefix parity evidence", flush=True)
        batched_parity_outputs = _capture_batched_parity_prefixes(
            model,
            parity_batches,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        (
            production_parity_model,
            parity_deployment,
            production_model_state_sha256,
        ) = load_model()
        if parity_deployment != deployment:
            raise RuntimeError("independent parity model deployment differs")
        if production_model_state_sha256 != batched_model_state_sha256:
            raise RuntimeError("independent parity model state differs")
        print("checking independent production B=1 parity", flush=True)
        parity_records = _compare_production_parity_prefixes(
            production_parity_model,
            parity_batches,
            batched_parity_outputs,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del production_parity_model
        del batched_parity_outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model, scoring_deployment, scoring_model_state_sha256 = load_model()
        if scoring_deployment != deployment:
            raise RuntimeError("fresh scoring model deployment differs")
        if scoring_model_state_sha256 != batched_model_state_sha256:
            raise RuntimeError("fresh scoring model state differs from parity models")
        streaming_parity = {
            "schema_version": 1,
            "required": True,
            "pass": True,
            "real_audio_prefix": True,
            "comparison": "batched_direct_forward_vs_independent_production_b1",
            "callbacks_per_stream": _PARITY_CALLBACKS,
            "prefix_samples": _PARITY_CALLBACKS * int(model.overlap_len),
            "atol": _PARITY_ATOL,
            "rtol": _PARITY_RTOL,
            "independent_checkpoint_model_instances": True,
            "checkpoint_sha256": checkpoint_hash,
            "model_state_sha256": batched_model_state_sha256,
            "model_state_identity_pass": True,
            "constructor_seed_reset_per_load": True,
            "model_load_count": 3,
            "parity_models_discarded_before_scoring": True,
            "state_reused_for_scoring": False,
            "batched_model_callback_count": sum(
                record["batched_model_callback_count"] for record in parity_records
            ),
            "production_b1_model_callback_count": sum(
                record["production_b1_model_callback_count"]
                for record in parity_records
            ),
            "batches": parity_records,
        }

    parameter = next(model.parameters())
    backend = {
        "model_dtype": str(parameter.dtype),
        "metric_dtype": "numpy.float64",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": getattr(
            torch.backends.cuda.matmul, "allow_tf32", None
        ),
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }

    if legacy_unbatched:
        mixture_callbacks = 0
        probe_callbacks = 0
        hop = int(model.overlap_len)
        for track_index, context in enumerate(contexts, start=1):
            name = context["name"]
            print(
                f"evaluating {track_index}/{len(contexts)}: {name} "
                "(legacy B=1)",
                flush=True,
            )
            estimates = _stream_audio(
                model,
                context["mixture_path"],
                context["estimate_intervals"],
                device=device,
                expected_frames=context["frames"],
            )
            mixture_callbacks += (
                max(end for _, end in context["estimate_intervals"]) + hop - 1
            ) // hop
            mixtures = [
                _read_excerpt(
                    context["mixture_path"], start, end,
                    expected_frames=context["frames"],
                )
                for start, end in context["reference_intervals"]
            ]
            references = [
                np.stack([
                    _read_excerpt(
                        path, start, end, expected_frames=context["frames"]
                    )
                    for path in context["stem_paths"]
                ])
                for start, end in context["reference_intervals"]
            ]
            result = _score_track(
                name,
                context["intervals"],
                mixtures,
                references,
                estimates,
                metric_config,
            )
            if name in probe_track_set:
                probe_outputs = [
                    _stream_audio(
                        model,
                        context["stem_paths"][source],
                        context["estimate_intervals"],
                        device=device,
                        expected_frames=context["frames"],
                    )
                    for source in range(len(SOURCE_ORDER))
                ]
                probe_callbacks += len(SOURCE_ORDER) * (
                    max(end for _, end in context["estimate_intervals"]) + hop - 1
                ) // hop
                result["single_source"] = _score_single_source_track(
                    probe_outputs, references, metric_config
                )
            track_results.append(result)
        execution_batches.append({
            "kind": "mixture",
            "batch_size": 1,
            "batch_count": len(contexts),
            "stream_count": len(contexts),
            "callback_count": mixture_callbacks,
            "scalar_equivalent_callback_count": mixture_callbacks,
            "io_block_hops": 1,
        })
        if probe_track_set:
            probe_stream_count = len(probe_track_set.intersection(
                context["name"] for context in contexts
            )) * len(SOURCE_ORDER)
            execution_batches.append({
                "kind": "isolated_source_probe",
                "batch_size": 1,
                "batch_count": probe_stream_count,
                "stream_count": probe_stream_count,
                "callback_count": probe_callbacks,
                "scalar_equivalent_callback_count": probe_callbacks,
                "io_block_hops": 1,
            })
    else:
        for group_start in range(0, len(contexts), _MAX_EVALUATION_BATCH_STREAMS):
            group = contexts[
                group_start:group_start + _MAX_EVALUATION_BATCH_STREAMS
            ]
            print(
                f"streaming mixture batch {group_start // _MAX_EVALUATION_BATCH_STREAMS + 1}: "
                f"B={len(group)}",
                flush=True,
            )
            batch_outputs, batch_metadata = _stream_audio_batch(
                model,
                [context["mixture_path"] for context in group],
                [context["estimate_intervals"] for context in group],
                device=device,
                expected_frames=[context["frames"] for context in group],
            )
            execution_batches.append({
                "kind": "mixture",
                "batch_index": len([
                    item for item in execution_batches
                    if item["kind"] == "mixture"
                ]) + 1,
                "stream_labels": [context["name"] for context in group],
                **batch_metadata,
            })
            for context, estimates in zip(group, batch_outputs):
                print(
                    f"scoring mixture {len(track_results) + 1}/{len(contexts)}: "
                    f"{context['name']}",
                    flush=True,
                )
                mixtures = [
                    _read_excerpt(
                        context["mixture_path"], start, end,
                        expected_frames=context["frames"],
                    )
                    for start, end in context["reference_intervals"]
                ]
                references = [
                    np.stack([
                        _read_excerpt(
                            path, start, end, expected_frames=context["frames"]
                        )
                        for path in context["stem_paths"]
                    ])
                    for start, end in context["reference_intervals"]
                ]
                track_results.append(_score_track(
                    context["name"],
                    context["intervals"],
                    mixtures,
                    references,
                    estimates,
                    metric_config,
                ))
            del batch_outputs

        result_by_name = {result["name"]: result for result in track_results}
        probe_contexts = [
            context for context in contexts if context["name"] in probe_track_set
        ]
        probe_tracks_per_batch = max(
            1, _MAX_EVALUATION_BATCH_STREAMS // len(SOURCE_ORDER)
        )
        for group_start in range(0, len(probe_contexts), probe_tracks_per_batch):
            group = probe_contexts[group_start:group_start + probe_tracks_per_batch]
            paths = [
                context["stem_paths"][source]
                for context in group
                for source in range(len(SOURCE_ORDER))
            ]
            intervals = [
                context["estimate_intervals"]
                for context in group
                for _ in SOURCE_ORDER
            ]
            frames = [
                context["frames"]
                for context in group
                for _ in SOURCE_ORDER
            ]
            print(
                f"streaming isolated-source batch "
                f"{group_start // probe_tracks_per_batch + 1}: B={len(paths)}",
                flush=True,
            )
            batch_outputs, batch_metadata = _stream_audio_batch(
                model,
                paths,
                intervals,
                device=device,
                expected_frames=frames,
            )
            execution_batches.append({
                "kind": "isolated_source_probe",
                "batch_index": len([
                    item for item in execution_batches
                    if item["kind"] == "isolated_source_probe"
                ]) + 1,
                "stream_labels": [
                    f"{context['name']}::{SOURCE_ORDER[source]}"
                    for context in group
                    for source in range(len(SOURCE_ORDER))
                ],
                **batch_metadata,
            })
            for context_index, context in enumerate(group):
                offset = context_index * len(SOURCE_ORDER)
                probe_outputs = batch_outputs[
                    offset:offset + len(SOURCE_ORDER)
                ]
                references = [
                    np.stack([
                        _read_excerpt(
                            path, start, end, expected_frames=context["frames"]
                        )
                        for path in context["stem_paths"]
                    ])
                    for start, end in context["reference_intervals"]
                ]
                result_by_name[context["name"]]["single_source"] = (
                    _score_single_source_track(
                        probe_outputs, references, metric_config
                    )
                )
            del batch_outputs

    aggregate = _aggregate_tracks(track_results)
    probed_tracks = tuple(sorted(
        track["name"] for track in track_results if "single_source" in track
    ))
    probe_metrics_complete = (
        None
        if sealed_test
        else _probe_metrics_complete(aggregate, configured_probe_tracks)
    )
    contract_complete = _contract_is_complete(
        sealed_test=sealed_test,
        selected_track_count=len(selected_tracks),
        manifest_track_count=len(manifest["tracks"]),
        max_tracks=args.max_tracks,
        probes_disabled=args.disable_single_source_probes,
        configured_probe_tracks=configured_probe_tracks,
        probed_tracks=probed_tracks,
        probe_metrics_complete=probe_metrics_complete,
    )
    decision_inputs_required = bool(
        not sealed_test
        and args.max_tracks is None
        and len(selected_tracks) == len(manifest["tracks"])
        and not args.disable_single_source_probes
        and configured_probe_tracks
    )
    if decision_inputs_required:
        decision_details = validate_decision_inputs(aggregate)
        decision_input_validation: dict[str, Any] = {
            "schema_version": 1,
            "required": True,
            "pass": True,
            "details": decision_details,
        }
    else:
        decision_input_validation = {
            "schema_version": 1,
            "required": False,
            "pass": None,
            "details": None,
        }
    actual_callbacks = sum(item["callback_count"] for item in execution_batches)
    scalar_callbacks = sum(
        item["scalar_equivalent_callback_count"] for item in execution_batches
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_peak_allocated_bytes: int | None = int(
            torch.cuda.max_memory_allocated(device)
        )
        cuda_peak_reserved_bytes: int | None = int(
            torch.cuda.max_memory_reserved(device)
        )
    else:
        cuda_peak_allocated_bytes = None
        cuda_peak_reserved_bytes = None
    memory = {
        "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    elapsed = time.monotonic() - started
    return {
        "schema_version": 1,
        "status": "ok",
        "contract_complete": contract_complete,
        "evaluation_split": manifest["split"],
        "sealed_test": sealed_test,
        "probe_metrics_complete": probe_metrics_complete,
        "source_order": list(SOURCE_ORDER),
        "single_source_probe_tracks": list(configured_probe_tracks),
        "probed_single_source_tracks": list(probed_tracks),
        "alignment_samples": int(config["alignment_samples"]),
        "metric_config": metric_config.to_dict(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_file_sha256": manifest_file_hash,
        "manifest_content_sha256": manifest["content_sha256"],
        "config": str(args.config.expanduser().resolve()),
        "config_file_sha256": config_file_hash,
        "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        "metrics_sha256": _sha256_file(Path(__file__).with_name("metrics.py").resolve()),
        "evaluation_schema_sha256": _sha256_file(
            Path(__file__).with_name("evaluation_schema.py").resolve()
        ),
        "device": str(device),
        "backend": backend,
        "memory": memory,
        "streaming_parity": streaming_parity,
        "decision_input_validation": decision_input_validation,
        "execution": {
            "mode": "legacy_unbatched" if legacy_unbatched else "batched",
            "causal_state_scope": "independent_per_stream",
            "production_streaming_api_modified": False,
            "max_batch_streams": 1 if legacy_unbatched else _MAX_EVALUATION_BATCH_STREAMS,
            "actual_model_callback_count": actual_callbacks,
            "scalar_equivalent_callback_count": scalar_callbacks,
            "model_callback_reduction": (
                float(scalar_callbacks / actual_callbacks)
                if actual_callbacks > 0 else None
            ),
            "batches": execution_batches,
        },
        "deployment": deployment,
        "num_parameters": int(model.num_parameters),
        "evaluated_track_count": len(track_results),
        "evaluation_seconds": elapsed,
        "aggregate": aggregate,
        "tracks": track_results,
    }


def _write_result(path: Path, result: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _summary(result: Mapping[str, Any]) -> None:
    aggregate = result["aggregate"]
    print("---")
    print(f"val_score:                 {aggregate['val_score']:.6f}")
    print(f"full_sdr_db:               {aggregate['full_sdr_db']:.6f}")
    for source in SOURCE_ORDER:
        value = aggregate["per_stem"][source]["full_sdr_db"]
        print(f"sdr_{source}:".ljust(27) + f"{value:.6f}")
    print(f"low_sdr_db:                {aggregate['low_sdr_db']:.6f}")
    print(f"bleed_sir_db:              {aggregate['bleed_sir_db']:.6f}")
    print(f"mixture_consistency_db:    {aggregate['mixture_consistency_db']:.6f}")
    print(f"num_params_m:              {result['num_parameters'] / 1e6:.6f}")
    print(f"evaluated_tracks:          {result['evaluated_track_count']}")
    print(f"evaluation_seconds:        {result['evaluation_seconds']:.3f}")
    print(f"evaluation_complete:       {int(result['contract_complete'])}")


def _self_test() -> None:
    class FakeModel:
        overlap_len = 8

        def __init__(self):
            self.callback_count = 0

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(()))

        def init_stateful_transform_fn(self, **_: Any):
            def transform(chunk: torch.Tensor) -> torch.Tensor:
                self.callback_count += 1
                return torch.stack((chunk, torch.zeros_like(chunk),
                                    torch.zeros_like(chunk), torch.zeros_like(chunk)))
            return transform

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "audio.wav"
        second_path = Path(temporary) / "audio-second.wav"
        audio = np.linspace(-0.1, 0.1, 128, dtype=np.float32).reshape(64, 2)
        second_audio = np.linspace(0.2, -0.05, 128, dtype=np.float32).reshape(64, 2)
        sf.write(path, audio, 44_100, subtype="FLOAT")
        sf.write(second_path, second_audio, 44_100, subtype="FLOAT")
        model = FakeModel()
        captured = _stream_audio(
            model, path, [(3, 17)], device=torch.device("cpu"),
            expected_frames=64
        )
        assert len(captured) == 1 and captured[0].shape == (4, 2, 14)
        np.testing.assert_allclose(captured[0][0], audio[3:17].T)
        assert np.count_nonzero(captured[0][1:]) == 0
        assert model.callback_count == 3  # ceil(max capture end / hop), not 8

        class FakeCausalModel:
            overlap_len = 8
            audio_channels = 2
            num_sources = 4

            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.zeros(()))

            def parameters(self):
                yield self.parameter

            def eval(self):
                return self

            def forward(
                self,
                audio_chunk: torch.Tensor,
                *,
                hiddens: torch.Tensor | None,
                return_reduced_sources: None,
                is_streaming: bool,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                assert return_reduced_sources is None and is_streaming
                if hiddens is None:
                    hiddens = torch.zeros(
                        (audio_chunk.shape[0], 1, 1),
                        dtype=audio_chunk.dtype,
                        device=audio_chunk.device,
                    )
                stateful = audio_chunk + hiddens
                transformed = torch.stack(
                    (stateful, stateful * 0.5, -stateful, stateful * 0.25),
                    dim=1,
                )
                next_hiddens = audio_chunk[..., -1:].mean(dim=1, keepdim=True)
                return transformed, next_hiddens

            def init_stateful_transform_fn(self, **_: Any):
                past_audio = torch.zeros((self.audio_channels, self.overlap_len))
                hiddens = None
                overlap_add_buffer = None

                def transform(audio_chunk: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
                    nonlocal past_audio, hiddens, overlap_add_buffer
                    numpy_input = isinstance(audio_chunk, np.ndarray)
                    if numpy_input:
                        audio_chunk = torch.from_numpy(audio_chunk)
                    full_chunk = torch.cat((past_audio, audio_chunk), dim=-1)[None]
                    transformed, hiddens = self.forward(
                        full_chunk,
                        hiddens=hiddens,
                        return_reduced_sources=None,
                        is_streaming=True,
                    )
                    transformed = transformed[0]
                    if overlap_add_buffer is None:
                        overlap_add_buffer = torch.zeros_like(transformed)
                    overlap_add_buffer += transformed
                    output = overlap_add_buffer[..., :self.overlap_len].clone()
                    overlap_add_buffer = torch.cat((
                        overlap_add_buffer[..., self.overlap_len:],
                        torch.zeros_like(output),
                    ), dim=-1)
                    past_audio = audio_chunk
                    return output.numpy() if numpy_input else output

                return transform

        first_intervals = [(3, 17), (24, 39)]
        second_intervals = [(5, 31)]
        first_legacy = _stream_audio(
            FakeCausalModel(), path, first_intervals,
            device=torch.device("cpu"), expected_frames=64
        )
        second_legacy = _stream_audio(
            FakeCausalModel(), second_path, second_intervals,
            device=torch.device("cpu"), expected_frames=64
        )
        batched, batched_metadata = _stream_audio_batch(
            FakeCausalModel(),
            [path, second_path],
            [first_intervals, second_intervals],
            device=torch.device("cpu"),
            expected_frames=[64, 64],
            io_block_hops=2,
        )
        for actual, expected in zip(batched[0], first_legacy):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(batched[1], second_legacy):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        assert batched_metadata == {
            "batch_size": 2,
            "callback_count": 5,
            "scalar_equivalent_callback_count": 9,
            "io_block_hops": 2,
        }

        parity_batches = [{
            "kind": "mixture",
            "batch_index": 1,
            "stream_labels": ["first", "second"],
            "audio_paths": [path, second_path],
            "expected_frames": [64, 64],
        }]
        parity_expected = _capture_batched_parity_prefixes(
            FakeCausalModel(),
            parity_batches,
            device=torch.device("cpu"),
            callback_count=4,
        )
        parity_records = _compare_production_parity_prefixes(
            FakeCausalModel(),
            parity_batches,
            parity_expected,
            device=torch.device("cpu"),
            callback_count=4,
        )
        assert parity_records[0]["pass"] is True
        assert parity_records[0]["batch_size"] == 2
        assert parity_records[0]["callback_count"] == 4
        assert parity_records[0]["max_abs_error"] == 0.0

        class BadProductionModel(FakeCausalModel):
            def init_stateful_transform_fn(self, **kwargs: Any):
                transform = super().init_stateful_transform_fn(**kwargs)

                def shifted(audio_chunk: torch.Tensor | np.ndarray):
                    return transform(audio_chunk) + np.float32(0.01)

                return shifted

        try:
            _compare_production_parity_prefixes(
                BadProductionModel(),
                parity_batches,
                parity_expected,
                device=torch.device("cpu"),
                callback_count=4,
            )
        except RuntimeError as error:
            assert "parity failed" in str(error)
        else:
            raise AssertionError("divergent production wrapper passed parity")

        class BatchCoupledModel(FakeCausalModel):
            def forward(self, audio_chunk: torch.Tensor, **kwargs: Any):
                transformed, hidden = super().forward(audio_chunk, **kwargs)
                coupled = audio_chunk.mean(dim=0, keepdim=True)
                return transformed + coupled[:, None], hidden

        coupled_expected = _capture_batched_parity_prefixes(
            BatchCoupledModel(),
            parity_batches,
            device=torch.device("cpu"),
            callback_count=4,
        )
        try:
            _compare_production_parity_prefixes(
                BatchCoupledModel(),
                parity_batches,
                coupled_expected,
                device=torch.device("cpu"),
                callback_count=4,
            )
        except RuntimeError as error:
            assert "parity failed" in str(error)
        else:
            raise AssertionError("batch-coupled model passed production parity")

    manifest = {"tracks": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
    assert _configured_probe_tracks(
        {"single_source_probe_tracks": ["A", "C"]}, manifest
    ) == ("A", "C")
    assert _configured_probe_tracks({"single_source_probes": False}, manifest) == ()
    try:
        _configured_probe_tracks(
            {"single_source_probe_tracks": ["C", "A"]}, manifest
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsorted probe track list was accepted")

    metric_config = MetricConfig(
        sample_rate=8_000,
        window_samples=8_000,
        hop_samples=8_000,
        bands_hz={
            "low_20_250": (20.0, 250.0),
            "low_20_80": (20.0, 80.0),
            "low_80_250": (80.0, 250.0),
            "low_250_500": (250.0, 500.0),
        },
    )
    time_axis = np.arange(8_000, dtype=np.float64) / metric_config.sample_rate
    active_references = np.stack([
        np.stack([0.1 * np.sin(2 * np.pi * frequency * time_axis)] * 2)
        for frequency in (50.0, 80.0, 110.0, 150.0)
    ])
    partly_silent_references = active_references.copy()
    partly_silent_references[0] = 0.0
    references = [partly_silent_references, active_references]
    probe_outputs = []
    for input_source in range(len(SOURCE_ORDER)):
        source_outputs = []
        for excerpt_index, refs in enumerate(references):
            outputs = np.stack([0.1 * refs[input_source]] * len(SOURCE_ORDER))
            outputs[input_source] = refs[input_source]
            if input_source == 0 and excerpt_index == 0:
                outputs[0].fill(0.01)
                outputs[1:].fill(0.001)
            source_outputs.append(outputs)
        probe_outputs.append(source_outputs)
    probe = _score_single_source_track(
        probe_outputs, references, metric_config
    )
    assert probe["input_activity"]["drums"]["active_excerpts"] == 1
    assert probe["input_activity"]["drums"]["silent_excerpts"] == 1
    assert abs(probe["output_to_input_db"][0][0]) < 1e-9
    assert abs(probe["silent_input_fp_dbfs"][0][0] - (-40.0)) < 1e-9
    assert probe["silent_input_fp_dbfs"][1][0] is None
    synthetic_track = {
        "name": "Synthetic",
        "full_sdr_db": 0.0,
        "low_sdr_db": 0.0,
        "bleed_sir_db": 0.0,
        "band_sdr_db": {name: 0.0 for name in metric_config.bands_hz},
        "per_stem": {
            source: {
                "full_sdr_db": 0.0,
                "band_sdr_db": {
                    name: 0.0 for name in metric_config.bands_hz
                },
                "sir_db": 0.0,
                "absent_fp_dbfs": -60.0,
                "absent_fp_ratio_db": -60.0,
            }
            for source in SOURCE_ORDER
        },
        "projection_attribution_db": [[0.0] * 4 for _ in range(4)],
        "mixture_consistency_db": 0.0,
        "mixture_consistency_error_rms": 0.0,
        "single_source": probe,
    }
    synthetic_aggregate = _aggregate_tracks([synthetic_track])
    assert _probe_metrics_complete(synthetic_aggregate, ("Synthetic",))
    decision_details = validate_decision_inputs(synthetic_aggregate)
    assert decision_details["guardrail_count"] == 40
    assert decision_details["required_projection_matrix_finite_cells"] == 16
    assert decision_details["required_active_matrix_finite_cells"] == 32
    assert decision_details["required_desired_metric_finite_values"] == 12
    nullable_silent = copy.deepcopy(synthetic_aggregate)
    nullable_silent["single_source"]["silent_input_fp_dbfs"][0][0] = None
    assert validate_decision_inputs(nullable_silent)["silent_matrix_null_cells"] >= 1
    for malformed in (
        ("missing projection", lambda item: item.pop("projection_attribution_db")),
        (
            "null projection",
            lambda item: item["projection_attribution_db"][0].__setitem__(1, None),
        ),
        (
            "null active probe",
            lambda item: item["single_source"]["output_to_input_db"][0].__setitem__(
                1, None
            ),
        ),
        (
            "null absent-target ratio",
            lambda item: item["per_stem"]["drums"].__setitem__(
                "absent_fp_ratio_db", None
            ),
        ),
    ):
        malformed_aggregate = copy.deepcopy(synthetic_aggregate)
        malformed[1](malformed_aggregate)
        try:
            validate_decision_inputs(malformed_aggregate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{malformed[0]} passed decision-input validation")
    assert all(
        _finite_metric(value)
        for value in synthetic_aggregate["single_source"][
            "silent_input_fp_dbfs_by_output"
        ]
    )

    def fake_manifest(split: str) -> dict[str, Any]:
        sealed = split == "test"
        count = 46 if sealed else 14
        payload: dict[str, Any] = {
            "schema_version": 1,
            "dataset": "MUSDB18-HQ",
            "root": "/unused",
            "split": split,
            "disk_subset": "test" if sealed else "train",
            "musdb_is_wav": True,
            "sample_rate": 44_100,
            "channels": 2,
            "source_order": list(SOURCE_ORDER),
            "track_count": count,
            "expected_local_track_count": count,
            "official_track_count": 50 if sealed else 14,
            "missing_from_official_count": 4 if sealed else 0,
            "tracks": [
                {
                    "name": f"Track {index:02d}",
                    "sample_rate": 44_100,
                    "channels": 2,
                    "stems": {source: f"{source}.wav" for source in SOURCE_ORDER},
                    "mixture_sum": {
                        "checked": True,
                        "max_abs_error": 0.0,
                        "max_abs_tolerance": 0.08,
                        "rms_error": 0.0,
                        "rms_tolerance": 0.00015,
                    },
                }
                for index in range(count)
            ],
        }
        payload["content_sha256"] = _canonical_hash(payload)
        return payload

    validation_manifest = fake_manifest("valid")
    test_manifest = fake_manifest("test")
    _validate_manifest(validation_manifest)
    _validate_manifest(test_manifest, allow_sealed_test=True)
    try:
        _validate_manifest(test_manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("sealed test manifest was accepted without opt-in")
    try:
        _validate_manifest(validation_manifest, allow_sealed_test=True)
    except ValueError:
        pass
    else:
        raise AssertionError("validation manifest was accepted in sealed-test mode")
    assert _contract_is_complete(
        sealed_test=True,
        selected_track_count=46,
        manifest_track_count=46,
        max_tracks=None,
        probes_disabled=False,
        configured_probe_tracks=(),
        probed_tracks=(),
        probe_metrics_complete=None,
    )
    assert not _contract_is_complete(
        sealed_test=True,
        selected_track_count=46,
        manifest_track_count=46,
        max_tracks=46,
        probes_disabled=False,
        configured_probe_tracks=(),
        probed_tracks=(),
        probe_metrics_complete=None,
    )
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == _CUBLAS_WORKSPACE_CONFIG
    print(json.dumps({"self_test": "pass"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-sealed-test",
        action="store_true",
        help="explicit final-only mode; accepts the frozen 46-track test schema",
    )
    parser.add_argument("--max-tracks", type=int,
                        help="smoke only; marks evaluation incomplete")
    parser.add_argument("--disable-single-source-probes", action="store_true",
                        help="smoke only; marks evaluation incomplete")
    parser.add_argument(
        "--legacy-unbatched",
        action="store_true",
        help="debug/equivalence mode: retain sequential production B=1 streaming",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    missing = [name for name in ("checkpoint", "manifest", "config", "output")
               if getattr(args, name) is None]
    if missing:
        parser.error("required arguments: " + ", ".join(f"--{name}" for name in missing))
    result = evaluate(args)
    _write_result(args.output.expanduser(), result, args.overwrite)
    _summary(result)


if __name__ == "__main__":
    main()
