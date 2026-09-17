"""In-memory CPU1 music evaluation for the isolated OLA512 prototype.

The caller supplies an already constructed model and explicit provenance.
Nothing imports Torch, loads a model/checkpoint, exports audio, or writes a
report on import. This module has no CLI or training entrypoint. Runtime uses
the existing panel, physical reference intervals, metrics, and aggregation,
with a separate real-input hop256 streamer. Interior captures never zero-flush.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "research/manifests/valid.json"
DEFAULT_CONFIG = ROOT / "research/eval_config.json"
HOP = 256
SOURCE_ORDER = ("drums", "bass", "vocals", "other")
OUTPUT_POLICY = "float32 Other = aligned mixture - sum(Drums,Bass,Vocals)"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OLAStreamPlan:
    expected_frames: int
    reference_intervals: tuple[tuple[int, int], ...]
    capture_intervals: tuple[tuple[int, int], ...]
    receive_end: int
    call_slices: tuple[tuple[int, int], ...]
    unroll_hops: int
    io_block_hops: int

    @property
    def literal_hop_count(self) -> int:
        return self.receive_end // HOP


def plan_ola512_stream(
    intervals: Sequence[Mapping[str, Any]],
    expected_frames: int,
    *,
    unroll_hops: int = 64,
    io_block_hops: int = 64,
) -> OLAStreamPlan:
    """Plan continuous real input and capture on the received-output timeline.

    A render at input position p emits physical p-256 onward, but Capture.add
    is addressed at p. Therefore each capture row must equal reference+256.
    The last group is shortened to avoid unnecessary complete-group padding.

    This music API requires real audio through the complete final callback.
    Captures too close to EOF are rejected instead of silently fabricating
    future input. Arbitrary EOF/flush evaluation is outside this API's scope.
    """
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    if (type(unroll_hops) is not int or unroll_hops <= 0
            or type(io_block_hops) is not int or io_block_hops <= 0):
        raise ValueError("Unroll and I/O block sizes must be positive hop counts")
    if not intervals:
        raise ValueError("At least one fixed interval is required")
    references, captures = [], []
    for row in intervals:
        if not isinstance(row, Mapping):
            raise ValueError("Each interval must be a mapping of sample coordinates")
        fields = tuple(row.get(key) for key in
                       ("reference_start", "reference_end", "estimate_start", "estimate_end"))
        if any(type(value) is not int for value in fields):
            raise ValueError("Interval coordinates must be integer sample indices")
        ref_start, ref_end, est_start, est_end = fields
        if not (0 <= ref_start < ref_end <= expected_frames):
            raise ValueError("Physical reference interval is outside the track")
        if (est_start, est_end) != (ref_start + HOP, ref_end + HOP):
            raise ValueError("OLA capture indices must equal physical reference indices plus 256")
        references.append((ref_start, ref_end))
        captures.append((est_start, est_end))
    ordered = sorted(references)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Physical reference intervals must not overlap")
    receive_end = ((max(end for _, end in captures) + HOP - 1) // HOP) * HOP
    if receive_end > expected_frames:
        raise ValueError("Capture lacks real input through its final 256-sample callback; EOF padding is not supported")
    group_samples = unroll_hops * HOP
    calls = tuple((start, min(start + group_samples, receive_end))
                  for start in range(0, receive_end, group_samples))
    return OLAStreamPlan(expected_frames, tuple(references), tuple(captures),
                         receive_end, calls, unroll_hops, io_block_hops)


def ola512_stream_metadata(plan: OLAStreamPlan) -> dict[str, Any]:
    """Describe actual hop256 work without inheriting hop512 stream fields."""
    return {
        "batch_size": 1,
        "hop_samples": HOP,
        "graph_alignment_samples": HOP,
        "capture_coordinate_system": "received-output indices = physical reference indices + 256",
        "reference_intervals": [list(row) for row in plan.reference_intervals],
        "capture_intervals": [list(row) for row in plan.capture_intervals],
        "receive_interval": [0, plan.receive_end],
        "physical_emitted_interval": [-HOP, plan.receive_end - HOP],
        "real_input_samples": plan.receive_end,
        "zero_input_samples": 0,
        "flush_hops": 0,
        "callback_count": plan.literal_hop_count,
        "scalar_equivalent_callback_count": plan.literal_hop_count,
        "processed_hops_per_stream": plan.literal_hop_count,
        "forward_call_count": len(plan.call_slices),
        "unroll_hops": plan.unroll_hops,
        "final_call_hops": (plan.call_slices[-1][1] - plan.call_slices[-1][0]) // HOP,
        "io_block_hops": plan.io_block_hops,
        "io_block_samples": plan.io_block_hops * HOP,
        "reader_hop_samples": HOP,
        "state": "reset once per track at sample zero; continuous through all prefixes and gaps",
        "interior_future_context": "genuine track samples through the full final callback",
        "host_queue_samples": HOP,
        "host_queue_implemented": False,
        "intended_total_latency_samples": 2 * HOP,
        "host_qualified": False,
        "host_latency_or_callback_timing_qualified": False,
    }


def validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Require an honest JSON identity; untrained models need no checkpoint."""
    if not isinstance(identity, Mapping):
        raise ValueError("Caller identity must be a mapping")
    value = json.loads(json.dumps(dict(identity), allow_nan=False))
    if not isinstance(value.get("label"), str) or not value["label"].strip():
        raise ValueError("Caller identity requires a nonempty label")
    if value.get("state_kind") not in {"untrained_initialization", "trained_in_memory", "checkpoint"}:
        raise ValueError("Declare untrained_initialization, trained_in_memory, or checkpoint state_kind")
    updates = value.get("training_updates")
    if type(updates) is not int or updates < 0:
        raise ValueError("Caller identity requires a nonnegative integer training_updates")
    if not isinstance(value.get("provenance"), dict) or not value["provenance"]:
        raise ValueError("Caller identity requires explicit nonempty provenance")
    checkpoint = value.get("checkpoint")
    if value["state_kind"] == "untrained_initialization" and (updates != 0 or checkpoint is not None):
        raise ValueError("An untrained in-memory initializer has zero updates and no checkpoint")
    if value["state_kind"] == "trained_in_memory" and (updates == 0 or checkpoint is not None):
        raise ValueError("A trained in-memory identity has positive updates and no checkpoint")
    if value["state_kind"] == "checkpoint":
        if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("path"), str)
                or not checkpoint["path"] or not isinstance(checkpoint.get("sha256"), str)
                or len(checkpoint["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint["sha256"])):
            raise ValueError("A checkpoint identity requires a path and lowercase SHA-256")
    return value


def _require_cpu1_model(model) -> None:
    import torch

    if (getattr(model, "hop_samples", None) != HOP
            or getattr(model, "graph_alignment_samples", None) != HOP
            or getattr(model, "flush_hops", None) != 1):
        raise ValueError("Evaluation requires the OLA512 hop256/delay256 model contract")
    if any(module.training for module in model.modules()):
        raise ValueError("Caller must place the complete model in eval mode")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("This bounded evaluation API requires CPU1 intra-op and inter-op threads")
    if torch.cuda.is_initialized():
        raise ValueError("This evaluation must run in a CPU-only process without initialized CUDA")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("The supplied model must remain on CPU in float32")
    gains = model.output_source_scales
    if gains.shape != (4,) or not torch.isfinite(gains).all() or not (gains > 0).all():
        raise ValueError("The model must supply four finite positive native source gains")


def model_state_sha256(model) -> str:
    """Hash actual in-memory CPU state without serializing a checkpoint."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stream_ola512_track(model, audio_path: Path, plan: OLAStreamPlan):
    """Return raw4 and delayed-mixture captures for one continuous real track.

    The legacy reader serves literal 256-sample hops, assembled into at most
    ``unroll_hops`` per render. Every group is full real audio, including the
    final shortened group. No caller state is accepted or retained afterward.
    """
    import numpy as np
    import torch
    from research import evaluate as legacy

    _require_cpu1_model(model)
    raw_capture = legacy._Capture(plan.capture_intervals, (4, 2))
    mixture_capture = legacy._Capture(plan.capture_intervals, (2,))
    readers = legacy._open_blocked_readers([Path(audio_path)], [plan.expected_frames],
                                          hop=HOP, block_hops=plan.io_block_hops)
    reader = readers[0]
    calls, literal_reads, cursor = 0, 0, 0
    try:
        with torch.inference_mode():
            state = model.initial_state(1, device=torch.device("cpu"))
            for start, stop in plan.call_slices:
                if start != cursor:
                    raise RuntimeError("Stream calls must be consecutive and nonoverlapping")
                count = stop - start
                staging = np.empty((1, 2, count), dtype=np.float32)
                for offset in range(0, count, HOP):
                    block = reader.read_hop()
                    if block.shape != (HOP, 2) or not np.isfinite(block).all():
                        raise RuntimeError(f"Real stereo input ended early or is non-finite: {audio_path}")
                    staging[0, :, offset:offset + HOP] = block.T
                    literal_reads += 1
                result = model.render(torch.from_numpy(staging), state)
                state = result.state
                if (tuple(result.raw.shape) != (1, 4, 2, count)
                        or tuple(result.delayed_mixture.shape) != (1, 2, count)
                        or result.raw.dtype != torch.float32
                        or result.delayed_mixture.dtype != torch.float32):
                    raise RuntimeError("OLA render output differs from its declared shape/precision")
                raw = result.raw[0].numpy()
                delayed = result.delayed_mixture[0].numpy()
                if not np.isfinite(raw).all() or not np.isfinite(delayed).all():
                    raise FloatingPointError("Non-finite OLA stream output")
                # Coordinates remain the received-output timeline. The single
                # +256 offset is already present in plan.capture_intervals.
                raw_capture.add(start, raw)
                mixture_capture.add(start, delayed)
                cursor = stop
                calls += 1
    finally:
        reader.close()
    if (cursor != plan.receive_end or calls != len(plan.call_slices)
            or literal_reads != plan.literal_hop_count):
        raise RuntimeError("Executed stream differs from its declared OLA plan")
    metadata = ola512_stream_metadata(plan)
    metadata.update(actual_forward_call_count=calls, actual_literal_read_count=literal_reads,
                    actual_received_samples=cursor, coverage_complete=True)
    return raw_capture.finish(), mixture_capture.finish(), metadata


def evaluate_ola512_music(
    model,
    *,
    identity: Mapping[str, Any],
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    panel: str = "full",
    track_indices: Sequence[int] | None = None,
    excerpt_starts: Sequence[float] | None = None,
    duration: float = 15.0,
    batch_size: int = 1,
    unroll_hops: int = 64,
    io_block_hops: int = 64,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Score a caller-supplied model and return JSON-compatible metrics only.

    Defaults match the existing 14-track/28-excerpt physical music panel. The
    caller owns model construction, CPU/thread setup, report persistence, and
    execution authorization. No source gains or model modes are changed here.
    ``checkpoint=None`` is explicit for in-memory initializers; their identity
    lives under ``model`` instead of pretending that a checkpoint was loaded.
    """
    import numpy as np
    import torch
    from research.direct import evaluate as shared
    from research import evaluate as legacy
    from research.metrics import MetricConfig

    if type(batch_size) is not int or batch_size != 1:
        raise ValueError("This bounded OLA music evaluator supports batch_size=1 only")
    if panel not in {"dev", "full"}:
        raise ValueError("panel must be dev or full")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be a callable or None")
    declared_identity = validate_identity(identity)
    _require_cpu1_model(model)
    started = time.monotonic()
    manifest_path, config_path = Path(manifest_path).resolve(), Path(config_path).resolve()
    source_paths = (Path(__file__).resolve(), ROOT / "research/direct/latency_ola512.py",
                    Path(shared.__file__), Path(legacy.__file__), ROOT / "research/metrics.py",
                    manifest_path, config_path)
    bindings_before = {str(path): file_sha256(path) for path in source_paths}
    checkpoint = declared_identity.get("checkpoint")
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint["path"]).resolve()
        if file_sha256(checkpoint_path) != checkpoint["sha256"]:
            raise ValueError("Supplied checkpoint identity does not match its file")
        bindings_before[str(checkpoint_path)] = checkpoint["sha256"]
    manifest = json.loads(manifest_path.read_text())
    source_config = json.loads(config_path.read_text())
    tracks, config = shared.select_panel(
        manifest, source_config, panel=panel, track_indices=track_indices,
        excerpt_starts=excerpt_starts, duration=duration, alignment_samples=HOP,
    )
    metric_config = MetricConfig.from_mapping(config["metrics"])
    if metric_config.sample_rate != 44_100:
        raise ValueError("Metric sample rate must match the 44100 Hz panel")
    intervals = [legacy._reference_intervals(track, config) for track in tracks]
    plans = [plan_ola512_stream(rows, int(track["frames"]), unroll_hops=unroll_hops,
                               io_block_hops=io_block_hops)
             for track, rows in zip(tracks, intervals, strict=True)]
    state_before = model_state_sha256(model)
    if (declared_identity.get("model_state_sha256") is not None
            and declared_identity["model_state_sha256"] != state_before):
        raise ValueError("Supplied model-state identity does not match actual in-memory tensors")
    gains_before = model.output_source_scales.detach().clone()
    rng_before = torch.get_rng_state().clone()
    root = Path(manifest["root"])
    scores, stream_batches = [], []
    reconstruction_max_abs = 0.0
    for track, rows, plan in zip(tracks, intervals, plans, strict=True):
        raw_outputs, delayed_mixtures, stream_info = stream_ola512_track(
            model, legacy._safe_dataset_path(root, track["mixture"]), plan,
        )
        stream_info["track"] = track["name"]
        stream_batches.append(stream_info)

        def read(relative, row):
            return legacy._read_excerpt(
                legacy._safe_dataset_path(root, relative),
                int(row["reference_start"]), int(row["reference_end"]),
                expected_frames=int(track["frames"]),
            )

        mixtures = [read(track["mixture"], row) for row in rows]
        references = [np.stack([read(track["stems"][source], row) for source in SOURCE_ORDER])
                      for row in rows]
        for delayed, physical in zip(delayed_mixtures, mixtures, strict=True):
            if not np.array_equal(delayed, physical.astype(np.float32)):
                raise RuntimeError("Captured delayed mixture differs from exact physical reference audio")
        estimates = [shared.shipping_residual(raw, mixture)
                     for raw, mixture in zip(raw_outputs, mixtures, strict=True)]
        for estimate, mixture in zip(estimates, mixtures, strict=True):
            reconstruction_max_abs = max(reconstruction_max_abs, float(np.max(np.abs(
                estimate.sum(axis=0, dtype=np.float32) - mixture.astype(np.float32),
            ))))
        # Preserve legacy float64 physical references and its within-track
        # active-window weighting before equal-track/equal-stem aggregation.
        score = legacy._score_track(track["name"], rows, mixtures, references, estimates, metric_config)
        scores.append(score)
        stream_info["physical_alignment_verified_by_delayed_mixture"] = True
        if progress is not None:
            progress({"event": "track", "model": declared_identity["label"],
                      "track": track["name"], "full_sdr_db": score["full_sdr_db"]})
        del raw_outputs, delayed_mixtures, mixtures, references, estimates
        del delayed, physical, estimate, mixture
    aggregate = legacy._aggregate_tracks(scores)
    state_after = model_state_sha256(model)
    bindings_after = {path: file_sha256(Path(path)) for path in bindings_before}
    if state_before != state_after or not torch.equal(gains_before, model.output_source_scales):
        raise RuntimeError("Evaluation changed model tensors or native source gains")
    if bindings_before != bindings_after:
        raise RuntimeError("An evaluation source or protocol file changed during execution")
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError("Evaluation unexpectedly changed the CPU RNG state")
    _require_cpu1_model(model)
    model_identity = {
        **declared_identity,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "output_source_scales": gains_before.tolist(),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_provenance": json.loads(json.dumps(model.provenance, allow_nan=False)),
        "architecture": json.loads(json.dumps(model.architecture_metadata, allow_nan=False)),
    }
    result = {
        "model": model_identity,
        "checkpoint": checkpoint,
        "aggregate": aggregate,
        "tracks": scores,
        "stream_batches": stream_batches,
        "reconstruction_max_abs": reconstruction_max_abs,
        "evaluation_seconds": time.monotonic() - started,
        "audio_dir": None,
    }
    return {
        "schema_version": 1,
        "evaluator_kind": "in_memory_ola512_music_cpu1",
        "panel": panel if track_indices is None and excerpt_starts is None and duration == 15.0
                 else "custom-development",
        "manifest": str(manifest_path),
        "manifest_sha256": bindings_before[str(manifest_path)],
        "config": str(config_path),
        "config_sha256": bindings_before[str(config_path)],
        "track_names": [track["name"] for track in tracks],
        "excerpts": config["default_excerpts"],
        "excerpt_count": sum(len(rows) for rows in intervals),
        "alignment_samples": HOP,
        "graph_alignment_samples": HOP,
        "intended_total_latency_samples": 2 * HOP,
        "host_queue_samples": HOP,
        "host_queue_implemented": False,
        "host_qualified": False,
        "streaming_state": "independent per track, continuous from sample zero",
        "output_policy": OUTPUT_POLICY,
        "metrics": metric_config.to_dict(),
        "metric_source_sha256": bindings_before[str(ROOT / "research/metrics.py")],
        "evaluator_sha256": bindings_before[str(Path(__file__).resolve())],
        "source_bindings_before": bindings_before,
        "source_bindings_after": bindings_after,
        "torch_version": torch.__version__,
        "device": "cpu",
        "precision": "float32",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_initialized": False,
        "cpu_rng_unchanged": True,
        "batch_size": 1,
        "unroll_hops": unroll_hops,
        "io_block_hops": io_block_hops,
        "retained_audio": False,
        "checkpoint_written": False,
        "normalization": "none",
        "vocal_gain_override": None,
        "results": [result],
        "limitations": ["Fixed validation music panel; no quality-retention decision is automatic",
                        "Graph alignment is measured separately from the unimplemented host queue",
                        "Grouped CPU evaluation does not qualify callback timing or actual host latency",
                        "This API intentionally rejects captures requiring synthetic EOF padding"],
    }
