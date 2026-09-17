#!/usr/bin/env python3
"""Compare C91 checkpoints using continuous FP32 streaming and shipped Other.

Examples::

    python -m research.direct.evaluate --checkpoint step-100000.pt --output dev.json
    python -m research.direct.evaluate --checkpoint candidate.pt --panel full \
        --output full.json --audio-dir listening/candidate

The default development panel uses valid-track indices 0, 4, 9, 13 and the
15-second excerpt at 30 seconds. ``--panel full`` uses all 14 tracks and both
original excerpts (30 and 75 seconds). Every stream starts at sample zero;
state remains continuous through unscored prefixes and gaps. Metrics are
imported from the existing evaluator, without its experiment-contract runner.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from hs_tasnet import HSTasNet
from research import evaluate as legacy
from research.metrics import MetricConfig, SOURCE_ORDER

DEFAULT_MANIFEST = ROOT / "research/manifests/valid.json"
DEFAULT_CONFIG = ROOT / "research/eval_config.json"
DEV_INDICES = (0, 4, 9, 13)


def load_checkpoint(path: Path, device: torch.device, *, allow_custom_gains=False) -> tuple[Any, dict[str, Any]]:
    """Load the standard local model/config deployment checkpoint strictly."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    if isinstance(config, bytes):
        config = pickle.loads(config)
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a mapping or pickled mapping")
    model = HSTasNet(**config)
    model.load_state_dict(payload["model"], strict=True)
    if (model.audio_channels, model.num_sources, model.overlap_len) != (2, 4, 512):
        raise ValueError("evaluation requires stereo C91 with four sources and hop 512")
    if getattr(model, "causal_current_chunk", False):
        raise ValueError("this evaluator expects C91's previous-input-chunk output")
    expected_scales = model.output_source_scales.new_tensor((0.5, 0.5, 0.4, 0.56))
    if ((not allow_custom_gains and not torch.equal(model.output_source_scales, expected_scales))
            or not model.conv_decode.hann_window_baked):
        raise ValueError(
            "checkpoint is not calibrated C91 deployment weights; export with "
            "research.direct.checkpoints.save_deployment before evaluation"
        )
    if not torch.isfinite(model.output_source_scales).all() or not (model.output_source_scales > 0).all():
        raise ValueError("deployment gains must be finite and positive")
    model = model.to(device=device, dtype=torch.float32).eval()
    return model, {
        "path": str(path.resolve()),
        "sha256": legacy._sha256_file(path),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "output_source_scales": model.output_source_scales.detach().cpu().tolist(),
        "decoder_hann_baked": bool(model.conv_decode.hann_window_baked),
        "step": payload.get("step", payload.get("update")),
    }


def shipping_residual(estimates: np.ndarray, mixture: np.ndarray) -> np.ndarray:
    """Keep DBV unchanged and derive Other from the aligned float32 mixture."""
    result = np.asarray(estimates, dtype=np.float32).copy()
    if result.shape != (4, 2, mixture.shape[-1]) or mixture.shape[0] != 2:
        raise ValueError("expected estimates (4,2,T) and aligned mixture (2,T)")
    result[3] = np.asarray(mixture, dtype=np.float32) - result[:3].sum(
        axis=0, dtype=np.float32
    )
    return result


def select_panel(
    manifest: dict[str, Any], config: dict[str, Any], *, panel: str,
    track_indices: Sequence[int] | None = None,
    excerpt_starts: Sequence[float] | None = None, duration: float = 15.0,
    alignment_samples: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("split") != "valid":
        raise ValueError("checkpoint development evaluation requires the valid split")
    if manifest.get("sample_rate") != 44_100 or manifest.get("channels") != 2:
        raise ValueError("the validation manifest must contain 44.1 kHz stereo audio")
    if tuple(manifest.get("source_order", ())) != SOURCE_ORDER:
        raise ValueError("validation source order differs from C91")
    all_tracks = manifest["tracks"]
    indices = list(track_indices) if track_indices is not None else (
        list(DEV_INDICES) if panel == "dev" else list(range(len(all_tracks)))
    )
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("select at least one track without duplicate indices")
    if any(index < 0 or index >= len(all_tracks) for index in indices):
        raise ValueError("track index is outside the validation manifest")
    starts = list(excerpt_starts) if excerpt_starts is not None else (
        [30.0] if panel == "dev" else [30.0, 75.0]
    )
    if not starts or not np.isfinite(duration) or duration < 1.0:
        raise ValueError("provide excerpt starts and a duration of at least one second")
    selected_config = dict(config)
    selected_config.update({
        "sample_rate": 44_100,
        "alignment_samples": alignment_samples,
        "excerpts": {},
        "default_excerpts": [
            {"id": str(index), "start_seconds": start, "duration_seconds": duration}
            for index, start in enumerate(starts)
        ],
    })
    tracks = [all_tracks[index] for index in indices]
    for track in tracks:
        legacy._reference_intervals(track, selected_config)
    return tracks, selected_config


def export_audio(
    directory: Path, name: str, intervals: Sequence[dict[str, Any]],
    mixtures: Sequence[np.ndarray], references: Sequence[np.ndarray],
    estimates: Sequence[np.ndarray],
) -> None:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    for index, (interval, mixture, refs, ests) in enumerate(
        zip(intervals, mixtures, references, estimates)
    ):
        destination = directory / slug / f"excerpt-{index}"
        destination.mkdir(parents=True, exist_ok=True)
        sf.write(destination / "mixture.wav", mixture.T, 44_100, subtype="FLOAT")
        for source, ref, est in zip(SOURCE_ORDER, refs, ests):
            sf.write(destination / f"reference-{source}.wav", ref.T, 44_100, subtype="FLOAT")
            sf.write(destination / f"estimate-{source}.wav", est.T, 44_100, subtype="FLOAT")
        (destination / "alignment.json").write_text(json.dumps(interval, indent=2) + "\n")


def evaluate_checkpoint(
    checkpoint: Path, *, manifest: dict[str, Any], tracks: list[dict[str, Any]],
    config: dict[str, Any], device: torch.device, batch_size: int = 4,
    audio_dir: Path | None = None, allow_custom_gains: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    model, identity = load_checkpoint(checkpoint, device, allow_custom_gains=allow_custom_gains)
    root = Path(manifest["root"])
    metric_config = MetricConfig.from_mapping(config["metrics"])
    scores = []
    stream_batches = []
    reconstruction_max_abs = 0.0
    for offset in range(0, len(tracks), batch_size):
        batch = tracks[offset:offset + batch_size]
        intervals = [legacy._reference_intervals(track, config) for track in batch]
        outputs, stream_info = legacy._stream_audio_batch(
            model,
            [legacy._safe_dataset_path(root, track["mixture"]) for track in batch],
            [[(int(item["estimate_start"]), int(item["estimate_end"])) for item in rows]
             for rows in intervals],
            device=device, expected_frames=[int(track["frames"]) for track in batch],
        )
        stream_batches.append(stream_info)
        for track, rows, raw in zip(batch, intervals, outputs):
            def read(relative: str, interval: dict[str, Any]) -> np.ndarray:
                return legacy._read_excerpt(
                    legacy._safe_dataset_path(root, relative),
                    int(interval["reference_start"]), int(interval["reference_end"]),
                    expected_frames=int(track["frames"]),
                )

            mixtures = [read(track["mixture"], item) for item in rows]
            references = [np.stack([read(track["stems"][source], item)
                                    for source in SOURCE_ORDER]) for item in rows]
            estimates = [shipping_residual(estimate, mixture)
                         for estimate, mixture in zip(raw, mixtures)]
            for mixture, estimate in zip(mixtures, estimates):
                reconstruction_max_abs = max(reconstruction_max_abs, float(np.max(
                    np.abs(estimate.sum(axis=0, dtype=np.float32) - mixture.astype(np.float32))
                )))
            score = legacy._score_track(
                track["name"], rows, mixtures, references, estimates, metric_config
            )
            scores.append(score)
            if audio_dir is not None:
                export_audio(audio_dir, track["name"], rows, mixtures, references, estimates)
            print(json.dumps({"event": "track", "checkpoint": checkpoint.name,
                              "track": track["name"], "full_sdr_db": score["full_sdr_db"]}),
                  flush=True)
    result = {
        "checkpoint": identity,
        "aggregate": legacy._aggregate_tracks(scores),
        "tracks": scores,
        "stream_batches": stream_batches,
        "reconstruction_max_abs": reconstruction_max_abs,
        "evaluation_seconds": time.monotonic() - started,
        "audio_dir": str(audio_dir.resolve()) if audio_dir is not None else None,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--panel", choices=("dev", "full"), default="dev")
    parser.add_argument("--track-indices", type=int, nargs="+")
    parser.add_argument("--excerpt-starts", type=float, nargs="+")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--allow-custom-gains", action="store_true", help="Evaluate an explicit output-gain ablation; the decoder must still be finalized.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    manifest = json.loads(args.manifest.read_text())
    tracks, config = select_panel(
        manifest, json.loads(DEFAULT_CONFIG.read_text()), panel=args.panel,
        track_indices=args.track_indices, excerpt_starts=args.excerpt_starts,
        duration=args.duration,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "panel": args.panel if args.track_indices is None and args.excerpt_starts is None
                 and args.duration == 15.0 else "custom-development",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": legacy._sha256_file(args.manifest),
        "track_names": [track["name"] for track in tracks],
        "excerpts": config["default_excerpts"],
        "alignment_samples": 512,
        "algorithmic_latency_samples": 1024,
        "streaming_state": "independent per track, continuous from sample zero",
        "output_policy": "float32 Other = aligned mixture - sum(Drums,Bass,Vocals)",
        "metrics": MetricConfig.from_mapping(config["metrics"]).to_dict(),
        "metric_source_sha256": legacy._sha256_file(ROOT / "research/metrics.py"),
        "streaming_source_sha256": legacy._sha256_file(ROOT / "research/evaluate.py"),
        "model_source_sha256": legacy._sha256_file(ROOT / "hs_tasnet/hs_tasnet.py"),
        "evaluator_sha256": legacy._sha256_file(Path(__file__)),
        "torch_version": torch.__version__,
        "device": args.device,
        "precision": "float32",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, checkpoint in enumerate(args.checkpoint):
        audio_dir = (args.audio_dir / f"{index:02d}-{checkpoint.stem}"
                     if args.audio_dir is not None else None)
        result = evaluate_checkpoint(
            checkpoint, manifest=manifest, tracks=tracks, config=config,
            device=torch.device(args.device), batch_size=args.batch_size, audio_dir=audio_dir,
            allow_custom_gains=args.allow_custom_gains,
        )
        report["results"].append(result)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"event": "checkpoint_complete", "checkpoint": str(checkpoint),
                          **{key: result["aggregate"][key]
                             for key in ("full_sdr_db", "low_sdr_db", "bleed_sir_db")}}),
              flush=True)


if __name__ == "__main__":
    main()
