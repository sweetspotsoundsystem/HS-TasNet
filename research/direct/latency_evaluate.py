"""Continuous FP32 evaluation for the 512-sample current-chunk models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.direct import evaluate as shared
from research.direct.latency11 import load_model, save_model
from research import evaluate as legacy
from research.metrics import MetricConfig, SOURCE_ORDER


@torch.inference_mode()
def stream_audio_batch(model, audio_paths, capture_intervals, *, device, expected_frames,
                       unroll_hops=1):
    """Stream callbacks, optionally evaluating several causal frames together.

    The default calls forward_chunk literally. Unrolled execution is a pilot
    optimization; it preserves the model's 512-sample receptive-field timing.
    """
    if not isinstance(unroll_hops, int) or unroll_hops < 1:
        raise ValueError("unroll_hops must be a positive integer")
    batch_size, hop = len(audio_paths), 512 * unroll_hops
    if not batch_size or len(capture_intervals) != batch_size or len(expected_frames) != batch_size:
        raise ValueError("Require equally sized, nonempty stream arguments")
    captures, ends = [], []
    for intervals, frames in zip(capture_intervals, expected_frames):
        if not intervals or any(start < 0 or end <= start or end > frames for start, end in intervals):
            raise ValueError("Capture interval is outside the declared stream")
        captures.append(legacy._Capture(intervals, (4, 2)))
        ends.append(max(end for _, end in intervals))
    state = model.initial_state(batch_size, device=device)
    readers = legacy._open_blocked_readers(audio_paths, expected_frames, hop=hop, block_hops=64)
    staging = np.zeros((batch_size, 2, hop), dtype=np.float32)
    callbacks = 0
    try:
        for position in range(0, max(ends), hop):
            staging.fill(0)
            valid_frames = []
            for index, (reader, end) in enumerate(zip(readers, ends)):
                if position >= end:
                    valid_frames.append(0)
                    continue
                block = reader.read_hop()
                valid = len(block)
                if valid < min(hop, end - position):
                    raise RuntimeError(f"Audio ended early: {audio_paths[index]}")
                staging[index, :, :valid] = block.T
                valid_frames.append(valid)
            tensor = torch.from_numpy(staging).to(device)
            output, state = (model.forward_chunk(tensor, state) if unroll_hops == 1
                             else model(tensor, state))
            if output.shape != (batch_size, 4, 2, hop):
                raise RuntimeError(f"Unexpected callback output shape: {output.shape}")
            overlaps = [capture.overlap_bounds(position, position + valid) if valid else None
                        for capture, valid in zip(captures, valid_frames)]
            if any(bounds is not None for bounds in overlaps):
                cpu = output.float().cpu().numpy()
                if not np.isfinite(cpu).all():
                    raise FloatingPointError("Non-finite streaming output")
                for index, bounds in enumerate(overlaps):
                    if bounds is not None:
                        start, end = bounds
                        captures[index].add(start, cpu[index, ..., start-position:end-position])
            callbacks += 1
    finally:
        for reader in readers:
            reader.close()
    return [capture.finish() for capture in captures], {
        "batch_size": batch_size, "callback_count": (max(ends) + 511) // 512,
        "scalar_equivalent_callback_count": sum((end + 511) // 512 for end in ends),
        "forward_call_count": callbacks, "processed_hops_per_stream": callbacks * unroll_hops,
        "unroll_hops": unroll_hops, "io_block_hops": 64 * unroll_hops,
    }


def evaluate_checkpoint(kind, checkpoint, *, manifest, tracks, config, device,
                        batch_size=14, audio_dir=None, vocal_gain=None, unroll_hops=1):
    if config.get("alignment_samples") != 0:
        raise ValueError("Current-chunk evaluation requires alignment_samples=0")
    started = time.monotonic()
    model = load_model(kind, checkpoint, device=device, vocal_gain=vocal_gain)
    # Materialize retained native parents once so the result always names an
    # exact direct-format checkpoint rather than an implicit module default.
    if checkpoint is None:
        checkpoint = Path("research/direct/runs/latency11/baselines") / f"{kind}-native.pt"
        if vocal_gain is not None:
            checkpoint = checkpoint.with_stem(f"{kind}-gain{vocal_gain:g}")
        save_model(model, checkpoint)
    identity = {"path": str(checkpoint.resolve()), "sha256": legacy._sha256_file(checkpoint),
                "kind": kind, "num_parameters": model.num_parameters,
                "output_source_scales": model.output_source_scales.cpu().tolist(),
                "vocal_gain_override": vocal_gain}
    root = Path(manifest["root"])
    metric_config = MetricConfig.from_mapping(config["metrics"])
    scores, stream_batches = [], []
    reconstruction_max_abs = 0.
    for offset in range(0, len(tracks), batch_size):
        batch = tracks[offset:offset + batch_size]
        rows = [legacy._reference_intervals(track, config) for track in batch]
        outputs, info = stream_audio_batch(
            model, [legacy._safe_dataset_path(root, track["mixture"]) for track in batch],
            [[(row["estimate_start"], row["estimate_end"]) for row in intervals] for intervals in rows],
            device=device, expected_frames=[track["frames"] for track in batch],
            unroll_hops=unroll_hops,
        )
        stream_batches.append(info)
        for track, intervals, raw in zip(batch, rows, outputs):
            def read(relative, interval):
                return legacy._read_excerpt(legacy._safe_dataset_path(root, relative),
                                            interval["reference_start"], interval["reference_end"],
                                            expected_frames=track["frames"])
            mixtures = [read(track["mixture"], row) for row in intervals]
            references = [np.stack([read(track["stems"][source], row) for source in SOURCE_ORDER])
                          for row in intervals]
            estimates = [shared.shipping_residual(estimate, mixture) for estimate, mixture in zip(raw, mixtures)]
            for estimate, mixture in zip(estimates, mixtures):
                reconstruction_max_abs = max(reconstruction_max_abs,
                    float(np.max(np.abs(estimate.sum(axis=0) - mixture.astype(np.float32)))))
            score = legacy._score_track(track["name"], intervals, mixtures, references, estimates, metric_config)
            scores.append(score)
            if audio_dir is not None:
                shared.export_audio(audio_dir, track["name"], intervals, mixtures, references, estimates)
            print(json.dumps({"event": "track", "checkpoint": checkpoint.name,
                              "track": track["name"], "full_sdr_db": score["full_sdr_db"]}), flush=True)
    result = {"checkpoint": identity, "aggregate": legacy._aggregate_tracks(scores), "tracks": scores,
              "stream_batches": stream_batches, "reconstruction_max_abs": reconstruction_max_abs,
              "evaluation_seconds": time.monotonic() - started,
              "audio_dir": str(audio_dir.resolve()) if audio_dir is not None else None}
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("c126", "c191"), required=True)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("dev", "full"), default="full")
    parser.add_argument("--track-indices", type=int, nargs="+")
    parser.add_argument("--excerpt-starts", type=float, nargs="+")
    parser.add_argument("--duration", type=float, default=15.)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--unroll-hops", type=int, default=1,
                        help="Causal frames evaluated together for pilots; 1 uses literal callbacks.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocal-gain", type=float)
    parser.add_argument("--audio-dir", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.unroll_hops) < 1:
        parser.error("Batch size, thread count and unroll hops must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    manifest = json.loads(shared.DEFAULT_MANIFEST.read_text())
    tracks, config = shared.select_panel(manifest, json.loads(shared.DEFAULT_CONFIG.read_text()),
        panel=args.panel, track_indices=args.track_indices, excerpt_starts=args.excerpt_starts,
        duration=args.duration, alignment_samples=0)
    report = {"schema_version": 1,
              "panel": args.panel if args.track_indices is None and args.excerpt_starts is None
                       and args.duration == 15.0 else "custom-development",
              "manifest": str(shared.DEFAULT_MANIFEST),
              "manifest_sha256": legacy._sha256_file(shared.DEFAULT_MANIFEST),
              "track_names": [track["name"] for track in tracks], "excerpts": config["default_excerpts"],
              "alignment_samples": 0, "algorithmic_latency_samples": 512,
              "streaming_state": "independent per track, continuous from sample zero",
              "output_policy": "float32 Other = aligned mixture - sum(Drums,Bass,Vocals)",
              "metrics": MetricConfig.from_mapping(config["metrics"]).to_dict(),
              "metric_source_sha256": legacy._sha256_file(shared.ROOT / "research/metrics.py"),
              "evaluator_sha256": legacy._sha256_file(Path(__file__)),
              "torch_version": torch.__version__, "device": args.device, "precision": "float32",
              "unroll_hops": args.unroll_hops,
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, checkpoint in enumerate(args.checkpoint or [None]):
        name = checkpoint.stem if checkpoint is not None else f"{args.kind}-native"
        audio_dir = args.audio_dir / f"{index:02d}-{name}" if args.audio_dir is not None else None
        result = evaluate_checkpoint(args.kind, checkpoint, manifest=manifest, tracks=tracks,
            config=config, device=torch.device(args.device), batch_size=args.batch_size,
            audio_dir=audio_dir, vocal_gain=args.vocal_gain, unroll_hops=args.unroll_hops)
        report["results"].append(result)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"event": "checkpoint_complete", "checkpoint": result["checkpoint"],
                          **{key: result["aggregate"][key] for key in ("full_sdr_db", "low_sdr_db", "bleed_sir_db")}}), flush=True)


if __name__ == "__main__":
    main()
