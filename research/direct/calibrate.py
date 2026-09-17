#!/usr/bin/env python3
"""Fit three output gains from one continuous validation pass.

Other is always mixture minus gained Drums/Bass/Vocals. Per-window quadratic
error statistics reproduce the existing scale-dependent SDR without rerunning
the network for each gain triple. The fit uses the existing validation set;
leave-one-track-out gain fitting is a stability diagnostic, not an unseen-test
result for a model already selected on these tracks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.direct import evaluate as direct
from research.metrics import MetricConfig, SOURCE_ORDER, fft_bandpass, frame_ranges, rms_dbfs, windowed_sdr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "research/direct/runs/c91-refined-v1/model.pt"
BANDS = {"full": None, "low": (20.0, 250.0)}


def window_statistics(dbv, mixture, references, config):
    """Store E_i(a)=a_i² A_ii-2a_i b_i+S_i; E_other(a)=c-2b·a+aᵀAa."""
    rows = {key: [] for key in ("signal", "active", "gram", "own_cross", "other_cross", "other_power")}
    for start, end in frame_ranges(references.shape[-1], config.window_samples, config.hop_samples):
        x = np.asarray(dbv[..., start:end], dtype=np.float64).reshape(3, -1)
        ref = np.asarray(references[..., start:end], dtype=np.float64).reshape(4, -1)
        q = np.asarray(mixture[..., start:end], dtype=np.float64).reshape(-1) - ref[3]
        rows["signal"].append(np.square(ref).sum(axis=1))
        rows["active"].append([rms_dbfs(stem, config.epsilon) > config.activity_dbfs for stem in ref])
        rows["gram"].append(x @ x.T)
        rows["own_cross"].append((x * ref[:3]).sum(axis=1))
        rows["other_cross"].append(x @ q)
        rows["other_power"].append(q @ q)
    return {key: np.asarray(value) for key, value in rows.items()}


def collect_statistics(checkpoint, device, batch_size):
    started = time.monotonic()
    manifest = json.loads(direct.DEFAULT_MANIFEST.read_text())
    tracks, config = direct.select_panel(
        manifest, json.loads(direct.DEFAULT_CONFIG.read_text()), panel="full"
    )
    metrics = MetricConfig.from_mapping(config["metrics"])
    model, identity = direct.load_checkpoint(checkpoint, device, allow_custom_gains=True)
    root = Path(manifest["root"])
    collected = {band: [] for band in BANDS}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for offset in range(0, len(tracks), batch_size):
        batch = tracks[offset:offset + batch_size]
        intervals = [direct.legacy._reference_intervals(track, config) for track in batch]
        outputs, _ = direct.legacy._stream_audio_batch(
            model, [direct.legacy._safe_dataset_path(root, track["mixture"]) for track in batch],
            [[(row["estimate_start"], row["estimate_end"]) for row in rows] for rows in intervals],
            device=device, expected_frames=[track["frames"] for track in batch],
        )
        for track, rows, predictions in zip(batch, intervals, outputs):
            track_stats = {band: [] for band in BANDS}
            for interval, prediction in zip(rows, predictions):
                def read(relative):
                    return direct.legacy._read_excerpt(
                        direct.legacy._safe_dataset_path(root, relative),
                        interval["reference_start"], interval["reference_end"],
                        expected_frames=track["frames"],
                    )

                mixture = read(track["mixture"])
                refs = np.stack([read(track["stems"][source]) for source in SOURCE_ORDER])
                # WAVs and model outputs are FP32-exact here; sufficient statistics
                # use float64 linear gains. Candidate deployment is checked separately.
                audio = np.concatenate((prediction[:3].astype(np.float64), mixture[None], refs))
                for band, limits in BANDS.items():
                    filtered = audio if limits is None else fft_bandpass(audio, 44_100, *limits)
                    track_stats[band].append(window_statistics(
                        filtered[:3], filtered[3], filtered[4:], metrics
                    ))
            for band in BANDS:
                collected[band].append({
                    key: np.concatenate([item[key] for item in track_stats[band]])
                    for key in track_stats[band][0]
                })
            print(json.dumps({"event": "statistics", "track": track["name"]}), flush=True)
    arrays = {f"{band}_{key}": np.stack([item[key] for item in values])
              for band, values in collected.items() for key in values[0]}
    metadata = {
        "checkpoint": identity,
        "track_names": [track["name"] for track in tracks],
        "manifest_sha256": direct.legacy._sha256_file(direct.DEFAULT_MANIFEST),
        "excerpts": config["default_excerpts"],
        "alignment_samples": 512,
        "current_absolute_gains": (2 * np.asarray(identity["output_source_scales"][:3])).tolist(),
        "metric_config": config["metrics"],
        "statistics_seconds": time.monotonic() - started,
        "peak_cuda_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                    if device.type == "cuda" else None),
        "torch_version": torch.__version__,
        "model_source_sha256": direct.legacy._sha256_file(ROOT / "hs_tasnet/hs_tasnet.py"),
        "evaluator_source_sha256": direct.legacy._sha256_file(Path(__file__)),
    }
    del model, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, metadata


def score_gains(arrays, metadata, gains):
    """Return [candidate, track, stem] full/low scores using legacy weighting."""
    config = MetricConfig.from_mapping(metadata["metric_config"])
    relative = np.asarray(gains, dtype=np.float64) / metadata["current_absolute_gains"]
    per_track = {}
    for band in BANDS:
        signal, active = arrays[f"{band}_signal"], arrays[f"{band}_active"]
        gram = arrays[f"{band}_gram"]
        a = relative[:, None, None, :]
        own_error = (a * a * np.diagonal(gram, axis1=-2, axis2=-1)[None]
                     - 2 * a * arrays[f"{band}_own_cross"][None] + signal[None, ..., :3])
        other_error = (arrays[f"{band}_other_power"][None]
                       - 2 * np.einsum("ci,twi->ctw", relative, arrays[f"{band}_other_cross"])
                       + np.einsum("ci,tw ij,cj->ctw", relative, gram, relative, optimize=True))
        error = np.concatenate((own_error, other_error[..., None]), axis=-1)
        db = np.clip(10 * np.log10((signal[None] + config.epsilon)
                                  / (np.maximum(error, 0) + config.epsilon)),
                     config.db_floor, config.db_ceiling)
        count = active.sum(axis=1)[None]
        total = np.where(active[None], db, 0).sum(axis=2)
        per_track[band] = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return per_track


def aggregate(values):
    return np.nanmean(values, axis=-2).mean(axis=-1)


def summary(index, gains, scores, metadata, *, include_tracks=False):
    result = {"absolute_gains": np.asarray(gains[index]).tolist()}
    result.update({f"{band}_sdr_db": float(aggregate(scores[band][index])) for band in BANDS})
    result["per_stem"] = {
        source: {f"{band}_sdr_db": float(np.nanmean(scores[band][index], axis=0)[stem])
                 for band in BANDS}
        for stem, source in enumerate(SOURCE_ORDER)
    }
    if include_tracks:
        result["per_track"] = [{
            "name": name,
            **{f"{band}_sdr_db": float(np.nanmean(scores[band][index, track])) for band in BANDS},
            "per_stem": {source: {
                f"{band}_sdr_db": (float(scores[band][index, track, stem])
                                   if np.isfinite(scores[band][index, track, stem]) else None)
                for band in BANDS} for stem, source in enumerate(SOURCE_ORDER)},
        } for track, name in enumerate(metadata["track_names"])]
    return result


def analyze(arrays, metadata, step=0.025, reference_report=None):
    ranges = [(0.90, 1.10), (0.90, 1.10), (0.75, 1.10)]
    grid = [np.round(np.arange(low, high + step * 0.1, step), 8) for low, high in ranges]
    gains = np.asarray([metadata["current_absolute_gains"], *itertools.product(*grid)])
    # Validate the current gain setting before using the statistics to select gains.
    baseline_scores = score_gains(arrays, metadata, gains[:1])
    baseline = summary(0, gains[:1], baseline_scores, metadata, include_tracks=True)
    check = None
    if reference_report is not None:
        reference = json.loads(reference_report.read_text())
        matching = [row for row in reference["results"]
                    if row["checkpoint"]["sha256"] == metadata["checkpoint"]["sha256"]]
        if len(matching) != 1:
            raise ValueError("reference report does not identify this exact checkpoint")
        check = {band: baseline[f"{band}_sdr_db"] - matching[0]["aggregate"][f"{band}_sdr_db"]
                 for band in BANDS}
        if max(abs(value) for value in check.values()) > 1e-5:
            raise ValueError(f"statistics disagree with the normal evaluator: {check}")
    scores = score_gains(arrays, metadata, gains)
    objective = aggregate(scores["full"])
    best_index = int(np.argmax(objective))
    best = summary(best_index, gains, scores, metadata, include_tracks=True)
    folds = []
    held_out = {band: [] for band in BANDS}
    for track, name in enumerate(metadata["track_names"]):
        training = np.delete(scores["full"], track, axis=1)
        selected = int(np.argmax(aggregate(training)))
        folds.append({
            "held_out_track": name, "absolute_gains": gains[selected].tolist(),
            **{f"{band}_sdr_delta_db": float(np.nanmean(scores[band][selected, track])
                                             - np.nanmean(scores[band][0, track])) for band in BANDS},
        })
        for band in BANDS:
            held_out[band].append(scores[band][selected, track])
    loo = {f"{band}_sdr_db": float(aggregate(np.asarray(held_out[band]))) for band in BANDS}
    loo.update({f"{band}_sdr_delta_db": loo[f"{band}_sdr_db"] - baseline[f"{band}_sdr_db"]
                for band in BANDS})
    return {
        "metadata": metadata,
        "selection_caveat": "Gains and model both use the existing validation set. Leave-one-track-out gain fitting is a stability diagnostic, not unseen-test evidence for the already selected model.",
        "numerics": "Float64 quadratic errors for linear rescaling of FP32 streamed outputs. Normal FP32 model evaluation must verify any selected deployment gain change.",
        "reference_report_delta_db": check,
        "baseline": baseline, "best": best,
        "best_delta_db": {band: best[f"{band}_sdr_db"] - baseline[f"{band}_sdr_db"] for band in BANDS},
        "leave_one_track_out": {**loo, "folds": folds},
        "grid": {"ranges": ranges, "step": step, "count": len(gains),
                 "results": [{"absolute_gains": gain.tolist(),
                              "full_sdr_db": float(objective[index]),
                              "low_sdr_db": float(aggregate(scores["low"][index]))}
                             for index, gain in enumerate(gains)]},
    }


def self_test():
    config = MetricConfig(sample_rate=1000, window_samples=100, hop_samples=100,
                          bands_hz={"low_20_250": (20, 250)})
    rng = np.random.default_rng(17)
    refs = rng.normal(0, 0.1, (4, 2, 300))
    refs[1, :, :100] = 0
    mixture = refs.sum(axis=0) + rng.normal(0, 0.001, (2, 300))
    dbv = refs[:3] + rng.normal(0, 0.03, (3, 2, 300))
    gains = np.array([[1, 1, 0.9], [0.95, 1.1, 1.0]])
    metadata = {"current_absolute_gains": gains[0], "metric_config": {
        "sample_rate": 1000, "window_samples": 100, "hop_samples": 100,
        "bands_hz": {"low_20_250": (20, 250)},
    }}
    arrays = {}
    for band, limits in BANDS.items():
        audio = np.concatenate((dbv, mixture[None], refs))
        if limits is not None:
            audio = fft_bandpass(audio, config.sample_rate, *limits)
        arrays.update({f"{band}_{key}": value[None] for key, value in
                       window_statistics(audio[:3], audio[3], audio[4:], config).items()})
    scores = score_gains(arrays, metadata, gains)
    maximum_error = 0.0
    for index, gain in enumerate(gains):
        estimate = np.empty_like(refs)
        estimate[:3] = dbv * (gain / gains[0])[:, None, None]
        estimate[3] = mixture - estimate[:3].sum(axis=0)
        for band, limits in BANDS.items():
            reference, prediction = refs, estimate
            if limits is not None:
                reference = fft_bandpass(reference, config.sample_rate, *limits)
                prediction = fft_bandpass(prediction, config.sample_rate, *limits)
            expected = [windowed_sdr(ref, est, config)["db"] for ref, est in zip(reference, prediction)]
            maximum_error = max(maximum_error, float(np.max(np.abs(scores[band][index, 0] - expected))))
    if maximum_error > 1e-10:
        raise AssertionError(f"gain statistic parity failed: {maximum_error}")
    print(json.dumps({"self_test": "passed", "maximum_sdr_error_db": maximum_error}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stats-from", type=Path, help="Reuse saved .npz statistics without GPU inference")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--grid-step", type=float, default=0.025)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or not 0.01 <= args.grid_step <= 0.1 or args.batch_size < 1:
        parser.error("require --output, grid step in [0.01,0.1], and positive batch size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stats_from is not None:
        with np.load(args.stats_from, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            arrays = {key: archive[key] for key in archive.files if key != "metadata"}
        stats_path = args.stats_from
    else:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        arrays, metadata = collect_statistics(args.checkpoint, torch.device(args.device), args.batch_size)
        stats_path = args.output.with_suffix(".stats.npz")
        np.savez_compressed(stats_path, metadata=json.dumps(metadata), **arrays)
    report = analyze(arrays, metadata, args.grid_step, args.reference_report)
    report["statistics_path"] = str(stats_path.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline": report["baseline"]["full_sdr_db"],
                      "best": report["best"]["full_sdr_db"],
                      "absolute_gains": report["best"]["absolute_gains"],
                      "leave_one_track_out_delta": report["leave_one_track_out"]["full_sdr_delta_db"]}), flush=True)


if __name__ == "__main__":
    main()
