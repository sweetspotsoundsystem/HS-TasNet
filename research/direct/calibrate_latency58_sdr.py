"""Collect coupled output-gain statistics with the preserved hop128 streamer.

The one-track functional mode checks the working baseline without fitting
gains. Full-panel diagnostic mode reuses the established three-gain analysis.
Neither mode writes model weights or changes the primary evaluation protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write

_MODEL = None
_PLAN = None
_REFERENCE = None


def initialize_worker(plan):
    global _MODEL, _PLAN, _REFERENCE
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    if plan["model_kind"] == "working_baseline":
        from research.direct.latency58_sdr_teacher import load_initial_student
        _MODEL = load_initial_student().eval().requires_grad_(False)
    else:
        if plan["model_kind"] == "context_candidate":
            from research.direct.evaluate_latency58_context import load_evaluation_model
        else:
            from research.direct.evaluate_latency58_sdr import load_evaluation_model
        _MODEL, _ = load_evaluation_model(plan)
    require(model_state_sha256(_MODEL) == plan["model_state_sha256"], "Calibration model differs")
    _PLAN = plan
    _REFERENCE = read(Path(plan["reference_quality_directory"]) / "result.json")["results"][0]


def collect_track(index):
    import numpy as np
    import torch
    from research.direct import evaluate as shared
    from research import evaluate as legacy
    from research.direct.calibrate import BANDS, window_statistics, score_gains
    from research.direct.latency58_evaluate import (
        DEFAULT_CONFIG, DEFAULT_MANIFEST, SOURCE_ORDER, model_state_sha256,
        plan_latency58_stream, stream_latency58_track,
    )
    from research.metrics import MetricConfig, fft_bandpass

    require(_MODEL is not None and _PLAN is not None and _REFERENCE is not None, "Missing worker state")
    began = time.monotonic()
    rng = torch.get_rng_state().clone()
    manifest = read(DEFAULT_MANIFEST)
    tracks, config = shared.select_panel(manifest, read(DEFAULT_CONFIG), panel="full",
                                         track_indices=[index], alignment_samples=128)
    track = tracks[0]
    intervals = legacy._reference_intervals(track, config)
    stream_plan = plan_latency58_stream(intervals, int(track["frames"]), unroll_hops=64, io_block_hops=64)
    root = Path(manifest["root"])
    predictions, delayed, stream = stream_latency58_track(
        _MODEL, legacy._safe_dataset_path(root, track["mixture"]), stream_plan)
    metrics = MetricConfig.from_mapping(config["metrics"])
    collected = {band: [] for band in BANDS}
    for interval, prediction, physical in zip(intervals, predictions, delayed, strict=True):
        def excerpt(relative):
            return legacy._read_excerpt(legacy._safe_dataset_path(root, relative),
                                         interval["reference_start"], interval["reference_end"],
                                         expected_frames=int(track["frames"]))
        mixture = excerpt(track["mixture"])
        references = np.stack([excerpt(track["stems"][s]) for s in SOURCE_ORDER])
        require(np.array_equal(physical, mixture.astype(np.float32)), "Calibration lost physical alignment")
        audio = np.concatenate((prediction[:3].astype(np.float64), mixture[None], references))
        for band, limits in BANDS.items():
            filtered = audio if limits is None else fft_bandpass(audio, 44100, *limits)
            collected[band].append(window_statistics(filtered[:3], filtered[3], filtered[4:], metrics))
    arrays = {f"{band}_{key}": np.concatenate([part[key] for part in parts])
              for band, parts in collected.items() for key in parts[0]}
    gains = (2 * _MODEL.output_source_scales.detach().numpy()[:3]).astype(np.float64).tolist()
    metadata = {"current_absolute_gains": gains, "metric_config": config["metrics"]}
    current = score_gains({key: value[None] for key, value in arrays.items()}, metadata, [gains])
    reference = _REFERENCE["tracks"][index]
    require(reference["name"] == track["name"], "Reference track order differs")
    maximum_error = 0.0
    for band in BANDS:
        for stem_index, stem in enumerate(SOURCE_ORDER):
            value = current[band][0, 0, stem_index]
            expected = reference["per_stem"][stem]["full_sdr_db"] if band == "full" else \
                reference["per_stem"][stem]["band_sdr_db"]["low_20_250"]
            if expected is None:
                require(np.isnan(value), "Activity masks differ")
            else:
                require(np.isfinite(value), "Nonfinite gain statistic score")
                maximum_error = max(maximum_error, abs(float(value) - expected))
    require(maximum_error <= 1e-5, "Statistics differ from the stored per-track/per-stem SDR")
    stream.update(track=track["name"], physical_alignment_verified_by_delayed_mixture=True)
    require(stream == _REFERENCE["stream_batches"][index], "Calibration changed the original stream geometry")
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and model_state_sha256(_MODEL) == _PLAN["model_state_sha256"], "Calibration changed model or RNG")
    return index, arrays, {"track": track["name"], "current_absolute_gains": gains,
        "metric_config": config["metrics"], "excerpts": config["default_excerpts"],
        "maximum_original_sdr_error_db": maximum_error, "stream_metadata_exact": True,
        "elapsed_seconds": time.monotonic() - began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Calibration plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-calibration-plan-v1"
            and plan["mode"] in ("functional_one_track", "full14_diagnostic")
            and plan["model_kind"] in ("working_baseline", "sdr_candidate", "context_candidate"),
            "Unknown calibration plan")
    functional = plan["mode"] == "functional_one_track"
    require(plan["track_indices"] == ([0] if functional else list(range(14)))
            and plan["workers"] == (1 if functional else 2)
            and (not functional or plan["model_kind"] == "working_baseline")
            and (functional or plan["model_kind"] != "working_baseline")
            and plan["grid_step"] == 0.025, "Calibration scope differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1 workers")
    bindings = plan["source_bindings"]
    require(all(sha(p) == s for p, s in bindings.items()), "Calibration inputs changed")
    from research.direct.report_latency58_sdr import load_completed
    evidence = {}
    _, reference = load_completed(Path(plan["reference_quality_directory"]), evidence,
                                   canonical_baseline=plan["model_kind"] == "working_baseline")
    require(all(bindings.get(p) == s for p, s in evidence.items())
            and len(reference["results"][0]["tracks"]) == 14
            and reference["results"][0]["model"]["model_state_sha256"] == plan["model_state_sha256"],
            "Require the completed original panel for this exact model")
    if not functional:
        proof = plan["functional_proof"]
        check = read(proof["path"])
        execution = plan["functional_execution"]
        completed = read(execution["path"])
        require(bindings.get(proof["path"]) == proof["sha256"] == sha(proof["path"])
                and bindings.get(execution["path"]) == execution["sha256"] == sha(execution["path"])
                and check["status"] == "pass" and check["mode"] == "functional_one_track"
                and check["source_bindings_unchanged"] and not check["gains_fitted"]
                and all(sha(p) == s for p, s in check["source_bindings"].items())
                and completed["actual_exit_code"] == 0 and not completed["timed_out"]
                and completed["source_bindings_unchanged"]
                and completed["plan_sha256"] == check["plan_sha256"], "Missing functional qualification")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists()
            and not (out / "statistics.npz").exists(), "Preserve previous calibration")
    import numpy as np
    from research.direct.calibrate import analyze, self_test
    if functional:
        self_test()
    began = time.monotonic()
    results = {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(collect_track, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, arrays, metadata = future.result()
            require(index == futures[future] and index not in results, "Worker returned the wrong track")
            results[index] = (arrays, metadata)
            row = {"event": "statistics", "index": index, **metadata}
            progress.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    require(set(results) == set(plan["track_indices"]), "Incomplete statistics")
    ordered = [results[i] for i in plan["track_indices"]]
    first = ordered[0][1]
    require(all(all(meta[k] == first[k] for k in ("current_absolute_gains", "metric_config", "excerpts"))
                for _, meta in ordered), "Worker protocols differ")
    arrays = {key: np.stack([data[key] for data, _ in ordered]) for key in ordered[0][0]}
    metadata = {"checkpoint": plan["checkpoint"], "model_state_sha256": plan["model_state_sha256"],
        "track_names": [meta["track"] for _, meta in ordered],
        **{k: first[k] for k in ("current_absolute_gains", "metric_config", "excerpts")},
        "alignment_samples": 128, "intended_total_latency_samples": 256,
        "manifest_sha256": reference["manifest_sha256"], "precision": reference["precision"]}
    fit = None if functional else analyze(arrays, metadata, step=0.025,
                                          reference_report=Path(plan["reference_quality_directory"]) / "result.json")
    require(all(sha(p) == s for p, s in bindings.items()), "Calibration inputs changed during execution")
    with (out / "statistics.npz").open("xb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)
    result = {"schema": "latency58-sdr-calibration-result-v1", "status": "pass", "mode": plan["mode"],
        "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True, "source_bindings": bindings,
        "metadata": metadata, "maximum_original_sdr_error_db": max(meta["maximum_original_sdr_error_db"] for _, meta in ordered),
        "all_stream_metadata_exact": True, "statistics_sha256": sha(out / "statistics.npz"),
        "gains_fitted": not functional, "fit": fit, "model_weights_written": False,
        "quality_selected": False, "normal_fp32_gain_evaluation_required": not functional,
        "elapsed_seconds": time.monotonic() - began}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "mode": plan["mode"], "gains_fitted": not functional,
        "maximum_original_sdr_error_db": result["maximum_original_sdr_error_db"],
        "best_delta_db": fit["best_delta_db"] if fit else None}), flush=True)


if __name__ == "__main__":
    main()
