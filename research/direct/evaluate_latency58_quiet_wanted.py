"""Measure wanted-source fidelity below the primary activity threshold.

The original full-mixture scorer and streamer must reproduce every stored
primary track exactly before these supplemental window measurements are used.
No primary score, activity threshold, source level or model gain is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space

_MODEL = _PLAN = _STORED = _SOURCES = None
PREFIXES = {"working": "teacher-half250", "focused": "vocal-focus-focused-250",
            "ordinary_only": "counterfactual-teacher-ordinary-only-250"}
QUIET_BUCKETS = ("(-60,-50]_dbfs", "(-70,-60]_dbfs", "(-80,-70]_dbfs")


def window_metrics(reference, estimate, epsilon=1e-12):
    import numpy as np
    from research.metrics import rms_dbfs
    reference, estimate = np.asarray(reference, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    require(reference.shape == estimate.shape == (2, 44100)
            and np.isfinite(reference).all() and np.isfinite(estimate).all(), "Different window geometry or finiteness")
    error = estimate - reference
    truth_energy = float(np.square(reference).sum())
    error_energy = float(np.square(error).sum())
    return {"reference_rms_dbfs": rms_dbfs(reference, epsilon),
            "output_rms_dbfs": rms_dbfs(estimate, epsilon), "error_rms_dbfs": rms_dbfs(error, epsilon),
            "scale_dependent_sdr_db": float(10 * np.log10((truth_energy + epsilon) / (error_energy + epsilon)))
                if truth_energy > 0 else None,
            "signed_reference_projection_gain": float((estimate * reference).sum() / (truth_energy + epsilon))
                if truth_energy > 0 else None}


def check_window_metrics():
    import numpy as np
    tone = np.sin(2 * np.pi * 100 * np.arange(44100, dtype=np.float64) / 44100) * .001
    reference = np.stack((tone, tone))
    cases = {name: window_metrics(reference, reference * gain)
             for name, gain in (("exact", 1), ("muted", 0), ("inverted_half", -.5), ("doubled", 2))}
    for name, gain in (("exact", 1), ("muted", 0), ("inverted_half", -.5), ("doubled", 2)):
        require(abs(cases[name]["signed_reference_projection_gain"] - gain) < 1e-9,
                "Quiet reference gain check failed")
    require(cases["exact"]["scale_dependent_sdr_db"] > 90
            and abs(cases["muted"]["scale_dependent_sdr_db"]) < 1e-12
            and abs(cases["doubled"]["scale_dependent_sdr_db"]) < 1e-12
            and abs(cases["inverted_half"]["scale_dependent_sdr_db"] - 10 * np.log10(1 / 2.25)) < 1e-8,
            "Quiet fidelity does not distinguish correct output from muting, gain or sign errors")
    silent = window_metrics(np.zeros_like(reference), reference)
    require(silent["scale_dependent_sdr_db"] is silent["signed_reference_projection_gain"] is None,
            "An exactly silent reference cannot have a desired-fidelity score")
    return {"status": "pass", "cases": cases, "silent_reference": silent}


def initialize_worker(plan):
    global _MODEL, _PLAN, _STORED, _SOURCES
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    kind = plan["model"]["kind"]
    if kind == "working":
        from research.direct.latency58_sdr_teacher import load_initial_student
        model = load_initial_student()
    else:
        if kind == "focused":
            from research.direct.evaluate_latency58_vocal_focus import load_evaluation_model
        else:
            require(kind == "ordinary_only", "Different quiet-fidelity checkpoint family")
            from research.direct.evaluate_latency58_counterfactual import load_evaluation_model
        model, _ = load_evaluation_model(read(plan["quality_plan"]["path"]))
    _MODEL, _PLAN = model.eval().requires_grad_(False), plan
    _STORED = read(plan["quality_result"]["path"])["results"][0]
    inventory = read(plan["reference_inventory"]["path"])
    _SOURCES = {(r["track_index"], r["excerpt_index"], r["stem"], r["physical_start"]): r
                for r in inventory["source_windows"]}
    require(model_state_sha256(_MODEL) == plan["model"]["model_state_sha256"] and not torch.cuda.is_initialized(),
            "Quiet-fidelity worker loaded a different model or initialized CUDA")


def score_track(index):
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import model_state_sha256, plan_latency58_stream, stream_latency58_track
    from research.metrics import SOURCE_ORDER, MetricConfig, frame_ranges, fft_bandpass
    require(_MODEL is not None, "Quiet-fidelity worker is not initialized")
    manifest, source_config = read(_PLAN["manifest"]["path"]), read(_PLAN["config"]["path"])
    tracks, config = shared.select_panel(manifest, source_config, panel="full", track_indices=[index],
                                        excerpt_starts=None, duration=15.0, alignment_samples=128)
    require(len(tracks) == 1, "Expected one primary track")
    track, root = tracks[0], Path(manifest["root"])
    intervals = legacy._reference_intervals(track, config)
    metric = MetricConfig.from_mapping(config["metrics"])
    require(metric.window_samples == metric.hop_samples == metric.sample_rate == 44100
            and metric.activity_dbfs == -50, "Different primary window protocol")
    paths = [legacy._safe_dataset_path(root, track["mixture"]),
             *(legacy._safe_dataset_path(root, track["stems"][stem]) for stem in SOURCE_ORDER)]
    require(all(_PLAN["source_bindings"].get(str(p)) == sha(p) for p in paths), "Changed primary source audio")
    def excerpt(path, row):
        return legacy._read_excerpt(path, int(row["reference_start"]), int(row["reference_end"]),
                                    expected_frames=int(track["frames"]))
    mixtures = [excerpt(paths[0], row) for row in intervals]
    references = [np.stack([excerpt(p, row) for p in paths[1:]]) for row in intervals]
    fingerprint, rng, began = model_state_sha256(_MODEL), torch.get_rng_state().clone(), time.monotonic()
    stream_plan = plan_latency58_stream(intervals, int(track["frames"]))
    raw, delayed, metadata = stream_latency58_track(_MODEL, paths[0], stream_plan)
    require(all(np.array_equal(a, b.astype(np.float32)) for a, b in zip(delayed, mixtures, strict=True)),
            "Quiet-fidelity capture differs from its physical mixture")
    estimates = [shared.shipping_residual(a, b) for a, b in zip(raw, mixtures, strict=True)]
    score = legacy._score_track(track["name"], intervals, mixtures, references, estimates, metric)
    metadata.update(track=track["name"], physical_alignment_verified_by_delayed_mixture=True)
    require(score == _STORED["tracks"][index] and metadata == _STORED["stream_batches"][index],
            "Quiet-fidelity capture does not reproduce the original primary score and stream exactly")
    windows = []
    for excerpt_index, (interval, refs, outputs) in enumerate(zip(intervals, references, estimates, strict=True)):
        bands = {"full": (refs, outputs),
                 "low_20_250": (fft_bandpass(refs, 44100, 20, 250), fft_bandpass(outputs, 44100, 20, 250))}
        for start, stop in frame_ranges(refs.shape[-1], 44100, 44100):
            physical_start = int(interval["reference_start"]) + start
            for stem_index, stem in enumerate(SOURCE_ORDER):
                source = _SOURCES[(index, excerpt_index, stem, physical_start)]
                raw_source = np.ascontiguousarray(refs[stem_index, :, start:stop], dtype=np.float32)
                require(hashlib.sha256(raw_source.tobytes()).hexdigest() == source["source_float32_sha256"],
                        "Quiet-source inventory differs from the captured physical reference")
                values = {band: window_metrics(a[stem_index, :, start:stop], b[stem_index, :, start:stop], metric.epsilon)
                          for band, (a, b) in bands.items()}
                require(values["full"]["reference_rms_dbfs"] == source["source_rms_dbfs"], "Source level differs")
                windows.append({**source, "bands": values})
    closure = max(float(np.abs(e.sum(axis=0, dtype=np.float32) - m.astype(np.float32)).max())
                  for e, m in zip(estimates, mixtures, strict=True))
    require(len(windows) == 120 and closure <= 1e-6 and model_state_sha256(_MODEL) == fingerprint
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(_PLAN["source_bindings"][str(p)] == sha(p) for p in paths),
            "Incomplete windows or changed model, RNG, input or CPU scope")
    return index, {"track": track["name"], "track_index": index, "windows": windows,
                   "primary_track_score_exact": True, "primary_stream_metadata_exact": True,
                   "source_window_samples_exact": True, "model_and_rng_unchanged": True,
                   "closure_max_abs": closure, "elapsed_seconds": time.monotonic() - began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a frozen CPU1 quiet-fidelity evaluation")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-quiet-wanted-evaluation-plan-v1" and plan["workers"] == 2
            and plan["track_indices"] == list(range(14)) and plan["model"]["kind"] in PREFIXES
            and plan["model"]["prefix"] == PREFIXES[plan["model"]["kind"]], "Different evaluation scope")
    evidence = {}
    quality, report = load_completed(PHASE / (plan["model"]["prefix"] + "-full14-001"), evidence,
                                     canonical_baseline=plan["model"]["kind"] == "working")
    require(report["results"][0]["model"]["model_state_sha256"] == plan["model"]["model_state_sha256"]
            and all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Different primary endpoint")
    for name in ("quality_plan", "quality_result", "manifest", "config", "reference_inventory", "inventory_execution"):
        item = plan[name]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound prerequisite")
    require(read(plan["quality_plan"]["path"]) == quality
            and read(plan["quality_result"]["path"]) == report, "Different worker scoring prerequisites")
    inventory, inventory_execution = read(plan["reference_inventory"]["path"]), read(plan["inventory_execution"]["path"])
    require(inventory["schema"] == "latency58-quiet-reference-inventory-v1"
            and inventory["status"] == "pass" and inventory["reference_only"]
            and inventory["source_activity_and_physical_intervals_exact"] and inventory["source_bindings_unchanged"]
            and inventory_execution["actual_exit_code"] == 0 and not inventory_execution["timed_out"]
            and inventory_execution["source_bindings_unchanged"]
            and inventory_execution["plan_sha256"] == inventory["plan_sha256"], "Incomplete source inventory")
    verify_inputs(inventory)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve quiet-fidelity output")
    metric_check = check_window_metrics()
    before, began, rows = require_space(plan, 5_000_000), time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as log, ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(plan,)) as pool:
        futures = {pool.submit(score_track, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, row = future.result()
            require(index == futures[future] and index not in rows, "Unexpected completed primary track")
            rows[index] = row
            event = {"event": "track", "model": plan["model"]["kind"], "index": index,
                     "track": row["track"], "primary_replay_exact": True}
            log.write(json.dumps(event) + "\n")
            print(event, flush=True)
    require(set(rows) == set(range(14)), "Incomplete primary-panel coverage")
    from research.metrics import SOURCE_ORDER, mean_or_none
    summaries = {}
    for stem in SOURCE_ORDER:
        summaries[stem] = {}
        for bucket in QUIET_BUCKETS:
            per_track = {}
            for index in range(14):
                selected = [w for w in rows[index]["windows"] if w["stem"] == stem and w["bucket"] == bucket]
                if selected:
                    per_track[rows[index]["track"]] = {"windows": len(selected), "bands": {
                        band: {key: mean_or_none(w["bands"][band][key] for w in selected)
                               for key in selected[0]["bands"][band]} for band in ("full", "low_20_250")}}
            summaries[stem][bucket] = {"windows": sum(row["windows"] for row in per_track.values()),
                                      "tracks": len(per_track), "per_track": per_track}
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-quiet-wanted-evaluation-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "model": plan["model"], "reference_inventory": plan["reference_inventory"],
          "tracks": [rows[i] for i in range(14)], "quiet_summaries": summaries,
          "all_primary_track_scores_and_streams_exact": True, "all_1680_source_windows_exact": True,
          "analytic_quiet_metric_check": metric_check,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_before": before,
          "counted_bytes_after": require_space(plan, 0), "audio_exported": False,
          "cuda_initialized": False, "quality_selected": False, "human_listening_completed": False,
          "primary_protocol_changed": False, "confirmation_material_used": False,
          "limitations": ["Quiet source energy can include recording bleed or noise; this is not proof of audible desired content.",
                          "Signed reference projection can include correlated interference; read it with SDR and error/output levels.",
                          "Supplemental SDR has no activity exclusion or dB clamp and is not the primary SDR score.",
                          "Full-band reference level defines buckets; low-band measurements use those same windows.",
                          "Exactly zero full-band references have no full-band desired SDR or gain; filtering can spread neighboring reference energy into a window."]})
    print({"status": "pass", "model": plan["model"]["kind"], "source_windows": 1680}, flush=True)


if __name__ == "__main__":
    main()
