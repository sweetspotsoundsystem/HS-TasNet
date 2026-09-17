"""Qualify four-second pitch/tempo crops on CPU before longer-context branch training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256
from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.latency58_recorded301_data import ROOT_WEIGHTS, select_tracks, selection_contract
from research.direct.latency58_long_context_data import (
    VERSION, CROP_SAMPLES, EXPANDED_SAMPLES, FFMPEG, SOURCE_NAMES,
    LongContextCropDataset as PitchTempoCropDataset, recipe, transform_crop, transform_stem)
from research.direct.latency58_remix_augmentation import augment

sys.path.insert(0, str(PRODUCTION))
import train_production as production


class TracedCrops(production.CounterAddressedCropDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_files = {}

    def _read(self, file, *, offset, frames):
        self.read_files[str(file.path)] = file.sha256
        return super()._read(file, offset=offset, frames=frames)


def audio_sha(mixture, targets):
    return state_sha256({"mixture": mixture, "targets": targets})


def synthetic_check():
    t = torch.arange(EXPANDED_SAMPLES, dtype=torch.float64) / 44100
    tone = (.1 * torch.sin(2 * torch.pi * 440 * t)).float().repeat(2, 1)
    fingerprint = state_sha256({"tone": tone})
    rows = []
    for source in (1, 2, 3):
        for semitones, tempo in ((0, 1.12), (2, 1.12), (-2, .88)):
            output = transform_stem(tone, semitones=semitones, tempo=tempo, source_index=source)
            for offset in (44100, CROP_SAMPLES - 44100):
                block = output[0, offset:offset + 44100].double().numpy()
                spectrum = np.abs(np.fft.rfft(block * np.hanning(len(block))))
                peak = int(spectrum.argmax())
                left, center, right = np.log(np.maximum(spectrum[peak - 1:peak + 2], 1e-100))
                actual_hz = peak + .5 * (left - right) / (left - 2 * center + right)
                expected_hz = 440 * 2 ** (semitones / 12)
                error_cents = float(1200 * np.log2(actual_hz / expected_hz))
                require(abs(error_cents) < 1 and torch.equal(output[0], output[1]),
                        "Sustained tone pitch or identical stereo channels changed unexpectedly")
                rows.append({"source": SOURCE_NAMES[source], "semitones": semitones, "tempo": tempo,
                             "analysis_offset_samples": offset, "expected_hz": expected_hz,
                             "actual_hz": float(actual_hz), "error_cents": error_cents})
    truth = torch.stack([tone, tone * .5, torch.zeros_like(tone), tone * 2])
    mixture = truth.sum(0) + .000123
    before = audio_sha(mixture, truth)
    neutral = transform_crop(mixture, truth, semitones=0, tempo=1.)
    require(torch.equal(neutral[0], mixture[..., :CROP_SAMPLES])
            and torch.equal(neutral[1], truth[..., :CROP_SAMPLES]), "Neutral recipe changed recorded audio")
    transformed = transform_crop(mixture, truth, semitones=-2, tempo=.88)
    repeated = transform_crop(mixture, truth, semitones=-2, tempo=.88)
    require(all(torch.equal(a, b) for a, b in zip(transformed, repeated, strict=True))
            and torch.equal(transformed[0], transformed[1].sum(0))
            and torch.count_nonzero(transformed[1][2]) == 0,
            "Float transform replay, source closure or silent vocal failed")
    loud = transform_stem(tone * 20, semitones=2, tempo=1.12, source_index=1)
    require(float(loud.abs().max()) > 1.5, "Float output appears clipped to PCM16 range")
    require(before == audio_sha(mixture, truth) and fingerprint == state_sha256({"tone": tone}),
            "Transform mutated reference audio")
    rng_before = (random.getstate(), torch.get_rng_state())
    choices = [recipe(seed=20261029, sample_index=i) for i in range(10000)]
    require(choices == [recipe(seed=20261029, sample_index=i) for i in range(10000)]
            and random.getstate() == rng_before[0] and torch.equal(torch.get_rng_state(), rng_before[1]),
            "Address recipe depends on or changes global RNG")
    return {"status": "pass", "sustained_tone_checks": rows, "pitch_tolerance_cents": 1,
            "neutral_pair_bit_exact": True, "transform_replay_bit_exact": True,
            "source_closure_bit_exact": True, "silent_vocal_exact": True, "input_unmodified": True,
            "float_output_exceeds_one": True, "loud_peak": float(loud.abs().max()),
            "global_rng_unchanged_by_recipe": True, "selected_count_of_10000": sum(c["selected"] for c in choices),
            "limitation": "Sustained tone accuracy covers Bass/Vocal/Other smooth mode. Drum crisp mode retains transient resets; percussion fidelity and perceptual quality are not established by this fixture."}


def load_batches(dataset, *, workers, first, stop, seed, original_hashes=None):
    kwargs = ({"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {})
    loader = DataLoader(dataset, batch_size=16, num_workers=workers,
                        sampler=production.AbsoluteIndexSampler(first, stop),
                        worker_init_fn=production.worker_init,
                        generator=torch.Generator().manual_seed(seed + 1), **kwargs)
    started = time.monotonic()
    iterator = iter(loader)
    rows, waits = [], []
    for batch in range((stop - first) // 16):
        waiting = time.monotonic()
        mixture, targets = next(iterator)
        waits.append(time.monotonic() - waiting)
        require(mixture.shape == (16, 2, CROP_SAMPLES) and targets.shape == (16, 4, 2, CROP_SAMPLES)
                and bool(torch.isfinite(mixture).all()) and bool(torch.isfinite(targets).all()),
                "Recorded batch is malformed")
        index = first + 16 * batch
        if original_hashes is not None:
            for j in range(16):
                choice = recipe(seed=seed, sample_index=index + j)
                if choice["selected"]:
                    require(torch.equal(mixture[j], targets[j].sum(0)), "Selected recorded crop lost source closure")
                else:
                    require(audio_sha(mixture[j], targets[j]) == original_hashes[index + j],
                            "Unselected recorded crop changed")
        remixed = augment(mixture, targets, seed=seed, first_sample_index=index)
        require(torch.equal(remixed[0][:4], mixture[:4]) and torch.equal(remixed[1][:4], targets[:4])
                and torch.equal(remixed[0][4:], remixed[1][4:].sum(1)), "Existing remix composition changed")
        rows.append({"first_index": index, "input_sha256": audio_sha(mixture, targets),
                     "after_remix_sha256": audio_sha(remixed[0], remixed[1])})
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Extra crop batch in bounded qualification")
    return {"workers": workers, "elapsed_seconds_including_startup_and_shutdown": time.monotonic() - started,
            "batch_wait_seconds": waits, "batches": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_directory.resolve()
    require(Path.cwd() == ROOT and out.is_relative_to(PHASE) and not out.exists()
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require a fresh CPU-only diagnostic directory and one thread per process")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    policy_path = PHASE / "branch-pitch-ema-002/plan.json"
    policy = read(policy_path)
    counted = require_space(policy, 450_000_000)
    source_paths = [Path(__file__), ROOT / "research/direct/latency58_long_context_data.py",
                    ROOT / "research/direct/latency58_pitch_ema_data.py",
                    ROOT / "research/direct/latency58_pitch_tempo_augmentation.py",
                    ROOT / "research/direct/latency58_remix_augmentation.py", ROOT / "research/direct/latency58_recorded301_data.py",
                    ROOT / "research/direct/train_latency58.py", ROOT / "research/direct/run_latency58_quality.py",
                    ROOT / "research/direct/latency58_sdr_checkpoint.py", PRODUCTION / "train_production.py",
                    PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json",
                    ROOT / "research/manifests/valid.json", policy_path, Path(FFMPEG),
                    Path("/lib/x86_64-linux-gnu/librubberband.so.2").resolve(),
                    Path("/lib/x86_64-linux-gnu/libavfilter.so.9").resolve()]
    bindings = {str(path): sha(path) for path in source_paths}
    started = time.monotonic()
    out.mkdir()
    selection = selection_contract()
    config = read(PRODUCTION / "full_config.json")
    _, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
                            expected_file_sha256=selection["source_manifest_sha256"], config=config)
    tracks = select_tracks(tracks, selection)
    require(tuple(production.SOURCE_NAMES) == SOURCE_NAMES, "Source order changed")
    first, stop, seed = 4028000, 4028064, 20261029
    kwargs = dict(root_weights=ROOT_WEIGHTS, seed=config["seed"],
                  vocal_active_probability=config["sampling"]["vocal_active_probability"], final_sample_index=stop)
    original = TracedCrops(tracks, crop_samples=CROP_SAMPLES, **kwargs)
    expanded = TracedCrops(tracks, crop_samples=EXPANDED_SAMPLES, **kwargs)
    dataset = PitchTempoCropDataset(original, expanded, seed=seed)
    original_hashes, input_rows = {}, []
    for index in range(first, stop):
        original_hashes[index] = audio_sha(*original[index])
        choice = recipe(seed=seed, sample_index=index)
        input_rows.append({"index": index, **choice, "original_sha256": original_hashes[index],
                           "expanded_sha256": audio_sha(*expanded[index]) if choice["selected"] else None})
    audio_files = {**original.read_files, **expanded.read_files}
    require(all(sha(path) == digest for path, digest in audio_files.items()), "Training audio differs from sealed manifest")
    bindings.update(audio_files)
    snapshot = {"schema": "latency58-branch-long-context-pitch-tempo-cpu-inputs-v1", "version": VERSION, "selection": selection,
                "first_sample_index": first, "stop_sample_index": stop, "augmentation_seed": seed,
                "source_bindings": bindings, "input_rows": input_rows,
                "artifact_policy": {key: policy[key] for key in ("counted_roots", "stop_counted_bytes", "outside_roots_reservation_bytes")}}
    write(out / "inputs.json", snapshot)
    print(json.dumps({"event": "inputs_authenticated", "training_files": len(audio_files), "crops": stop - first}), flush=True)
    synthetic = synthetic_check()
    print(json.dumps({"event": "synthetic_pass", "maximum_pitch_error_cents": max(abs(r["error_cents"]) for r in synthetic["sustained_tone_checks"])}), flush=True)
    baseline = load_batches(original, workers=2, first=first, stop=stop, seed=seed)
    runs = []
    for workers in (0, 2):
        run = load_batches(dataset, workers=workers, first=first, stop=stop, seed=seed, original_hashes=original_hashes)
        runs.append(run)
        print(json.dumps({"event": "recorded_loader_complete", "workers": workers,
                          "seconds": run["elapsed_seconds_including_startup_and_shutdown"]}), flush=True)
    require(runs[0]["batches"] == runs[1]["batches"], "Worker count changed actual audio or remixed batches")
    require(all(sha(path) == digest for path, digest in bindings.items()) and not torch.cuda.is_initialized(),
            "Diagnostic inputs changed or CUDA was initialized")
    after = require_space(policy, 450_000_000)
    result = {"status": "pass", "version": VERSION, "inputs_sha256": sha(out / "inputs.json"),
              "source_bindings_unchanged": True, "synthetic": synthetic, "baseline_loader": baseline,
              "augmented_loaders": runs, "training_crop_count": stop - first,
              "selected_training_crops": sum(row["selected"] for row in input_rows),
              "unselected_recorded_crops_bit_exact": True, "selected_mixture_equals_stems_bit_exact": True,
              "worker_count_replay_and_composed_remix_bit_exact": True, "validation_audio_decoded": False,
              "gpu_used": False, "quality_measured": False, "checkpoint_written": False,
              "elapsed_seconds": time.monotonic() - started, "counted_bytes_before": counted,
              "forecast_bytes": after + 450_000_000 + policy["outside_roots_reservation_bytes"],
              "versions": {"torch": torch.__version__, "numpy": np.__version__,
                           "ffmpeg": subprocess.check_output([FFMPEG, "-version"], text=True).splitlines()[0]},
              "limitations": "64 four-second training crops, four B16 batches per loader. Timings include startup/shutdown; no steady-state GPU throughput or SDR gain is established. Selected crops use longer reads with different offsets; unselected crops retain original offsets. Drum transient fidelity, vocal formant quality, and audible quality require separate assessment. This prototype is not connected to a trainer."}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "selected_training_crops", "training_crop_count", "elapsed_seconds", "forecast_bytes", "quality_measured")}), flush=True)


if __name__ == "__main__":
    main()
