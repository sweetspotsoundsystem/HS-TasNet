"""Current four-second pitch/tempo and sixteen-example remix recipe.

The RNG namespace is retained byte-for-byte from the qualified recipe so
absolute sample addresses preserve their transform choices. FFmpeg and its
Rubber Band build must also match when reproducing exact transformed audio.
"""
from __future__ import annotations

import hashlib
import math
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import SAMPLE_RATE, SOURCE_NAMES

ADDRESS_RECIPE_VERSION = "recorded301-addressed-float32-pitch-tempo-v1"
VERSION = "recorded301-addressed-four-second-score-float32-pitch-tempo-v1"
AUGMENTATION = "quarter_original_quarter_same_crop_half_cross_crop_gain3db_channel_swap_v1"
WARMUP_SAMPLES = 88064
SCORED_SAMPLES = 176512
CROP_SAMPLES = WARMUP_SAMPLES + SCORED_SAMPLES
EXPANDED_SAMPLES = 300672
GROUP_SIZE = 16


def pitch_tempo_recipe(*, seed, sample_index):
    if type(seed) is not int or type(sample_index) is not int or sample_index < 0:
        raise ValueError("Require an integer seed and nonnegative absolute sample index")
    material = f"{ADDRESS_RECIPE_VERSION}:{seed}:{sample_index}".encode("ascii")
    rng = random.Random(int.from_bytes(hashlib.blake2b(material, digest_size=16).digest(), "big"))
    selected = rng.random() < .2
    semitones = rng.randint(-2, 2) if selected else 0
    tempo_percent = max(-12., min(12., rng.gauss(0., 5.))) if selected else 0.
    return {"selected": selected, "semitones": semitones, "tempo": 1 + tempo_percent / 100}


def filter_string(*, semitones, tempo, source_index):
    if (type(semitones) is not int or not -2 <= semitones <= 2
            or not math.isfinite(tempo) or not .88 <= tempo <= 1.12
            or type(source_index) is not int or not 0 <= source_index < 4):
        raise ValueError("Invalid source or pitch/tempo setting")
    # R2 transient phase resets detune sustained test tones. Retain them only
    # for drums; the spectral-envelope preservation setting is vocal-specific.
    return (f"rubberband=tempo={tempo:.12f}:pitch={2 ** (semitones / 12):.12f}"
            f":pitchq=quality:channels=together:transients={'crisp' if source_index == 0 else 'smooth'}"
            f":formant={'preserved' if source_index == 2 else 'shifted'}")


def transform_stem(audio, *, semitones, tempo, source_index, ffmpeg="ffmpeg"):
    filt = filter_string(semitones=semitones, tempo=tempo, source_index=source_index)
    if (audio.device.type != "cpu" or audio.dtype != torch.float32 or audio.requires_grad
            or audio.shape != (2, EXPANDED_SAMPLES) or not bool(torch.isfinite(audio).all())):
        raise ValueError("Require finite FP32 stereo audio of the expanded six-second crop length")
    if semitones == 0 and tempo == 1.:
        return audio[..., :CROP_SAMPLES].clone()
    encoded = np.ascontiguousarray(audio.numpy().T, dtype="<f4").tobytes()
    completed = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "pipe:0",
         "-af", filt, "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"],
        input=encoded, capture_output=True, check=True, timeout=20)
    if len(completed.stdout) % 8:
        raise RuntimeError("Malformed float32 stereo output from FFmpeg")
    output = np.frombuffer(completed.stdout, dtype="<f4").reshape(-1, 2)
    if len(output) < CROP_SAMPLES or not np.isfinite(output).all():
        raise RuntimeError("Short or nonfinite six-second pitch/tempo output")
    return torch.from_numpy(output[:CROP_SAMPLES].T.copy())


def transform_crop(mixture, targets, *, semitones, tempo, ffmpeg="ffmpeg"):
    if (mixture.shape != (2, EXPANDED_SAMPLES) or targets.shape != (4, 2, EXPANDED_SAMPLES)
            or any(v.device.type != "cpu" or v.dtype != torch.float32 or v.requires_grad
                   or not bool(torch.isfinite(v).all()) for v in (mixture, targets))):
        raise ValueError("Require aligned finite expanded mixture and four source stems")
    filter_string(semitones=semitones, tempo=tempo, source_index=0)
    if semitones == 0 and tempo == 1.:
        return mixture[..., :CROP_SAMPLES].clone(), targets[..., :CROP_SAMPLES].clone()
    sources = torch.stack([transform_stem(stem, semitones=semitones, tempo=tempo, source_index=i, ffmpeg=ffmpeg)
                           for i, stem in enumerate(targets)])
    return sources.sum(0), sources


class LongContextCropDataset(Dataset):
    def __init__(self, original, expanded, *, seed, ffmpeg="ffmpeg"):
        if (original.crop_samples != CROP_SAMPLES or expanded.crop_samples != EXPANDED_SAMPLES
                or type(seed) is not int):
            raise ValueError("Wrong long-context crop geometry or seed")
        for key in ("seed", "final_sample_index", "vocal_active_probability", "root_ids",
                    "cumulative_weights", "tracks_by_root"):
            if getattr(original, key) != getattr(expanded, key):
                raise ValueError("Original and expanded datasets differ: " + key)
        if any(t.effective_frames < EXPANDED_SAMPLES for tracks in expanded.tracks_by_root.values() for t in tracks):
            raise ValueError("A selected training track is shorter than the expanded crop")
        self.original, self.expanded, self.seed = original, expanded, seed
        self.crop_samples = CROP_SAMPLES
        self.ffmpeg = ffmpeg

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        choice = pitch_tempo_recipe(seed=self.seed, sample_index=index)
        if not choice["selected"]:
            return self.original[index]
        return transform_crop(*self.expanded[index], semitones=choice["semitones"], tempo=choice["tempo"], ffmpeg=self.ffmpeg)


def remix_recipe(*, seed, first_sample_index):
    if type(seed) is not int or type(first_sample_index) is not int or first_sample_index < 0 or first_sample_index % GROUP_SIZE:
        raise ValueError("Use an aligned sixteen-crop counter-address group")
    generator = torch.Generator().manual_seed(seed * 1_000_003 + first_sample_index)
    # Distinct nonzero cyclic shifts ensure that a remixed output uses four
    # different crop addresses, each different from that output's own address.
    shifts = (torch.randperm(GROUP_SIZE - 1, generator=generator)[:4] + 1).tolist()
    values = torch.rand((GROUP_SIZE, 10), generator=generator).tolist()
    indices, swaps, factors = [], [], []
    for row, randoms in enumerate(values):
        changed, remixed = row >= 4, row >= 8
        indices.append([(row + shifts[stem]) % GROUP_SIZE if remixed else row for stem in range(4)])
        swaps.append([changed and randoms[4 + stem] < .5 for stem in range(4)])
        polarity = -1. if randoms[8] < .5 else 1.
        factors.append([polarity * math.pow(10., (6 * randoms[stem] - 3) / 20) if changed else 1.
                        for stem in range(4)])
    return {"source_indices": torch.tensor(indices, dtype=torch.int64),
            "swap_channels": torch.tensor(swaps, dtype=torch.bool),
            "factors": torch.tensor(factors, dtype=torch.float32),
            "changed": torch.arange(GROUP_SIZE) >= 4,
            "cross_crop": torch.arange(GROUP_SIZE) >= 8}


def remix_batch(mixture, targets, *, seed, first_sample_index):
    if mixture.device.type != "cpu" or targets.device.type != "cpu" or mixture.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError("Transform aligned FP32 training audio on CPU")
    if targets.ndim != 4 or targets.shape[:3] != (GROUP_SIZE, 4, 2) or mixture.shape != (GROUP_SIZE, 2, targets.shape[-1]):
        raise ValueError("Require sixteen aligned stereo mixtures and four source stems")
    if mixture.requires_grad or targets.requires_grad:
        raise ValueError("Training references must remain fixed")
    selected = remix_recipe(seed=seed, first_sample_index=first_sample_index)
    stems = torch.arange(4)[None, :].expand(GROUP_SIZE, -1)
    sources = targets[selected["source_indices"], stems]
    sources = torch.where(selected["swap_channels"][:, :, None, None], sources.flip(2), sources)
    sources = sources * selected["factors"][:, :, None, None]
    rendered = torch.where(selected["changed"][:, None, None], sources.sum(1), mixture)
    return rendered, sources, selected["changed"], selected["factors"]
