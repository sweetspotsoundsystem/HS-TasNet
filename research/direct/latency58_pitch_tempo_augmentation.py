"""Prospective CPU training augmentation; no model or evaluation dependency.

Selected crops use a longer addressed read, which changes their crop offset.
Unselected crops use the original dataset directly. This is inspired by the
Demucs v2 recipe, but uses float32 FFmpeg/Rubber Band instead of SoundStretch.
"""
from __future__ import annotations

import hashlib
import math
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

VERSION = "recorded301-addressed-float32-pitch-tempo-v1"
SAMPLE_RATE = 44100
CROP_SAMPLES = 132224
EXPANDED_SAMPLES = 150272
FFMPEG = "/usr/bin/ffmpeg"
SOURCE_NAMES = ("drums", "bass", "vocals", "other")


def recipe(*, seed, sample_index):
    if type(seed) is not int or type(sample_index) is not int or sample_index < 0:
        raise ValueError("Require an integer seed and nonnegative absolute sample index")
    material = f"{VERSION}:{seed}:{sample_index}".encode("ascii")
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


def transform_stem(audio, *, semitones, tempo, source_index):
    filt = filter_string(semitones=semitones, tempo=tempo, source_index=source_index)
    if (audio.device.type != "cpu" or audio.dtype != torch.float32
            or audio.requires_grad or audio.shape != (2, EXPANDED_SAMPLES)
            or not bool(torch.isfinite(audio).all())):
        raise ValueError("Require finite, fixed FP32 stereo training audio of the expanded length")
    if semitones == 0 and tempo == 1.:
        return audio[..., :CROP_SAMPLES].clone()
    encoded = np.ascontiguousarray(audio.numpy().T, dtype="<f4").tobytes()
    completed = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "pipe:0",
         "-af", filt, "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"],
        input=encoded, capture_output=True, check=True, timeout=20)
    if len(completed.stdout) % 8:
        raise RuntimeError("Malformed float32 stereo output from FFmpeg")
    output = np.frombuffer(completed.stdout, dtype="<f4").reshape(-1, 2)
    if len(output) < CROP_SAMPLES or not np.isfinite(output).all():
        raise RuntimeError("Short or non-finite pitch/tempo output")
    return torch.from_numpy(output[:CROP_SAMPLES].T.copy())


def transform_crop(mixture, targets, *, semitones, tempo):
    # Validate the pair before processing so malformed or non-finite reference
    # audio cannot disappear when the transformed mixture is reconstructed.
    if (mixture.shape != (2, EXPANDED_SAMPLES) or targets.shape != (4, 2, EXPANDED_SAMPLES)
            or any(value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad
                   or not bool(torch.isfinite(value).all()) for value in (mixture, targets))):
        raise ValueError("Require an aligned finite FP32 expanded mixture and four stems")
    filter_string(semitones=semitones, tempo=tempo, source_index=0)
    if semitones == 0 and tempo == 1.:
        return mixture[..., :CROP_SAMPLES].clone(), targets[..., :CROP_SAMPLES].clone()
    sources = torch.stack([transform_stem(stem, semitones=semitones, tempo=tempo, source_index=index)
                           for index, stem in enumerate(targets)])
    return sources.sum(0), sources


class PitchTempoCropDataset(Dataset):
    """Addressed choice between the original crop and an expanded transformed crop."""

    def __init__(self, original, expanded, *, seed):
        if (original.crop_samples != CROP_SAMPLES or expanded.crop_samples != EXPANDED_SAMPLES
                or type(seed) is not int):
            raise ValueError("Wrong original/expanded training geometry or seed")
        for attribute in ("seed", "final_sample_index", "vocal_active_probability", "root_ids",
                          "cumulative_weights", "tracks_by_root"):
            if getattr(original, attribute) != getattr(expanded, attribute):
                raise ValueError(f"Original and expanded dataset settings differ: {attribute}")
        if any(track.effective_frames < EXPANDED_SAMPLES
               for tracks in expanded.tracks_by_root.values() for track in tracks):
            raise ValueError("A selected training track is shorter than the expanded crop")
        self.original, self.expanded, self.seed = original, expanded, seed

    def __len__(self):
        return len(self.original)

    def __getitem__(self, sample_index):
        choice = recipe(seed=self.seed, sample_index=sample_index)
        if not choice["selected"]:
            return self.original[sample_index]
        mixture, targets = self.expanded[sample_index]
        return transform_crop(mixture, targets, semitones=choice["semitones"], tempo=choice["tempo"])
