"""Four-second training crops with the qualified pitch/tempo and B16 remixes.

Only crop geometry changes: 88,064 warmup samples plus 88,320 scored samples.
The existing addressed transform choices, FFmpeg filter settings and remix
mapping are retained. Selected expanded reads have different crop offsets.
"""
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

from research.direct.latency58_pitch_tempo_augmentation import (
    VERSION as RECIPE_VERSION, FFMPEG, SAMPLE_RATE, SOURCE_NAMES, recipe, filter_string)
from research.direct.latency58_pitch_ema_data import policy as parent_policy, audio_sha, batch_recipes

VERSION = "recorded301-addressed-long-context-float32-pitch-tempo-v1"
WARMUP_SAMPLES = 88064
SCORED_SAMPLES = 88320
CROP_SAMPLES = WARMUP_SAMPLES + SCORED_SAMPLES
EXPANDED_SAMPLES = 200448


def transform_stem(audio, *, semitones, tempo, source_index):
    filt = filter_string(semitones=semitones, tempo=tempo, source_index=source_index)
    if (audio.device.type != "cpu" or audio.dtype != torch.float32 or audio.requires_grad
            or audio.shape != (2, EXPANDED_SAMPLES) or not bool(torch.isfinite(audio).all())):
        raise ValueError("Require finite FP32 stereo audio of the expanded four-second crop length")
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
        raise RuntimeError("Short or nonfinite four-second pitch/tempo output")
    return torch.from_numpy(output[:CROP_SAMPLES].T.copy())


def transform_crop(mixture, targets, *, semitones, tempo):
    if (mixture.shape != (2, EXPANDED_SAMPLES) or targets.shape != (4, 2, EXPANDED_SAMPLES)
            or any(v.device.type != "cpu" or v.dtype != torch.float32 or v.requires_grad
                   or not bool(torch.isfinite(v).all()) for v in (mixture, targets))):
        raise ValueError("Require aligned finite expanded mixture and four source stems")
    filter_string(semitones=semitones, tempo=tempo, source_index=0)
    if semitones == 0 and tempo == 1.:
        return mixture[..., :CROP_SAMPLES].clone(), targets[..., :CROP_SAMPLES].clone()
    sources = torch.stack([transform_stem(stem, semitones=semitones, tempo=tempo, source_index=i)
                           for i, stem in enumerate(targets)])
    return sources.sum(0), sources


class LongContextCropDataset(Dataset):
    def __init__(self, original, expanded, *, seed):
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

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        choice = recipe(seed=self.seed, sample_index=index)
        if not choice["selected"]:
            return self.original[index]
        return transform_crop(*self.expanded[index], semitones=choice["semitones"], tempo=choice["tempo"])


def dataset(production, tracks, config, final_index, *, dataset_class=None):
    crop_type = production.CounterAddressedCropDataset if dataset_class is None else dataset_class
    kwargs = dict(root_weights=config["root_weights"], seed=config["data_seed"],
                  vocal_active_probability=config["vocal_active_probability"], final_sample_index=final_index)
    return LongContextCropDataset(crop_type(tracks, crop_samples=CROP_SAMPLES, **kwargs),
                                  crop_type(tracks, crop_samples=EXPANDED_SAMPLES, **kwargs), seed=config["seed"])


def policy():
    return {**parent_policy(), "version": "latency58-two-second-score-pitch-tempo-remix-v1",
            "pitch_tempo_version": VERSION, "address_recipe_version": RECIPE_VERSION,
            "expanded_crop_samples": EXPANDED_SAMPLES, "returned_crop_samples": CROP_SAMPLES,
            "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
            "complete_scored_one_second_windows": 2,
            "expanded_length_rule": "ceil(returned_crop_samples / .88 / 128) * 128"}
