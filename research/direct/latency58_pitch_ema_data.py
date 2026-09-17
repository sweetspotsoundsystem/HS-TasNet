"""Compose the qualified addressed pitch/tempo crops with ordinary B16 remixing."""
from research.direct.latency58_pitch_tempo_augmentation import (
    VERSION, CROP_SAMPLES, EXPANDED_SAMPLES, PitchTempoCropDataset, recipe)
from research.direct.latency58_remix_augmentation import AUGMENTATION
from research.direct.train_latency58 import state_sha256


def policy():
    return {"version": "latency58-pitch-tempo-then-ordinary-remix-v1",
            "pitch_tempo_version": VERSION, "selection_probability": .2,
            "semitones_uniform_inclusive": [-2, 2], "tempo_percent_normal_std": 5.,
            "tempo_percent_clamp": [-12., 12.], "expanded_crop_samples": EXPANDED_SAMPLES,
            "returned_crop_samples": CROP_SAMPLES,
            "selected_crop_offset": "expanded addressed read; differs from original crop offset",
            "selected_mixture": "sum transformed stems; neutral transform preserves recorded mixture",
            "unselected_crop": "original addressed dataset bit exact",
            "source_order": ["drums", "bass", "vocals", "other"],
            "transients": ["crisp", "smooth", "smooth", "smooth"],
            "formants": ["shifted", "shifted", "preserved", "shifted"],
            "encoding": "FFmpeg float32 stereo in and out",
            "after_pitch_tempo": AUGMENTATION,
            "remix_first_four": "unchanged by remix; may already have pitch/tempo augmentation"}


def dataset(production, tracks, config, final_index):
    kwargs = dict(root_weights=config["root_weights"], seed=config["data_seed"],
                  vocal_active_probability=config["vocal_active_probability"], final_sample_index=final_index)
    original = production.CounterAddressedCropDataset(tracks, crop_samples=CROP_SAMPLES, **kwargs)
    expanded = production.CounterAddressedCropDataset(tracks, crop_samples=EXPANDED_SAMPLES, **kwargs)
    return PitchTempoCropDataset(original, expanded, seed=config["seed"])


def batch_recipes(config, first_index):
    return [{"sample_index": index, **recipe(seed=config["seed"], sample_index=index)}
            for index in range(first_index, first_index + config["batch_size"])]


def audio_sha(mixture, targets):
    return state_sha256({"mixture": mixture, "targets": targets})
