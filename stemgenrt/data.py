"""Portable manifests and the current deterministic training crop pipeline.

Use separate train/valid/test manifests and ``require_disjoint`` before training.
The scientific defaults retain a 0.85 vocal-anchor probability, data seed 60,
augmentation seed 20261102 and 16-example remix groups. Root weights come from
the supplied manifest: select the intended recorded sources explicitly (the
current two-corpus recipe gives each recorded corpus weight 0.5).
"""
from __future__ import annotations

import hashlib

from ._data.manifest import (
    ACTIVITY_POWER_FLOOR, SAMPLE_RATE, SOURCE_NAMES, FILE_NAMES, CorpusManifest,
    FrozenAudioFile, FrozenTrack, load_manifest, build_manifest, require_disjoint, file_sha256,
)
from ._data.sampling import CounterAddressedCropDataset, AbsoluteIndexSampler, worker_init
from ._data.duration import DurationWeightedCropDataset
from ._data.augmentation import (
    ADDRESS_RECIPE_VERSION, VERSION, AUGMENTATION, GROUP_SIZE, WARMUP_SAMPLES,
    SCORED_SAMPLES, CROP_SAMPLES, EXPANDED_SAMPLES, LongContextCropDataset,
    pitch_tempo_recipe, filter_string, transform_stem, transform_crop,
    remix_recipe, remix_batch,
)


def make_dataset(tracks, config, final_index):
    """Build the original/expanded addressed reads used by the active recipe.

    ``tracks`` may be a CorpusManifest (weights default to its recorded order)
    or a track sequence (``config['root_weights']`` is then required).
    Selected crops use the expanded read's own offset, as in the frozen recipe.
    """
    manifest = tracks if isinstance(tracks, CorpusManifest) else None
    if manifest is not None:
        if manifest.split != "train":
            raise ValueError("Training requires a train manifest")
        tracks = manifest.tracks
    weights = config.get("root_weights")
    if weights is None:
        if manifest is None:
            raise ValueError("Provide root_weights or a CorpusManifest")
        weights = manifest.root_weights
    kwargs = dict(root_weights=weights, seed=config.get("data_seed", 60),
                  vocal_active_probability=config.get("vocal_active_probability", .85),
                  final_sample_index=final_index)
    mode = config.get("track_sampling", "uniform")
    if mode not in ("uniform", "duration"):
        raise ValueError("track_sampling must be uniform or duration")
    def crops(frames):
        original = CounterAddressedCropDataset(tracks, crop_samples=frames, **kwargs)
        return DurationWeightedCropDataset(original) if mode == "duration" else original
    return LongContextCropDataset(
        crops(CROP_SAMPLES), crops(EXPANDED_SAMPLES),
        seed=config.get("seed", 20261102), ffmpeg=config.get("ffmpeg", "ffmpeg"))


def batch_recipes(config, first_index):
    return [{"sample_index": index, **pitch_tempo_recipe(seed=config.get("seed", 20261102), sample_index=index)}
            for index in range(first_index, first_index + config.get("batch_size", GROUP_SIZE))]


def audio_sha(mixture, targets):
    """Hash input tensor identity using the original journal encoding."""
    digest = hashlib.sha256()
    for name, value in (("mixture", mixture), ("targets", targets)):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def policy(track_sampling="uniform"):
    if track_sampling not in ("uniform", "duration"):
        raise ValueError("track_sampling must be uniform or duration")
    result = {"version": "latency58-four-second-score-pitch-tempo-remix-v1",
            "pitch_tempo_version": VERSION, "address_recipe_version": ADDRESS_RECIPE_VERSION,
            "selection_probability": .2, "semitones_uniform_inclusive": [-2, 2],
            "tempo_percent_normal_std": 5., "tempo_percent_clamp": [-12., 12.],
            "expanded_crop_samples": EXPANDED_SAMPLES, "returned_crop_samples": CROP_SAMPLES,
            "selected_crop_offset": "expanded addressed read; differs from original crop offset",
            "selected_mixture": "sum transformed stems; neutral transform preserves recorded mixture",
            "unselected_crop": "original addressed dataset bit exact", "source_order": list(SOURCE_NAMES),
            "transients": ["crisp", "smooth", "smooth", "smooth"],
            "formants": ["shifted", "shifted", "preserved", "shifted"],
            "encoding": "FFmpeg float32 stereo in and out", "after_pitch_tempo": AUGMENTATION,
            "remix_first_four": "unchanged by remix; may already have pitch/tempo augmentation",
            "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
            "complete_scored_one_second_windows": 4,
            "expanded_length_rule": "ceil(returned_crop_samples / .88 / 128) * 128"}
    if track_sampling == "duration":
        result.update(version="recorded301-duration-weighted-track-selection-v1",
            track_selection={"within_corpus_probability": "effective_frames / sum(effective_frames)",
                "integer_weights": True, "root_sampling_weights_changed": False,
                "original_and_expanded_use_same_track_weights": True,
                "source_reader_and_anchor_policy_changed": False,
                "addressed_track_and_offset_sequence_changed": True},
            production_recipe_selected=False)
    return result


def validate_checkpoint(config, provenance):
    """Require duration checkpoints to identify the sampler that produced them."""
    mode = config.get("track_sampling", "uniform")
    expected = policy(mode)
    saved = provenance.get("branch_memory_current_stage_augmentation")
    if mode == "duration" or (isinstance(saved, dict) and "track_selection" in saved):
        if saved != expected:
            raise ValueError("Checkpoint track sampling policy differs from its configuration")
