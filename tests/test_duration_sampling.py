"""Duration weighting, independent audio addressing and spawned-worker replay."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from stemgenrt.data import (CounterAddressedCropDataset, DurationWeightedCropDataset,
    SAMPLE_RATE, SOURCE_NAMES, EXPANDED_SAMPLES, audio_sha, build_manifest,
    load_manifest, make_dataset, policy, validate_checkpoint)
from test_data import write_song


@pytest.fixture(scope="module")
def duration_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("duration-audio")
    roots = {"a": root / "a", "b": root / "b"}
    write_song(roots["a"] / "short", frequency=123, frames=8 * SAMPLE_RATE)
    write_song(roots["a"] / "long", frequency=147, frames=16 * SAMPLE_RATE)
    write_song(roots["b"] / "third", frequency=174, frames=9 * SAMPLE_RATE)
    manifest = build_manifest(roots, root / "manifest.json", weights={"a": .5, "b": .5})
    return load_manifest(manifest, crop_samples=EXPANDED_SAMPLES)


def crops(corpus, final=4132032):
    return DurationWeightedCropDataset(CounterAddressedCropDataset(corpus.tracks,
        root_weights=corpus.root_weights, seed=60, crop_samples=1024,
        vocal_active_probability=.85, final_sample_index=final))


def test_integer_duration_probability_mass():
    base = SimpleNamespace(seed=60, crop_samples=1, vocal_active_probability=0.,
        final_sample_index=10, root_ids=("a",), cumulative_weights=(1.,),
        tracks_by_root={"a": [SimpleNamespace(effective_frames=n) for n in (3, 7, 2)]})
    dataset = DurationWeightedCropDataset(base)
    class Ticket:
        def __init__(self, value):
            self.value = value
        def randrange(self, stop):
            assert stop == 12
            return self.value
    choices = [dataset._weighted_track_index("a", Ticket(i)) for i in range(12)]
    assert choices == [0] * 3 + [1] * 7 + [2] * 2


def test_duration_crops_match_independently_decoded_addresses(duration_corpus):
    dataset = crops(duration_corpus)
    for index in (0, 1, 7, 17, 4132000, 4132009, 4132015):
        rng = dataset.original._rng(index)
        root = "a" if rng.random() < .5 else "b"
        tracks = sorted((t for t in duration_corpus.tracks if t.root_id == root), key=lambda t: t.track_id)
        ticket = rng.randrange(sum(t.effective_frames for t in tracks))
        for track in tracks:
            if ticket < track.effective_frames:
                break
            ticket -= track.effective_frames
        maximum = track.effective_frames - 1024
        if track.vocal_active_seconds and rng.random() < .85:
            anchor = track.vocal_active_seconds[rng.randrange(len(track.vocal_active_seconds))] * SAMPLE_RATE
            offset = max(0, min(anchor - rng.randrange(1024), maximum))
        else:
            offset = rng.randrange(maximum + 1)
        mix, targets = dataset[index]
        for name, actual in (("mixture", mix), *zip(SOURCE_NAMES, targets)):
            expected, _ = sf.read(track.files[name].path, start=offset, frames=1024,
                                  dtype="float32", always_2d=True)
            assert np.array_equal(actual.numpy(), expected.T)
        assert not torch.equal(mix, targets.sum(0))


def test_duration_spawned_workers_and_restart_are_exact(duration_corpus):
    dataset = crops(duration_corpus)
    addresses = [0, 17, 4132000, 4132015]
    expected = [audio_sha(*dataset[index]) for index in addresses]
    loader = DataLoader(dataset, batch_size=None, sampler=addresses, num_workers=2,
                        multiprocessing_context="spawn", timeout=30)
    assert [audio_sha(*pair) for pair in loader] == expected
    resumed = crops(duration_corpus, final=4132016)
    assert [audio_sha(*resumed[index]) for index in addresses[2:]] == expected[2:]


def test_full_pipeline_weights_and_default_uniform_compatibility(duration_corpus):
    duration = make_dataset(duration_corpus, {"track_sampling": "duration"}, 4132032)
    assert duration.original.cumulative_frames == duration.expanded.cumulative_frames
    assert audio_sha(*duration[4132000]) == audio_sha(*duration.original[4132000])
    default = make_dataset(duration_corpus, {}, 4132032)
    explicit = make_dataset(duration_corpus, {"track_sampling": "uniform"}, 4132032)
    assert type(default.original) is CounterAddressedCropDataset
    assert audio_sha(*default[4132000]) == audio_sha(*explicit[4132000])
    with pytest.raises(ValueError, match="track_sampling"):
        make_dataset(duration_corpus, {"track_sampling": "unknown"}, 4132032)


def test_checkpoint_sampler_provenance_is_required_and_consistent():
    key = "branch_memory_current_stage_augmentation"
    validate_checkpoint({}, {})  # Historical uniform checkpoints remain readable.
    validate_checkpoint({"track_sampling": "duration"}, {key: policy("duration")})
    for config, saved in [({"track_sampling": "duration"}, {}),
                          ({"track_sampling": "duration"}, {key: policy()}),
                          ({}, {key: policy("duration")})]:
        with pytest.raises(ValueError, match="track sampling policy"):
            validate_checkpoint(config, saved)
