"""Synthetic-audio checks for addressing, split integrity and augmentation."""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import socket
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from hs_tasnet.data import (
    SAMPLE_RATE, SOURCE_NAMES, CROP_SAMPLES, EXPANDED_SAMPLES, WARMUP_SAMPLES, SCORED_SAMPLES,
    AbsoluteIndexSampler, CounterAddressedCropDataset, audio_sha, batch_recipes,
    build_manifest, load_manifest, make_dataset, pitch_tempo_recipe, remix_batch,
    remix_recipe, require_disjoint, transform_crop, transform_stem, worker_init,
)


def write_song(folder, *, frequency=110., frames=8 * SAMPLE_RATE):
    folder.mkdir(parents=True)
    time = np.arange(frames) / SAMPLE_RATE
    stems = []
    for index, name in enumerate(SOURCE_NAMES):
        mono = (.08 * np.sin(2 * np.pi * (frequency + index * 55) * time)).astype(np.float32)
        stereo = np.stack((mono, mono * np.float32(.75)), axis=1)
        stems.append(stereo)
        sf.write(folder / (name + ".wav"), stereo, SAMPLE_RATE, subtype="FLOAT")
    mixture = np.stack(stems).sum(0, dtype=np.float32) + np.float32(.000123)
    sf.write(folder / "mixture.wav", mixture, SAMPLE_RATE, subtype="FLOAT")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-corpus")
    roots = {"a": root / "audio-a", "b": root / "audio-b"}
    for index, folder in enumerate(roots.values()):
        write_song(folder / "song", frequency=110. + index * 23.)
    path = build_manifest(roots, root / "manifest.json", weights={"a": .5, "b": .5})
    return load_manifest(path, crop_samples=EXPANDED_SAMPLES)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dataset(corpus, *, final=4132032, crop=CROP_SAMPLES):
    return CounterAddressedCropDataset(corpus.tracks, root_weights=corpus.root_weights, seed=60,
                                       crop_samples=crop, vocal_active_probability=.85,
                                       final_sample_index=final)


def test_manifest_is_relative_and_preserves_order_and_activity(corpus):
    stored = json.loads(corpus.path.read_text())
    assert all(not Path(root["path"]).is_absolute() for root in stored["roots"])
    assert corpus.root_ids == ("a", "b") and corpus.root_weights == {"a": .5, "b": .5}
    assert all(track.vocal_active_seconds == tuple(range(8)) for track in corpus.tracks)
    assert WARMUP_SAMPLES + SCORED_SAMPLES == CROP_SAMPLES == 264576
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(corpus.path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="split"):
        load_manifest(corpus.path, expected_split="valid")


@pytest.mark.parametrize("index,root,offset", [
    (0, "b", 0), (1, "b", 43394), (17, "a", 82314),
    (4132000, "b", 4832), (4132015, "b", 41286),
])
def test_crop_addresses_match_qualified_sampler(corpus, index, root, offset):
    # Frozen vectors from the original seed-60 counter-addressed sampler.
    # Independently decode the expected physical samples, including the real
    # mixture/stem mismatch, so a self-consistent offset error cannot pass.
    track = next(track for track in corpus.tracks if track.root_id == root)
    actual_mix, actual_stems = dataset(corpus)[index]
    for name, actual in (("mixture", actual_mix), *zip(SOURCE_NAMES, actual_stems)):
        expected, _ = sf.read(track.files[name].path, start=offset, frames=CROP_SAMPLES,
                              dtype="float32", always_2d=True)
        assert np.array_equal(actual.numpy(), expected.T)
    assert not torch.equal(actual_mix, actual_stems.sum(0))


def test_addressing_is_independent_of_resume_stop_and_worker_rng(corpus):
    first = dataset(corpus, final=18, crop=1024)
    resumed = dataset(corpus, final=100, crop=1024)
    expected = [audio_sha(*first[i]) for i in (0, 1, 17)]
    assert [audio_sha(*resumed[i]) for i in (0, 1, 17)] == expected
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for worker_id in (0, 1, 7):
                torch.manual_seed(97 + worker_id)
                worker_init(worker_id)
                assert [audio_sha(*resumed[i]) for i in (0, 1, 17)] == expected
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert list(AbsoluteIndexSampler(17, 20)) == [17, 18, 19]


def test_two_worker_loader_keeps_absolute_addresses(corpus):
    # PyTorch shares CPU tensor handles over a local socket. Some restricted
    # runners disable this OS capability before a worker can return any data.
    try:
        with socket.socket(socket.AF_UNIX):
            pass
    except PermissionError:
        pytest.skip("Runner disables local sockets required by PyTorch tensor sharing")
    crops = dataset(corpus, final=18, crop=1024)
    expected = [audio_sha(*crops[i]) for i in (0, 1, 17)]
    loader = DataLoader(crops, batch_size=None, sampler=[0, 1, 17], num_workers=2,
                        worker_init_fn=worker_init, multiprocessing_context="spawn",
                        generator=torch.Generator().manual_seed(7), timeout=30)
    assert [audio_sha(*pair) for pair in loader] == expected


def test_pitch_choices_and_unselected_crop_are_unchanged(corpus):
    config = {"seed": 20261102, "data_seed": 60, "batch_size": 16,
              "vocal_active_probability": .85}
    choices = batch_recipes(config, 4132000)
    assert [row["sample_index"] for row in choices if row["selected"]] == [4132009]
    assert pitch_tempo_recipe(seed=config["seed"], sample_index=4132009) == {
        "selected": True, "semitones": -1, "tempo": 1.0321744280994778}
    augmented = make_dataset(corpus, config, 4132032)
    assert audio_sha(*augmented[4132000]) == audio_sha(*dataset(corpus)[4132000])
    assert augmented.crop_samples == CROP_SAMPLES


def test_remix_mapping_preserves_originals_and_source_alignment():
    truth = torch.arange(16 * 4 * 2 * 32, dtype=torch.float32).reshape(16, 4, 2, 32) / 10000
    mixture = truth.sum(1) + .000123
    original = audio_sha(mixture, truth)
    options = {"seed": 20261102, "first_sample_index": 4132000}
    mixed, stems, changed, factors = remix_batch(mixture, truth, **options)
    recipe = remix_recipe(**options)
    assert torch.equal(mixed[:4], mixture[:4]) and torch.equal(stems[:4], truth[:4])
    assert torch.equal(mixed[4:], stems[4:].sum(1))
    for row in range(16):
        for source in range(4):
            expected = truth[recipe["source_indices"][row, source], source]
            if recipe["swap_channels"][row, source]:
                expected = expected.flip(0)
            assert torch.equal(stems[row, source], expected * factors[row, source])
        if row >= 8:
            indices = recipe["source_indices"][row].tolist()
            assert len(set(indices)) == 4 and row not in indices
    assert changed.tolist() == [False] * 4 + [True] * 12
    assert audio_sha(mixture, truth) == original
    assert audio_sha(mixed, stems) == audio_sha(*remix_batch(mixture, truth, **options)[:2])


def test_manifest_excludes_held_out_paths_and_bytes(tmp_path):
    write_song(tmp_path / "held-out" / "song", frames=2048)
    write_song(tmp_path / "train" / "keep", frequency=123, frames=2048)
    shutil.copytree(tmp_path / "held-out" / "song", tmp_path / "train" / "renamed-copy")
    valid_path = build_manifest({"valid": tmp_path / "held-out"}, tmp_path / "valid.json", split="valid")
    train_path = build_manifest({"train": tmp_path / "train"}, tmp_path / "train.json",
                                exclude_manifests=[valid_path])
    train, valid = load_manifest(train_path), load_manifest(valid_path)
    assert [t.name for t in train.tracks] == ["keep"]
    require_disjoint(train, valid)
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint(valid, valid)
    with pytest.raises(ValueError, match="train manifest"):
        make_dataset(valid, {}, 1)


def test_manifest_rejects_audio_tampering_and_escape(tmp_path):
    write_song(tmp_path / "audio" / "song", frames=2048)
    path = build_manifest({"train": tmp_path / "audio"}, tmp_path / "manifest.json")
    with (tmp_path / "audio" / "song" / "drums.wav").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(path)
    # An escape must be rejected even when the referenced bytes are genuine.
    data = json.loads(path.read_text())
    outside = tmp_path / "outside.wav"
    shutil.copyfile(tmp_path / "audio" / "song" / "mixture.wav", outside)
    data["tracks"][0]["files"]["mixture"]["path"] = "../outside.wav"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(path)


def test_neutral_transform_preserves_recorded_mixture():
    targets = torch.zeros(4, 2, EXPANDED_SAMPLES)
    mixture = torch.full((2, EXPANDED_SAMPLES), .000123)
    mixed, transformed = transform_crop(mixture, targets, semitones=0, tempo=1., ffmpeg="not-needed")
    assert torch.equal(mixed, mixture[..., :CROP_SAMPLES])
    assert torch.equal(transformed, targets[..., :CROP_SAMPLES])


def test_ffmpeg_pitch_and_tempo_keep_four_scored_seconds():
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg is not installed")
    filters = subprocess.run([executable, "-hide_banner", "-filters"], capture_output=True, text=True, check=True)
    if "rubberband" not in filters.stdout:
        pytest.skip("FFmpeg lacks the Rubber Band filter")
    time = torch.arange(EXPANDED_SAMPLES, dtype=torch.float64) / SAMPLE_RATE
    tone = (.1 * torch.sin(2 * torch.pi * 440 * time)).float().repeat(2, 1)
    result = transform_stem(tone, semitones=2, tempo=1.12, source_index=1, ffmpeg=executable)
    assert result.shape == (2, CROP_SAMPLES) and torch.equal(result[0], result[1])
    last_second = result[0, -SAMPLE_RATE:].double().numpy()
    peak_hz = int(np.abs(np.fft.rfft(last_second * np.hanning(SAMPLE_RATE))).argmax())
    assert abs(peak_hz - 440 * 2 ** (2 / 12)) < 1
    assert torch.equal(result, transform_stem(tone, semitones=2, tempo=1.12,
                                             source_index=1, ffmpeg=executable))
