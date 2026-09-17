from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "c91_production_loader_rejection", ROOT / "train_production.py"
)
assert SPEC is not None and SPEC.loader is not None
production = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = production
SPEC.loader.exec_module(production)


def _config() -> dict:
    return json.loads((ROOT / "full_config.json").read_text(encoding="utf-8"))


def _mutated_manifest(tmp_path: Path, mutate) -> Path:
    manifest = json.loads(
        (ROOT / "manifests/combined.manifest.json").read_text(encoding="utf-8")
    )
    mutate(manifest)
    manifest["content_sha256"] = production._manifest_content_hash(manifest)
    path = tmp_path / "mutated.manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return path


def _load_mutated(path: Path) -> None:
    production.load_corpus_manifest(
        path,
        expected_file_sha256=None,
        config=_config(),
    )


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    path = _mutated_manifest(
        tmp_path,
        lambda manifest: manifest["tracks"][0]["files"]["mixture"].__setitem__(
            "relative_path", "../../outside.wav"
        ),
    )
    with pytest.raises(RuntimeError, match="path escapes root"):
        _load_mutated(path)


def test_manifest_rejects_missing_required_stem(tmp_path: Path) -> None:
    path = _mutated_manifest(
        tmp_path,
        lambda manifest: manifest["tracks"][0]["files"].pop("vocals"),
    )
    with pytest.raises(RuntimeError, match="files must contain exactly"):
        _load_mutated(path)


def test_manifest_rejects_duplicate_track_identity(tmp_path: Path) -> None:
    def duplicate(manifest: dict) -> None:
        manifest["tracks"][1]["id"] = manifest["tracks"][0]["id"]

    path = _mutated_manifest(tmp_path, duplicate)
    with pytest.raises(RuntimeError, match="invalid or duplicate track id"):
        _load_mutated(path)


def test_manifest_rejects_bad_vocal_activity_metadata(tmp_path: Path) -> None:
    path = _mutated_manifest(
        tmp_path,
        lambda manifest: manifest["tracks"][0].__setitem__(
            "vocal_active_seconds", [1, 0, 1]
        ),
    )
    with pytest.raises(RuntimeError, match="sorted unique non-negative"):
        _load_mutated(path)


def _synthetic_track(tmp_path: Path) -> production.FrozenTrack:
    frames = 4096
    audio = np.zeros((frames, 2), dtype=np.float32)
    files = {}
    for label in production.FILE_NAMES:
        path = tmp_path / f"{label}.wav"
        sf.write(path, audio, 44_100, subtype="PCM_16")
        info = sf.info(path)
        files[label] = production.FrozenAudioFile(
            path=path,
            sha256=production.sha256_file(path),
            size_bytes=path.stat().st_size,
            frames=int(info.frames),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
        )
    return production.FrozenTrack(
        track_id="synthetic:track",
        root_id="synthetic",
        name="track",
        effective_frames=frames,
        files=files,
        vocal_active_seconds=(),
    )


def test_inventory_rejects_changed_size_and_header(tmp_path: Path) -> None:
    track = _synthetic_track(tmp_path)
    assert production.validate_audio_inventory([track], full_hash=False)[
        "files_checked"
    ] == 5

    bad_size_files = dict(track.files)
    bad_size_files["mixture"] = replace(
        track.files["mixture"], size_bytes=track.files["mixture"].size_bytes + 1
    )
    bad_size = replace(track, files=bad_size_files)
    with pytest.raises(RuntimeError, match="audio size changed"):
        production.validate_audio_inventory([bad_size], full_hash=False)

    bad_header_files = dict(track.files)
    bad_header_files["mixture"] = replace(
        track.files["mixture"], frames=track.files["mixture"].frames + 1
    )
    bad_header = replace(track, files=bad_header_files)
    with pytest.raises(RuntimeError, match="audio header changed"):
        production.validate_audio_inventory([bad_header], full_hash=False)


def test_inventory_rejects_disappeared_file(tmp_path: Path) -> None:
    track = _synthetic_track(tmp_path)
    track.files["vocals"].path.unlink()
    with pytest.raises(RuntimeError, match="audio file disappeared"):
        production.validate_audio_inventory([track], full_hash=False)


def test_inventory_rejects_same_size_content_tampering(tmp_path: Path) -> None:
    track = _synthetic_track(tmp_path)
    mixture = track.files["mixture"]
    original_size = mixture.path.stat().st_size
    replacement = np.full((mixture.frames, 2), 0.125, dtype=np.float32)
    sf.write(mixture.path, replacement, 44_100, subtype="PCM_16")
    assert mixture.path.stat().st_size == original_size
    with pytest.raises(RuntimeError, match="audio SHA-256 changed"):
        production.validate_audio_inventory([track], full_hash=True)


def test_reader_rejects_short_crop(tmp_path: Path) -> None:
    track = _synthetic_track(tmp_path)
    mixture = track.files["mixture"]
    with pytest.raises(RuntimeError, match="short or malformed crop"):
        production.CounterAddressedCropDataset._read(
            mixture, offset=0, frames=mixture.frames + 1
        )


def test_dataset_rejects_nonfinite_audio(tmp_path: Path) -> None:
    track = _synthetic_track(tmp_path)
    mixture = track.files["mixture"]
    poisoned = np.full((mixture.frames, 2), np.nan, dtype=np.float32)
    sf.write(mixture.path, poisoned, 44_100, subtype="FLOAT")
    info = sf.info(mixture.path)
    files = dict(track.files)
    files["mixture"] = production.FrozenAudioFile(
        path=mixture.path,
        sha256=production.sha256_file(mixture.path),
        size_bytes=mixture.path.stat().st_size,
        frames=int(info.frames),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
    )
    poisoned_track = replace(track, files=files)
    dataset = production.CounterAddressedCropDataset(
        [poisoned_track],
        root_weights={"synthetic": 1.0},
        seed=60,
        crop_samples=2048,
        vocal_active_probability=0.85,
        final_sample_index=1,
    )
    with pytest.raises(RuntimeError, match="non-finite crop"):
        dataset[0]
