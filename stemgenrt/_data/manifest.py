"""Portable, checksum-bound inventories of native stereo stem recordings."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf

SOURCE_NAMES = ("drums", "bass", "vocals", "other")
FILE_NAMES = ("mixture", *SOURCE_NAMES)
SAMPLE_RATE = 44100
ACTIVITY_POWER_FLOOR = 1073742 / 32768**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenAudioFile:
    path: Path
    sha256: str
    size_bytes: int
    frames: int
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class FrozenTrack:
    track_id: str
    root_id: str
    name: str
    effective_frames: int
    files: Mapping[str, FrozenAudioFile]
    vocal_active_seconds: tuple[int, ...]


@dataclass(frozen=True)
class CorpusManifest:
    path: Path
    sha256: str
    tracks: tuple[FrozenTrack, ...]
    root_weights: Mapping[str, float]
    split: str

    @property
    def root_ids(self):
        return tuple(self.root_weights)


def load_manifest(path, *, expected_sha256=None, verify_audio_hashes=True,
                  crop_samples=1, expected_split=None):
    """Load a manifest and verify audio size, format, frame counts and hashes.

    Roots are resolved relative to the manifest. No audio is normalized or
    resampled. Hash verification can be disabled for an already authenticated
    corpus; header and size checks still apply. This verifies exact files, not
    acoustic similarity between different encodings of the same recording.
    """
    path = Path(path).resolve(strict=True)
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    require(expected_sha256 is None or digest == expected_sha256, "Manifest checksum changed")
    manifest = json.loads(encoded)
    require(manifest.get("schema") == "hs-tasnet-stem-manifest-v1"
            and manifest.get("sample_rate") == SAMPLE_RATE and manifest.get("channels") == 2
            and manifest.get("source_order") == list(SOURCE_NAMES),
            "Expected a stereo 44.1 kHz stem manifest")
    require(type(crop_samples) is int and crop_samples > 0, "Invalid crop length")
    split = manifest.get("split", "train")
    require(split in ("train", "valid", "test") and (expected_split is None or split == expected_split),
            "Manifest split differs from the requested split")
    floor = manifest.get("activity_power_floor")
    require(type(floor) in (float, int) and floor == ACTIVITY_POWER_FLOOR,
            "Manifest vocal activity policy differs from the current recipe")
    roots, weights = {}, {}
    for record in manifest["roots"]:
        root_id, location, weight = record["id"], record["path"], record["weight"]
        require(isinstance(root_id, str) and root_id and root_id not in roots
                and isinstance(location, str) and location
                and type(weight) in (float, int) and math.isfinite(weight) and weight > 0,
                "Invalid manifest root")
        roots[root_id] = (path.parent / location).resolve(strict=True)
        require(roots[root_id].is_dir(), "Manifest root is not a directory")
        weights[root_id] = float(weight)
    require(roots, "Manifest contains no roots")
    tracks, identities, mixtures = [], set(), set()
    for record in manifest["tracks"]:
        root_id, track_id = record["root_id"], record["track_id"]
        require(root_id in roots and isinstance(track_id, str) and track_id and track_id not in identities,
                "Unknown root or duplicate track identity")
        identities.add(track_id)
        frames = record["effective_frames"]
        require(type(frames) is int and frames >= crop_samples, "A track is shorter than the requested crop")
        require(set(record["files"]) == set(FILE_NAMES), "A track must contain mixture and all four sources")
        files = {}
        for label, item in record["files"].items():
            relative = Path(item["path"])
            require(not relative.is_absolute(), "Audio paths must be relative to their root")
            location = (roots[root_id] / relative).resolve(strict=True)
            require(location.is_relative_to(roots[root_id]), "Source path escapes its declared root")
            expected_hash = item["sha256"]
            require(isinstance(expected_hash, str) and len(expected_hash) == 64
                    and all(c in "0123456789abcdef" for c in expected_hash), "Invalid audio checksum")
            file_frames = item.get("frames", frames)
            require(type(file_frames) is int and file_frames >= frames, "Invalid source frame count")
            info = sf.info(location)
            require(info.samplerate == SAMPLE_RATE and info.channels == 2 and info.frames == file_frames
                    and location.stat().st_size == item["bytes"]
                    and (not verify_audio_hashes or file_sha256(location) == expected_hash),
                    "Audio format, alignment or checksum changed: " + str(location))
            files[label] = FrozenAudioFile(location, expected_hash, item["bytes"], file_frames, SAMPLE_RATE, 2)
        require(frames == min(item.frames for item in files.values()), "Effective length must match the shortest source")
        require(files["mixture"].sha256 not in mixtures, "Duplicate mixture bytes occur in the manifest")
        mixtures.add(files["mixture"].sha256)
        active = record["vocal_active_seconds"]
        require(isinstance(active, list) and all(type(v) is int and v >= 0 and (v + 1) * SAMPLE_RATE <= frames
                for v in active) and active == sorted(set(active)), "Invalid vocal activity anchors")
        require(isinstance(record["name"], str) and record["name"], "Track name is missing")
        tracks.append(FrozenTrack(track_id, root_id, record["name"], frames, files, tuple(active)))
    require(tracks and set(weights) == {track.root_id for track in tracks}, "Every root must contain tracks")
    return CorpusManifest(path, digest, tuple(tracks), weights, split)


def require_disjoint(training_tracks, validation_tracks):
    """Reject shared mixture paths or exact bytes, including renamed files.

    This is an exact-file leakage guard. Users must additionally keep related
    versions of a recording in one split; acoustic deduplication is not implied.
    """
    if isinstance(training_tracks, CorpusManifest):
        training_tracks = training_tracks.tracks
    if isinstance(validation_tracks, CorpusManifest):
        validation_tracks = validation_tracks.tracks
    paths = {track.files["mixture"].path for track in training_tracks}
    hashes = {track.files["mixture"].sha256 for track in training_tracks}
    require(not any(track.files["mixture"].path in paths or track.files["mixture"].sha256 in hashes
                    for track in validation_tracks), "Training and held-out manifests overlap")


def build_manifest(roots, output, *, weights=None, split="train", excluded_names=(),
                   exclude_manifests=(), activity_power_floor=ACTIVITY_POWER_FLOOR):
    """Inventory stem folders, excluding explicitly supplied held-out material.

    Each root contains song folders with mixture/drums/bass/vocals/other.wav.
    Root insertion order is preserved because it is part of sample addressing.
    Exclusion manifests match mixture paths/bytes; excluded_names match relative
    song folder names. No corpus membership or song counts are hardcoded.
    """
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require(roots and split in ("train", "valid", "test"), "Invalid manifest inputs")
    require(activity_power_floor == ACTIVITY_POWER_FLOOR, "Keep the current vocal activity threshold")
    roots = {name: Path(path).resolve(strict=True) for name, path in roots.items()}
    weights = {name: 1. for name in roots} if weights is None else dict(weights)
    require(all(isinstance(name, str) and name for name in roots) and set(weights) == set(roots)
            and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in weights.values()),
            "Root names and positive weights must match")
    excluded_paths, excluded_hashes, exclusions = set(), set(), []
    for path in exclude_manifests:
        held_out = load_manifest(path)
        excluded_paths.update(t.files["mixture"].path for t in held_out.tracks)
        excluded_hashes.update(t.files["mixture"].sha256 for t in held_out.tracks)
        exclusions.append({"manifest_sha256": held_out.sha256, "split": held_out.split})
    excluded_names = set(excluded_names)
    result = {"schema": "hs-tasnet-stem-manifest-v1", "split": split,
              "sample_rate": SAMPLE_RATE, "channels": 2, "source_order": list(SOURCE_NAMES),
              "activity_power_floor": activity_power_floor,
              "roots": [{"id": name, "path": os.path.relpath(path, output.parent), "weight": weights[name]}
                        for name, path in roots.items()],
              "exclusions": {"names": sorted(excluded_names), "manifests": exclusions}, "tracks": []}
    mixtures = set()
    for root_id, root in roots.items():
        require(root.is_dir(), "Audio root is not a directory: " + str(root))
        count = 0
        for mixture in sorted(root.rglob("mixture.wav")):
            folder = mixture.parent
            name = folder.relative_to(root).as_posix()
            if name in excluded_names or mixture.resolve() in excluded_paths:
                continue
            mixture_hash = file_sha256(mixture)
            if mixture_hash in excluded_hashes:
                continue
            files = {}
            for label in FILE_NAMES:
                path = folder / (label + ".wav")
                require(path.resolve().is_relative_to(root), "Source path escapes its declared root")
                info = sf.info(path)
                require(info.samplerate == SAMPLE_RATE and info.channels == 2 and info.frames > 0,
                        "Every source must be nonempty stereo 44.1 kHz: " + str(path))
                files[label] = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                                "frames": int(info.frames),
                                "sha256": mixture_hash if label == "mixture" else file_sha256(path)}
                with sf.SoundFile(path) as stream:
                    for block in stream.blocks(blocksize=SAMPLE_RATE, dtype="float32", always_2d=True):
                        require(np.isfinite(block).all(), "Nonfinite source audio: " + str(path))
            require(mixture_hash not in mixtures, "Duplicate mixture bytes: " + str(folder))
            mixtures.add(mixture_hash)
            frames = min(item["frames"] for item in files.values())
            active = []
            with sf.SoundFile(folder / "vocals.wav") as stream:
                for second in range(frames // SAMPLE_RATE):
                    audio = stream.read(SAMPLE_RATE, dtype="float64", always_2d=True)
                    if np.mean(audio**2) > activity_power_floor:
                        active.append(second)
            result["tracks"].append({"track_id": root_id + ":" + name, "root_id": root_id, "name": name,
                                     "effective_frames": frames, "files": files, "vocal_active_seconds": active})
            count += 1
        require(count, "Audio root contains no selected songs: " + str(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output
