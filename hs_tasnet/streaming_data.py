"""Checksum-bound stem manifests and deterministic, absolute-index crop sampling."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from .streaming_model import SOURCE_ORDER, require
from .streaming_checkpoint import file_sha256

SOURCE_NAMES = SOURCE_ORDER
FILE_NAMES = ("mixture", *SOURCE_NAMES)


def load_manifest(path, *, crop_samples):
    """Validate every source file and return stable tracks and root weights."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == "hs-tasnet-stem-manifest-v1" and manifest.get("sample_rate") == 44100
            and manifest.get("channels") == 2 and manifest.get("source_order") == list(SOURCE_NAMES),
            "Expected a stereo 44.1 kHz stem manifest")
    roots, weights = {}, {}
    for record in manifest["roots"]:
        root_id, location, weight = record["id"], record["path"], record["weight"]
        require(isinstance(root_id, str) and root_id and root_id not in roots
                and type(weight) in (float, int) and math.isfinite(weight) and weight > 0, "Invalid manifest root")
        roots[root_id] = (path.parent / location).resolve()
        weights[root_id] = float(weight)
    require(roots, "Manifest contains no roots")
    tracks, identities, mixtures = [], set(), set()
    for record in manifest["tracks"]:
        root_id, track_id = record["root_id"], record["track_id"]
        require(root_id in roots and isinstance(track_id, str) and track_id and track_id not in identities,
                "Unknown root or duplicate track identity")
        identities.add(track_id)
        frames = record["effective_frames"]
        require(type(frames) is int and frames >= crop_samples, "A training track is shorter than the requested crop")
        require(set(record["files"]) == set(FILE_NAMES), "A track must contain mixture and all four sources")
        files = {}
        for label, file in record["files"].items():
            location = (roots[root_id] / file["path"]).resolve()
            require(location.is_relative_to(roots[root_id]), "Source path escapes its declared root")
            info = sf.info(location)
            require(info.samplerate == 44100 and info.channels == 2 and info.frames == frames
                    and location.stat().st_size == file["bytes"] and file_sha256(location) == file["sha256"],
                    "Audio format, alignment or checksum changed: " + str(location))
            files[label] = FrozenAudioFile(location, file["sha256"], file["bytes"], frames, 44100, 2)
        require(files["mixture"].sha256 not in mixtures, "Duplicate mixture bytes occur in the training manifest")
        mixtures.add(files["mixture"].sha256)
        active = record["vocal_active_seconds"]
        require(isinstance(active, list) and all(type(value) is int and 0 <= value and (value + 1) * 44100 <= frames
                for value in active) and active == sorted(set(active)), "Invalid vocal activity anchors")
        tracks.append(FrozenTrack(track_id, root_id, record["name"], frames, files, tuple(active)))
    require(tracks and set(weights) == {track.root_id for track in tracks}, "Every root must contain tracks")
    return tracks, weights, file_sha256(path)


def require_disjoint(training_tracks, validation_tracks):
    """Reject exact mixture-file or path overlap between two manifests."""
    paths = {track.files["mixture"].path for track in training_tracks}
    hashes = {track.files["mixture"].sha256 for track in training_tracks}
    require(not any(track.files["mixture"].path in paths or track.files["mixture"].sha256 in hashes
                    for track in validation_tracks), "Training and validation manifests overlap")


def build_manifest(roots, output, *, weights=None, activity_power_floor=1073742 / 32768**2):
    """Inventory dedicated training roots containing one stem folder per song.

    WAVs are read without normalization or resampling. Complete one-second vocal
    blocks above the native-level power floor become optional sampling anchors.
    """
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require(roots and math.isfinite(activity_power_floor) and activity_power_floor >= 0, "Invalid manifest inputs")
    roots = {name: Path(path).resolve() for name, path in roots.items()}
    weights = {name: 1. for name in roots} if weights is None else weights
    require(set(weights) == set(roots) and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                             for value in weights.values()), "Root weights must be positive")
    result = {"schema": "hs-tasnet-stem-manifest-v1", "sample_rate": 44100, "channels": 2,
              "source_order": list(SOURCE_NAMES), "activity_power_floor": activity_power_floor,
              "roots": [{"id": name, "path": str(path), "weight": weights[name]} for name, path in roots.items()],
              "tracks": []}
    mixtures = set()
    for root_id, root in roots.items():
        require(root.is_dir(), "Training root does not exist: " + str(root))
        found = sorted(root.rglob("mixture.wav"))
        require(found, "Training root contains no mixture.wav files: " + str(root))
        for mixture in found:
            folder = mixture.parent
            files, counts = {}, set()
            for label in FILE_NAMES:
                path = folder / (label + ".wav")
                info = sf.info(path)
                require(info.samplerate == 44100 and info.channels == 2 and info.frames > 0,
                        "Every source must be nonempty stereo 44.1 kHz: " + str(path))
                counts.add(info.frames)
                files[label] = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                                "sha256": file_sha256(path)}
            require(len(counts) == 1, "Mixture and source lengths differ: " + str(folder))
            require(files["mixture"]["sha256"] not in mixtures, "Duplicate mixture bytes: " + str(folder))
            mixtures.add(files["mixture"]["sha256"])
            frames = counts.pop()
            active = []
            with sf.SoundFile(folder / "vocals.wav") as stream:
                for second in range(frames // 44100):
                    audio = stream.read(44100, dtype="float64", always_2d=True)
                    require(np.isfinite(audio).all(), "Nonfinite vocal audio")
                    if np.mean(audio**2) > activity_power_floor:
                        active.append(second)
            name = str(folder.relative_to(root))
            result["tracks"].append({"track_id": root_id + ":" + name, "root_id": root_id, "name": name,
                                     "effective_frames": frames, "files": files, "vocal_active_seconds": active})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output


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


class CounterAddressedCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Every sample is a pure function of (seed, absolute sample index)."""

    def __init__(
        self,
        tracks: Sequence[FrozenTrack],
        *,
        root_weights: Mapping[str, float],
        seed: int,
        crop_samples: int,
        vocal_active_probability: float,
        final_sample_index: int,
    ) -> None:
        self.seed = int(seed)
        self.crop_samples = int(crop_samples)
        self.vocal_active_probability = float(vocal_active_probability)
        self.final_sample_index = int(final_sample_index)
        self.root_ids = tuple(root_weights)
        weights = tuple(float(root_weights[root_id]) for root_id in self.root_ids)
        require(all(weight > 0 and math.isfinite(weight) for weight in weights), "root weights must be finite and positive")
        total = sum(weights)
        self.cumulative_weights = tuple(sum(weights[: index + 1]) / total for index in range(len(weights)))
        grouped: dict[str, list[FrozenTrack]] = {root_id: [] for root_id in self.root_ids}
        for track in tracks:
            grouped[track.root_id].append(track)
        self.tracks_by_root = {root_id: tuple(sorted(grouped[root_id], key=lambda item: item.track_id)) for root_id in self.root_ids}
        require(all(self.tracks_by_root.values()), "every configured root must contain tracks")

    def __len__(self) -> int:
        return self.final_sample_index

    def _rng(self, sample_index: int) -> random.Random:
        material = f"hs-tasnet-c91-sample-v1:{self.seed}:{sample_index}".encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(material, digest_size=16).digest(), "big")
        return random.Random(seed)

    @staticmethod
    def _read(file: FrozenAudioFile, *, offset: int, frames: int) -> torch.Tensor:
        with sf.SoundFile(str(file.path), "r") as handle:
            handle.seek(offset)
            audio = handle.read(frames, dtype="float32", always_2d=True)
        require(audio.shape == (frames, 2), f"short or malformed crop from {file.path}: {audio.shape}")
        return torch.from_numpy(np.ascontiguousarray(audio.T))

    def __getitem__(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        require(0 <= sample_index < self.final_sample_index, f"sample index out of range: {sample_index}")
        rng = self._rng(int(sample_index))
        draw = rng.random()
        root_index = next((index for index, boundary in enumerate(self.cumulative_weights) if draw < boundary), len(self.root_ids) - 1)
        root_id = self.root_ids[root_index]
        root_tracks = self.tracks_by_root[root_id]
        track = root_tracks[rng.randrange(len(root_tracks))]
        max_start = track.effective_frames - self.crop_samples
        if track.vocal_active_seconds and rng.random() < self.vocal_active_probability:
            anchor = track.vocal_active_seconds[rng.randrange(len(track.vocal_active_seconds))] * 44_100
            offset = max(0, min(anchor - rng.randrange(self.crop_samples), max_start))
        else:
            offset = rng.randrange(max_start + 1)
        mixture = self._read(track.files["mixture"], offset=offset, frames=self.crop_samples)
        targets = torch.stack([self._read(track.files[source], offset=offset, frames=self.crop_samples) for source in SOURCE_NAMES])
        require(bool(torch.isfinite(mixture).all()) and bool(torch.isfinite(targets).all()), f"non-finite crop from {track.track_id}")
        return mixture, targets


class AbsoluteIndexSampler(Sampler[int]):
    def __init__(self, first: int, stop: int) -> None:
        self.first = int(first)
        self.stop = int(stop)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.first, self.stop))

    def __len__(self) -> int:
        return max(0, self.stop - self.first)


def worker_init(worker_id: int) -> None:
    # Crop selection uses its own counter-addressed RNG.  These seeds merely
    # prevent accidental future worker-local randomness from being inherited.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % (2**32))
