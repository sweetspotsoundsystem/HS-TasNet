"""Counter-addressed crop sampling; independent of worker order and resume point."""
from __future__ import annotations

import hashlib
import math
import random
from typing import Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from .manifest import FrozenAudioFile, FrozenTrack, SOURCE_NAMES, require


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
        require(type(seed) is int and type(crop_samples) is int and crop_samples > 0
                and type(final_sample_index) is int and final_sample_index > 0
                and math.isfinite(vocal_active_probability) and 0 <= vocal_active_probability <= 1,
                "Invalid crop addressing or activity configuration")
        require(root_weights and tracks, "Provide training tracks and root weights")
        require(all(track.effective_frames >= crop_samples for track in tracks),
                "A training track is shorter than the requested crop")
        require({track.root_id for track in tracks} == set(root_weights),
                "Root weights must match the selected tracks")
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
        require(type(first) is int and type(stop) is int and 0 <= first <= stop,
                "Use ordered nonnegative absolute sample indices")
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
