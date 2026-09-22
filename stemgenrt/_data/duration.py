"""Duration-weighted track selection with the established addressed reader."""
from bisect import bisect_right
from itertools import accumulate

import torch
from torch.utils.data import Dataset

from .manifest import SOURCE_NAMES, require


class DurationWeightedCropDataset(Dataset):
    """Wrap the original reader and RNG, changing only within-corpus track choice.

    Integer frame weights are identical for original and expanded crop geometry.
    Corpus draws retain the existing addressed RNG. The weighted track draw can
    consume a different number of RNG bits, so later anchor/offset choices are
    reproducible under this new recipe, but need not match the original recipe.
    """

    def __init__(self, original):
        self.original = original
        for key in ('seed', 'crop_samples', 'vocal_active_probability', 'final_sample_index',
                    'root_ids', 'cumulative_weights', 'tracks_by_root'):
            setattr(self, key, getattr(original, key))
        self.cumulative_frames = {}
        for root, tracks in self.tracks_by_root.items():
            weights = [track.effective_frames for track in tracks]
            require(bool(weights) and all(type(n) is int and n >= self.crop_samples for n in weights),
                    'Require positive integer durations covering the requested crop')
            self.cumulative_frames[root] = tuple(accumulate(weights))

    def __len__(self):
        return len(self.original)

    def _rng(self, sample_index):
        return self.original._rng(sample_index)

    def _read(self, file, *, offset, frames):
        return self.original._read(file, offset=offset, frames=frames)

    def _weighted_track_index(self, root_id, rng):
        cumulative = self.cumulative_frames[root_id]
        return bisect_right(cumulative, rng.randrange(cumulative[-1]))

    def __getitem__(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        require(0 <= sample_index < self.final_sample_index, f"sample index out of range: {sample_index}")
        rng = self._rng(int(sample_index))
        draw = rng.random()
        root_index = next((index for index, boundary in enumerate(self.cumulative_weights) if draw < boundary), len(self.root_ids) - 1)
        root_id = self.root_ids[root_index]
        root_tracks = self.tracks_by_root[root_id]
        track = root_tracks[self._weighted_track_index(root_id, rng)]
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
