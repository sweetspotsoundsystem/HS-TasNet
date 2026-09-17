"""Addressed source remixing on complete batches of sixteen recorded crops.

Keep four original examples, transform four same-crop examples, and remix the
remaining eight from other crops in the batch. The batch is the atomic source
selection group; any gradient microbatching happens after this transformation.
"""
import math

import torch

AUGMENTATION = "quarter_original_quarter_same_crop_half_cross_crop_gain3db_channel_swap_v1"
GROUP_SIZE = 16


def recipe(*, seed, first_sample_index):
    if type(seed) is not int or type(first_sample_index) is not int or first_sample_index < 0 or first_sample_index % GROUP_SIZE:
        raise ValueError("Use an aligned sixteen-crop counter-address group")
    generator = torch.Generator().manual_seed(seed * 1_000_003 + first_sample_index)
    # Distinct nonzero cyclic shifts ensure that a remixed output uses four
    # different crop addresses, each different from that output's own address.
    shifts = (torch.randperm(GROUP_SIZE - 1, generator=generator)[:4] + 1).tolist()
    values = torch.rand((GROUP_SIZE, 10), generator=generator).tolist()
    indices, swaps, factors = [], [], []
    for row, randoms in enumerate(values):
        changed, remixed = row >= 4, row >= 8
        indices.append([(row + shifts[stem]) % GROUP_SIZE if remixed else row for stem in range(4)])
        swaps.append([changed and randoms[4 + stem] < .5 for stem in range(4)])
        polarity = -1. if randoms[8] < .5 else 1.
        factors.append([polarity * math.pow(10., (6 * randoms[stem] - 3) / 20) if changed else 1.
                        for stem in range(4)])
    return {"source_indices": torch.tensor(indices, dtype=torch.int64),
            "swap_channels": torch.tensor(swaps, dtype=torch.bool),
            "factors": torch.tensor(factors, dtype=torch.float32),
            "changed": torch.arange(GROUP_SIZE) >= 4,
            "cross_crop": torch.arange(GROUP_SIZE) >= 8}


def augment(mixture, targets, *, seed, first_sample_index):
    if mixture.device.type != "cpu" or targets.device.type != "cpu" or mixture.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError("Transform aligned FP32 training audio on CPU")
    if targets.ndim != 4 or targets.shape[:3] != (GROUP_SIZE, 4, 2) or mixture.shape != (GROUP_SIZE, 2, targets.shape[-1]):
        raise ValueError("Require sixteen aligned stereo mixtures and four source stems")
    if mixture.requires_grad or targets.requires_grad:
        raise ValueError("Training references must remain fixed")
    selected = recipe(seed=seed, first_sample_index=first_sample_index)
    stems = torch.arange(4)[None, :].expand(GROUP_SIZE, -1)
    sources = targets[selected["source_indices"], stems]
    sources = torch.where(selected["swap_channels"][:, :, None, None], sources.flip(2), sources)
    sources = sources * selected["factors"][:, :, None, None]
    rendered = torch.where(selected["changed"][:, None, None], sources.sum(1), mixture)
    return rendered, sources, selected["changed"], selected["factors"]


def check():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261012)
    targets = torch.randn((GROUP_SIZE, 4, 2, 2048), generator=generator) * .01
    targets[:, 2, 0] += .02
    # Real recorded mixture/reference mismatch must be retained in untouched rows.
    mixture = targets.sum(1) + .000123
    mixture_copy, targets_copy = mixture.clone(), targets.clone()
    args = {"seed": 20261012, "first_sample_index": 2_700_000}
    result = augment(mixture, targets, **args)
    audio, truth, changed, factors = result
    selected = recipe(**args)
    assert torch.equal(mixture, mixture_copy) and torch.equal(targets, targets_copy)
    assert torch.equal(audio[:4], mixture[:4]) and torch.equal(truth[:4], targets[:4])
    assert torch.equal(audio[changed], truth[changed].sum(1))
    assert changed.sum() == 12 and selected["cross_crop"].sum() == 8
    assert bool((factors[changed].abs() >= 10 ** (-3 / 20)).all())
    assert bool((factors[changed].abs() <= 10 ** (3 / 20)).all())
    assert selected["swap_channels"][4:].any() and (~selected["swap_channels"][4:]).any()
    for row in range(GROUP_SIZE):
        for stem in range(4):
            source = targets[selected["source_indices"][row, stem], stem]
            if selected["swap_channels"][row, stem]:
                source = source.flip(0)
            assert torch.equal(truth[row, stem], source * factors[row, stem])
        if row >= 8:
            assert len(set(selected["source_indices"][row].tolist())) == 4
            assert bool((selected["source_indices"][row] != row).all())
    assert all(torch.equal(a, b) for a, b in zip(result, augment(mixture, targets, **args), strict=True))
    other = augment(mixture, targets, seed=20261012, first_sample_index=2_700_016)
    assert not torch.equal(audio[4:], other[0][4:])
    zeros = augment(torch.zeros_like(mixture), torch.zeros_like(targets), **args)
    assert torch.count_nonzero(zeros[0]) == torch.count_nonzero(zeros[1]) == 0
    assert not torch.cuda.is_initialized()
    return {"status": "pass", "augmentation": AUGMENTATION, "group_size": GROUP_SIZE,
            "ordinary_mixture_and_stems_bit_exact": True, "source_audio_unmodified": True,
            "transformed_mixture_equals_stem_sum": True, "cross_crop_sources_distinct_and_not_own": True,
            "stereo_channel_and_gain_mapping_exact": True, "counter_address_replay_exact": True,
            "silence_exact": True, "gain_bounds_db": [-3, 3], "gpu_used": False}


if __name__ == "__main__":
    import json
    print(json.dumps(check()), flush=True)
