"""Prospective source-controlled views added to the current sixteen-crop remix.

Keep twelve ordinary remixes, then use two instrumental and two vocals-only
views. All source gains, crop addresses and channel transforms are inherited.
This module does not select a parent, training schedule or GPU experiment.
"""
import torch

from research.direct.latency58_remix_augmentation import augment as base_augment, recipe as base_recipe

AUGMENTATION = "latency58-remix16-twelve-ordinary-two-instrumental-two-vocals-v1"
VIEW_NAMES = ("instrumental", "vocals_only", "ordinary")


def recipe(*, seed, first_sample_index):
    original = base_recipe(seed=seed, first_sample_index=first_sample_index)
    codes = torch.full((16,), 2, dtype=torch.int64)
    codes[12:14], codes[14:16] = 0, 1
    keep = torch.ones((16, 4), dtype=torch.bool)
    keep[12:14, 2] = False
    keep[14:16] = False
    keep[14:16, 2] = True
    return {**original, "base_factors": original["factors"],
            "factors": original["factors"] * keep, "source_keep": keep,
            "view_codes": codes, "controlled": codes != 2}


def augment(mixture, targets, *, seed, first_sample_index):
    mixed, desired, changed, factors = base_augment(mixture, targets, seed=seed, first_sample_index=first_sample_index)
    selected = recipe(seed=seed, first_sample_index=first_sample_index)
    desired = desired * selected["source_keep"][:, :, None, None]
    mixed = torch.where(selected["controlled"][:, None, None], desired.sum(1), mixed)
    return mixed, desired, changed, factors * selected["source_keep"]


def check():
    """Check physical views, preservation of ordinary audio and silent-source gradients."""
    from research.direct.latency58_branch_sdr_blend import objective
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261023)
    truth = .03 * torch.randn(16, 4, 2, 44160, generator=generator)
    # Keep a real-mixture/reference mismatch on untouched examples.
    mixed = truth.sum(1) + .000123
    before_audio, before_truth = mixed.clone(), truth.clone()
    args = {"seed": 20261023, "first_sample_index": 3_800_000}
    cpu_rng = torch.get_rng_state().clone()
    base = base_augment(mixed, truth, **args)
    transformed = augment(mixed, truth, **args)
    repeated = augment(mixed, truth, **args)
    selected = recipe(**args)
    audio, targets, changed, factors = transformed
    assert all(torch.equal(a, b) for a, b in zip(transformed, repeated, strict=True))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(mixed, before_audio) and torch.equal(truth, before_truth)
    assert torch.equal(audio[:12], base[0][:12]) and torch.equal(targets[:12], base[1][:12])
    assert torch.equal(audio[:4], mixed[:4]) and not torch.equal(audio[:4], targets[:4].sum(1))
    assert torch.equal(changed, base[2]) and torch.equal(factors, selected["factors"])
    assert torch.equal(audio[12:], targets[12:].sum(1))
    assert torch.count_nonzero(targets[12:14, 2]) == 0
    assert torch.count_nonzero(targets[14:, [0, 1, 3]]) == 0
    assert torch.equal(targets[12:14, [0, 1, 3]], base[1][12:14, [0, 1, 3]])
    assert torch.equal(targets[14:, 2], base[1][14:, 2])
    counts = torch.bincount(selected["view_codes"], minlength=3).tolist()
    assert counts == [2, 2, 12]
    noise = .008 * torch.randn(targets.shape, generator=generator)
    estimate = (targets + noise).requires_grad_()
    terms = objective(estimate, estimate, targets, audio)
    gradient, = torch.autograd.grad(terms.total, estimate)
    assert bool(torch.isfinite(gradient).all())
    silent = ~selected["source_keep"]
    assert bool((gradient[silent] * noise[silent]).sum() > 0)
    assert torch.count_nonzero(gradient[12:14, 2]) > 0
    assert torch.count_nonzero(gradient[14:, [0, 1, 3]]) > 0
    assert terms.active_window_counts.tolist() == [14, 14, 14, 14]
    assert terms.absent_window_counts.tolist() == [2, 2, 2, 2]
    with torch.no_grad():
        improved = objective(targets + .5 * noise, targets + .5 * noise, targets, audio)
        assert improved.total < terms.total and improved.absence_db < terms.absence_db
    assert not torch.cuda.is_initialized()
    return {"status": "pass", "augmentation": AUGMENTATION,
            "view_counts": dict(zip(VIEW_NAMES, counts, strict=True)),
            "all_twelve_ordinary_examples_bit_exact_to_current_remix": True,
            "original_recorded_mixture_mismatch_preserved": True,
            "controlled_views_sum_exactly_to_their_targets": True,
            "excluded_sources_exactly_zero_and_kept_sources_unchanged": True,
            "pristine_inputs_and_global_rng_unchanged": True,
            "addressed_replay_bit_exact": True, "current_blended_objective_has_restoring_silent_source_gradient": True,
            "active_windows": terms.active_window_counts.tolist(), "absent_windows": terms.absent_window_counts.tolist(),
            "parent_selected": False, "production_recipe_selected": False, "training_updates": 0,
            "validation_audio_used": False, "gpu_used": False, "quality_measured": False,
            "limitation": "Synthetic CPU data and output-space gradient checks only. This does not establish model learning, held-out vocal rejection, SDR improvement or native timing."}


if __name__ == "__main__":
    import json
    print(json.dumps(check()))
