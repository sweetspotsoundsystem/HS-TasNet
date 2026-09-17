"""Optional pure-Bass replacement using the existing continuous note sampler."""

import hashlib
import math

import torch

from research.direct.latency_augment import add_bass_tones


SEED_DOMAIN = "hs-tasnet/pure-bass/v1"


def _domain_seed(seed, purpose):
    payload = f"{SEED_DOMAIN}/{purpose}/{seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def replace_with_pure_bass(mixture, targets, deranged, probability=0.0, *, seed=None):
    """Return mixture, targets, derangement and the replacement mask.

    Inputs are same-device FP32 [B,2,T] and [B,4,2,T], T >= 512, with a
    Boolean [B] derangement mask. Sources are D/B/V/O. Independently selected
    examples become one newly sampled Bass note and targets [0,note,0,0];
    their derangement flags are cleared. Unselected examples are bit-preserved.

    Selection and note generation use separate deterministic seeds derived
    from the explicit nonnegative data-index seed. Notes reuse add_bass_tones
    at probability1 on zeros, preserving its continuous-frequency distribution.
    No global RNG is read or advanced. Probability0 returns the three original
    objects and an all-false mask without constructing any generator.
    """
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Pure-Bass probability must be finite and in [0,1]")
    if (not isinstance(mixture, torch.Tensor) or mixture.ndim != 3
            or mixture.shape[0] < 1 or mixture.shape[1] != 2 or mixture.shape[-1] < 512
            or not isinstance(targets, torch.Tensor)
            or targets.shape != (mixture.shape[0], 4, 2, mixture.shape[-1])
            or targets.device != mixture.device
            or mixture.dtype != torch.float32 or targets.dtype != torch.float32
            or not isinstance(deranged, torch.Tensor) or deranged.shape != (mixture.shape[0],)
            or deranged.dtype != torch.bool or deranged.device != mixture.device):
        raise ValueError("Expected same-device FP32 [B,2,T]/[B,4,2,T] and Boolean [B], B > 0, T >= 512")
    if not probability:
        return mixture, targets, deranged, torch.zeros(mixture.shape[0], dtype=torch.bool, device=mixture.device)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Pure-Bass replacement requires an explicit nonnegative integer data-index seed")
    generator = torch.Generator(device=mixture.device).manual_seed(_domain_seed(seed, "selection"))
    selected = torch.rand(mixture.shape[0], device=mixture.device, generator=generator) < probability
    notes, note_targets, _ = add_bass_tones(
        torch.zeros_like(mixture), torch.zeros_like(targets),
        probability=1.0, seed=_domain_seed(seed, "notes"),
    )
    replaced_mixture = torch.where(selected[:, None, None], notes, mixture)
    replaced_targets = torch.where(selected[:, None, None, None], note_targets, targets)
    return replaced_mixture, replaced_targets, deranged & ~selected, selected
