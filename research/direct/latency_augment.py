"""Optional supervised bass-note additions, independent of global training RNG."""

import math

import torch


def _note_envelope(time, onset, offset, attack, release):
    """Raised-cosine attack/release, exactly zero outside the note support."""
    attack_position = ((time - onset) / attack).clamp(0.0, 1.0)
    release_position = ((offset - time) / release).clamp(0.0, 1.0)
    return (0.5 - 0.5 * torch.cos(math.pi * attack_position)) * (
        0.5 - 0.5 * torch.cos(math.pi * release_position))


def add_bass_tones(mixture, targets, probability, seed):
    """Return mixture, targets, and a Boolean mask of augmented examples.

    Inputs are FP32 [B,2,T] mixture and [B,4,2,T] targets, ordered D/B/V/O,
    at 44.1 kHz with T >= 512. Add after ordinary remixing and before complete
    silence replacement. The same FP32 signal is added to mixture and Bass;
    the other three targets and every unselected example are unchanged.

    Independently per example: select with the supplied probability; draw f0
    log-uniform over [32,220] Hz; choose 1..4 harmonics uniformly with amplitude
    h**(-decay), decay uniform [1,3]. Fundamental phase is uniform [0,2*pi],
    and right-minus-left phase uniform [-pi/4,pi/4]; harmonic phases scale by h.
    Onset is uniform [0,0.25] of crop duration and offset uniform [0.75,1].
    Attack and release each draw uniform [20,80] ms, capped at one quarter of
    note duration. Raised-cosine ramps give zero amplitude and slope at edges.
    The enveloped stereo waveform is normalized by its joint absolute peak,
    then scaled to a log-uniform peak in [0.01,0.15]. No mixture clipping.

    All draws use a fresh device-local Generator seeded only by ``seed``;
    global RNG is untouched. A zero probability returns the original tensor
    objects and an all-false mask without constructing a generator or drawing.
    """
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Bass-tone probability must be finite and in [0,1]")
    if (mixture.ndim != 3 or mixture.shape[1] != 2 or mixture.shape[-1] < 512
            or targets.shape != (mixture.shape[0], 4, 2, mixture.shape[-1])
            or targets.device != mixture.device):
        raise ValueError("Expected matching same-device [B,2,T] and [B,4,2,T], T >= 512")
    if not probability:
        return mixture, targets, torch.zeros(mixture.shape[0], dtype=torch.bool, device=mixture.device)
    if mixture.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError("Bass-tone augmentation requires FP32 input audio")

    generator = torch.Generator(device=mixture.device).manual_seed(seed)
    batch, _, samples = mixture.shape

    def draw():
        return torch.rand(batch, 1, 1, device=mixture.device,
                          dtype=torch.float32, generator=generator)

    selected = draw().flatten() < probability
    frequency = (math.log(32.0) + draw() * math.log(220.0 / 32.0)).exp()
    harmonics = torch.randint(1, 5, (batch, 1, 1), device=mixture.device, generator=generator)
    decay = 1.0 + 2.0 * draw()
    phase = 2.0 * math.pi * draw()
    stereo_phase = (draw() - 0.5) * (math.pi / 2.0)
    phase = torch.cat((phase, phase + stereo_phase), dim=1)
    duration = samples / 44_100.0
    onset = 0.25 * duration * draw()
    offset = (0.75 + 0.25 * draw()) * duration
    attack = torch.minimum(0.020 + 0.060 * draw(), (offset - onset) / 4.0)
    release = torch.minimum(0.020 + 0.060 * draw(), (offset - onset) / 4.0)
    peak = (math.log(0.01) + draw() * math.log(15.0)).exp()
    time = torch.arange(samples, dtype=torch.float32, device=mixture.device)[None, None] / 44_100.0
    angle = 2.0 * math.pi * frequency * (time - onset) + phase
    tone = torch.zeros_like(mixture)
    for harmonic in range(1, 5):
        amplitude = harmonic ** (-decay) * (harmonics >= harmonic)
        tone = tone + amplitude * torch.sin(harmonic * angle)
    tone = tone * _note_envelope(time, onset, offset, attack, release)
    tone = tone * (peak / tone.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12))
    chosen = selected[:, None, None]
    augmented_mixture = torch.where(chosen, mixture + tone, mixture)
    augmented_targets = targets.clone()
    augmented_targets[:, 1] = torch.where(chosen, targets[:, 1] + tone, targets[:, 1])
    return augmented_mixture, augmented_targets, selected
