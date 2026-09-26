"""Frozen, dependency-light metrics for the StemgenRT-5.8 evaluator.

The functions in this module deliberately use only NumPy.  They operate on
channel-first audio and never perform permutation or delay searches.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SOURCE_ORDER = ("drums", "bass", "vocals", "other")


@dataclass(frozen=True)
class MetricConfig:
    """All numerical choices which must be frozen before baseline scoring."""

    sample_rate: int = 44_100
    window_samples: int = 44_100
    hop_samples: int = 44_100
    activity_dbfs: float = -50.0
    estimate_floor_dbfs: float = -100.0
    epsilon: float = 1e-12
    db_floor: float = -60.0
    db_ceiling: float = 60.0
    bands_hz: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "low_20_250": (20.0, 250.0),
            "low_20_80": (20.0, 80.0),
            "low_80_250": (80.0, 250.0),
            "low_250_500": (250.0, 500.0),
        }
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "MetricConfig":
        values = dict(values or {})
        if "bands_hz" in values:
            values["bands_hz"] = {
                str(name): tuple(float(v) for v in limits)
                for name, limits in values["bands_hz"].items()
            }
        config = cls(**values)
        if config.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if config.window_samples <= 0 or config.hop_samples <= 0:
            raise ValueError("metric window and hop must be positive")
        if config.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if config.db_floor >= config.db_ceiling:
            raise ValueError("db_floor must be less than db_ceiling")
        nyquist = config.sample_rate / 2
        for name, (low, high) in config.bands_hz.items():
            if not (0 <= low < high <= nyquist):
                raise ValueError(f"invalid band {name!r}: {(low, high)}")
        return config

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["bands_hz"] = {
            name: [float(limits[0]), float(limits[1])]
            for name, limits in self.bands_hz.items()
        }
        out["bandpass_method"] = "real FFT rectangular mask; low <= bin < high"
        out["stereo_projection_gain"] = "one scalar over both channels and time"
        return out


def _audio64(audio: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if ndim is not None and audio.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape {audio.shape}")
    if audio.shape[-1] == 0:
        raise ValueError("audio must contain at least one sample")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite values")
    return audio


def mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite, dtype=np.float64)) if finite else None


def rms_dbfs(audio: np.ndarray, epsilon: float = 1e-12) -> float:
    audio = _audio64(audio)
    power = float(np.mean(np.square(audio), dtype=np.float64))
    return float(10.0 * math.log10(max(power, epsilon)))


def db_ratio(
    numerator: float,
    denominator: float,
    *,
    epsilon: float,
    floor: float,
    ceiling: float,
) -> float:
    value = 10.0 * math.log10((max(float(numerator), 0.0) + epsilon) /
                              (max(float(denominator), 0.0) + epsilon))
    return float(np.clip(value, floor, ceiling))


def frame_ranges(length: int, window: int, hop: int) -> list[tuple[int, int]]:
    """Return full fixed windows, or one short window for a short excerpt."""

    if length <= 0:
        return []
    if length < window:
        return [(0, length)]
    return [(start, start + window) for start in range(0, length - window + 1, hop)]


def fft_bandpass(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Apply the evaluator's fixed ideal zero-phase real-FFT band-pass.

    The mask includes bins whose frequency is ``low_hz <= f < high_hz``.
    Filtering is performed independently for every leading-dimension signal.
    """

    audio = _audio64(audio)
    length = audio.shape[-1]
    frequencies = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    spectrum = np.fft.rfft(audio, axis=-1)
    spectrum *= mask
    return np.fft.irfft(spectrum, n=length, axis=-1)


def windowed_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    config: MetricConfig,
) -> dict[str, Any]:
    """Scale-dependent SDR, averaged in dB over active fixed windows."""

    reference = _audio64(reference)
    estimate = _audio64(estimate)
    if reference.shape != estimate.shape:
        raise ValueError(f"SDR shape mismatch: {reference.shape} != {estimate.shape}")

    values: list[float] = []
    ranges = frame_ranges(reference.shape[-1], config.window_samples, config.hop_samples)
    for start, end in ranges:
        ref = reference[..., start:end]
        if rms_dbfs(ref, config.epsilon) <= config.activity_dbfs:
            continue
        est = estimate[..., start:end]
        signal = float(np.sum(np.square(ref), dtype=np.float64))
        error = float(np.sum(np.square(est - ref), dtype=np.float64))
        values.append(db_ratio(signal, error, epsilon=config.epsilon,
                               floor=config.db_floor, ceiling=config.db_ceiling))

    return {
        "db": mean_or_none(values),
        "active_windows": len(values),
        "total_windows": len(ranges),
    }


def band_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    band_hz: Sequence[float],
    config: MetricConfig,
) -> dict[str, Any]:
    low, high = float(band_hz[0]), float(band_hz[1])
    filtered_ref = fft_bandpass(reference, config.sample_rate, low, high)
    filtered_est = fft_bandpass(estimate, config.sample_rate, low, high)
    return windowed_sdr(filtered_ref, filtered_est, config)


def projection_metrics(
    references: np.ndarray,
    estimates: np.ndarray,
    config: MetricConfig,
) -> dict[str, Any]:
    """Length-1 instantaneous projection SIR and bleed diagnostics.

    For estimated head ``i`` and reference stem ``j`` in each window, the
    scalar projection is ``g_ij = <y_i,s_j> / (||s_j||^2 + eps)``.  Its
    attributed energy is ``||g_ij s_j||^2``.  SIR is desired attributed
    energy divided by the sum of off-target attributed energies.  No delay,
    FIR filter, source permutation, or estimate rescaling is searched.
    A silent/near-silent estimate receives ``db_floor`` rather than a useful
    score.  One scalar is fitted jointly over stereo channels and time.
    """

    references = _audio64(references, ndim=3)
    estimates = _audio64(estimates, ndim=3)
    if references.shape != estimates.shape:
        raise ValueError(
            f"projection shape mismatch: {references.shape} != {estimates.shape}"
        )
    source_count = references.shape[0]
    ranges = frame_ranges(references.shape[-1], config.window_samples, config.hop_samples)
    sir_values: list[list[float]] = [[] for _ in range(source_count)]
    absent_dbfs: list[list[float]] = [[] for _ in range(source_count)]
    absent_ratio: list[list[float]] = [[] for _ in range(source_count)]
    attribution: list[list[list[float]]] = [
        [[] for _ in range(source_count)] for _ in range(source_count)
    ]

    for start, end in ranges:
        refs = references[..., start:end]
        ests = estimates[..., start:end]
        ref_energy = np.sum(np.square(refs), axis=(1, 2), dtype=np.float64)
        ref_active = np.asarray([
            rms_dbfs(refs[j], config.epsilon) > config.activity_dbfs
            for j in range(source_count)
        ])

        for head in range(source_count):
            est = ests[head]
            est_energy = float(np.sum(np.square(est), dtype=np.float64))
            est_floor = rms_dbfs(est, config.epsilon) <= config.estimate_floor_dbfs
            projected_energy = np.zeros(source_count, dtype=np.float64)

            for source in range(source_count):
                if ref_energy[source] <= config.epsilon:
                    continue
                gain = (
                    float(np.sum(est * refs[source], dtype=np.float64))
                    / (float(ref_energy[source]) + config.epsilon)
                )
                projected_energy[source] = gain * gain * float(ref_energy[source])
                if ref_active[source] and not est_floor:
                    attribution[head][source].append(
                        db_ratio(projected_energy[source], est_energy,
                                 epsilon=config.epsilon,
                                 floor=config.db_floor,
                                 ceiling=0.0)
                    )

            if ref_active[head]:
                desired = float(projected_energy[head])
                interference = float(projected_energy.sum() - projected_energy[head])
                if est_floor or desired <= config.epsilon:
                    sir_values[head].append(config.db_floor)
                else:
                    sir_values[head].append(
                        db_ratio(desired, interference, epsilon=config.epsilon,
                                 floor=config.db_floor, ceiling=config.db_ceiling)
                    )
            else:
                absent_dbfs[head].append(rms_dbfs(est, config.epsilon))
                other_energy = float(ref_energy.sum() - ref_energy[head])
                absent_ratio[head].append(
                    db_ratio(est_energy, other_energy, epsilon=config.epsilon,
                             floor=config.db_floor, ceiling=config.db_ceiling)
                )

    per_stem = []
    for source in range(source_count):
        per_stem.append({
            "sir_db": mean_or_none(sir_values[source]),
            "active_windows": len(sir_values[source]),
            "absent_windows": len(absent_dbfs[source]),
            "absent_fp_dbfs": mean_or_none(absent_dbfs[source]),
            "absent_fp_ratio_db": mean_or_none(absent_ratio[source]),
        })

    matrix = [
        [mean_or_none(attribution[head][source]) for source in range(source_count)]
        for head in range(source_count)
    ]
    return {
        "sir_db": mean_or_none(item["sir_db"] for item in per_stem),
        "per_stem": per_stem,
        "projection_attribution_db": matrix,
        "matrix_rows": "estimated_head",
        "matrix_columns": "reference_source",
    }


def mixture_consistency(
    mixture: np.ndarray,
    estimates: np.ndarray,
    config: MetricConfig,
) -> dict[str, Any]:
    mixture = _audio64(mixture, ndim=2)
    estimates = _audio64(estimates, ndim=3)
    if estimates.shape[1:] != mixture.shape:
        raise ValueError("mixture consistency shape mismatch")
    reconstructed = estimates.sum(axis=0, dtype=np.float64)
    error = reconstructed - mixture
    mixture_energy = float(np.sum(np.square(mixture), dtype=np.float64))
    error_energy = float(np.sum(np.square(error), dtype=np.float64))
    consistency_db = db_ratio(mixture_energy, error_energy, epsilon=config.epsilon,
                              floor=config.db_floor, ceiling=config.db_ceiling)
    return {
        "db": consistency_db,
        "error_db": -consistency_db,
        "error_rms": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
    }


def single_source_probe_metrics(
    reference_source: np.ndarray,
    estimates: np.ndarray,
    desired_head: int,
    config: MetricConfig,
) -> dict[str, Any]:
    """Score one isolated input stem against every model output head."""

    reference_source = _audio64(reference_source, ndim=2)
    estimates = _audio64(estimates, ndim=3)
    if estimates.shape[1:] != reference_source.shape:
        raise ValueError("single-source probe shape mismatch")
    if not 0 <= desired_head < estimates.shape[0]:
        raise ValueError("desired_head is out of range")

    ref_energy = float(np.sum(np.square(reference_source), dtype=np.float64))
    ratios = []
    low_ratios = []
    low_band = config.bands_hz["low_20_250"]
    low_ref = fft_bandpass(reference_source, config.sample_rate, *low_band)
    low_ref_energy = float(np.sum(np.square(low_ref), dtype=np.float64))
    for output in estimates:
        output_energy = float(np.sum(np.square(output), dtype=np.float64))
        ratios.append(db_ratio(output_energy, ref_energy, epsilon=config.epsilon,
                               floor=config.db_floor, ceiling=config.db_ceiling))
        low_output = fft_bandpass(output, config.sample_rate, *low_band)
        low_output_energy = float(np.sum(np.square(low_output), dtype=np.float64))
        low_ratios.append(db_ratio(low_output_energy, low_ref_energy,
                                   epsilon=config.epsilon,
                                   floor=config.db_floor,
                                   ceiling=config.db_ceiling))

    return {
        "desired_sdr_db": windowed_sdr(
            reference_source, estimates[desired_head], config
        )["db"],
        "desired_gain_db": ratios[desired_head],
        "desired_low_sdr_db": band_sdr(
            reference_source, estimates[desired_head], low_band, config
        )["db"],
        "output_to_input_db": ratios,
        "low_output_to_input_db": low_ratios,
    }
