#!/usr/bin/env python3
"""Small CPU audio diagnostics for bass fidelity in all four output stems.

Score aligned exports from direct.evaluate; never shift waveforms again. The
reference_start in alignment.json determines the physical callback phase.
The old phase-zero error-D1 statistic is explicitly distinct from actual
86.13/172.27 Hz waveform-error energy. All measurements are diagnostic, without
promotion gates. Whole-excerpt SDR here is not the main evaluator's window SDR.

Examples::
    python -m research.direct.bass --audio-dir runs/audio/00-model --output bass.json
    python -m research.direct.bass --write-probes runs/bass-probes

Synthetic inputs have no prescribed source routing. score_tone_outputs fits
each output's input frequencies independently, so clean allocation to any
stem is allowed; generated frequencies and time variation remain measurable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44_100
SOURCES = ("drums", "bass", "vocals", "other")
PERIODS = (512, 256)
BANDS = {"20_80": (20., 80.), "80_250": (80., 250.), "20_250": (20., 250.)}
POWER_FLOOR = 1e-12  # -120 dBFS is a display floor, not a comparison veto.
ACTIVE_POWER = 1e-6  # -60 dBFS reference RMS; quiet stems get levels, not SDR.


def _db(power: float) -> float:
    return float(10 * np.log10(max(float(power), POWER_FLOOR)))


def _ratio_db(numerator: float, denominator: float) -> float:
    return _db(numerator) - _db(denominator)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > POWER_FLOOR else None


def _validate(estimates: np.ndarray, references: np.ndarray, mixture: np.ndarray) -> None:
    if (estimates.shape != references.shape or estimates.shape[:2] != (4, 2)
            or estimates.ndim != 3 or mixture.shape != estimates.shape[1:]
            or estimates.shape[-1] < SAMPLE_RATE):
        raise ValueError("need aligned estimates/references (4,2,T), mixture (2,T), T >= 44100")
    if not all(np.isfinite(value).all() for value in (estimates, references, mixture)):
        raise ValueError("audio contains non-finite values")


def seam_metrics(error: np.ndarray, reference_start: int, period: int) -> dict[str, Any]:
    """Physical-phase error derivative, with absolute errors and old ratios."""
    derivative = np.diff(np.asarray(error, dtype=np.float64), axis=-1)
    phase = (reference_start + np.arange(1, error.shape[-1])) % period
    count = np.bincount(phase, minlength=period) * 2
    phase_power = np.stack([
        sum(np.bincount(phase, weights=channel ** 2, minlength=period)
            for channel in source) / count
        for source in derivative
    ])
    near = np.minimum(phase, period - phase) <= 32
    interior_power = np.mean(derivative[..., ~near] ** 2, axis=(1, 2))
    near_power = np.mean(derivative[..., near] ** 2, axis=(1, 2))
    result = {}
    for index, source in enumerate(SOURCES):
        exact = float(phase_power[index, 0])
        median = float(np.median(phase_power[index, 1:]))
        result[source] = {
            "exact_error_d1_rms": float(np.sqrt(exact)),
            "near32_error_d1_rms": float(np.sqrt(near_power[index])),
            "interior_error_d1_rms": float(np.sqrt(interior_power[index])),
            "exact_to_interior_ratio": _ratio(np.sqrt(exact), np.sqrt(interior_power[index])),
            "near32_to_interior_ratio": _ratio(np.sqrt(near_power[index]), np.sqrt(interior_power[index])),
            # Historical phase-power ratios use their original 1e-18 floor;
            # absolute waveform levels elsewhere use the stated -120dB floor.
            "phase0_error_d1_power_over_median_db": float(10 * np.log10(
                max(exact, 1e-18) / max(median, 1e-18))),
        }
    return {"period_samples": period, "hop_rate_hz": SAMPLE_RATE / period,
            "first_boundary_offset": (-reference_start) % period,
            "exact_crossings": int(count[0] // 2), "per_stem": result}


def score_excerpt(estimates: np.ndarray, references: np.ndarray, mixture: np.ndarray,
                  reference_start: int) -> dict[str, Any]:
    estimates, references, mixture = (np.asarray(x, dtype=np.float64)
                                      for x in (estimates, references, mixture))
    _validate(estimates, references, mixture)
    if reference_start < 0:
        raise ValueError("reference_start must be the nonnegative absolute sample index")
    samples = mixture.shape[-1]
    error = estimates - references
    frequency = np.fft.rfftfreq(samples, 1 / SAMPLE_RATE)
    # Parseval weights preserve waveform mean power, including both channels.
    weights = np.full(frequency.size, 2. / samples ** 2)
    weights[0] *= .5
    if samples % 2 == 0:
        weights[-1] *= .5
    spectrum_ref, spectrum_est = np.fft.rfft(references), np.fft.rfft(estimates)

    def power(spectrum: np.ndarray, keep: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(spectrum[..., keep]) ** 2 * weights[keep], axis=-1).mean(axis=-1)

    bands = {}
    for name, (low, high) in BANDS.items():
        keep = (frequency >= low) & (frequency <= high)
        ref, est = power(spectrum_ref, keep), power(spectrum_est, keep)
        err = power(spectrum_est - spectrum_ref, keep)
        cross = np.sum((spectrum_est[..., keep] * spectrum_ref[..., keep].conj()).real
                       * weights[keep], axis=-1).mean(axis=-1)
        bands[name] = {source: {
            "reference_rms_dbfs": _db(ref[i]), "estimate_rms_dbfs": _db(est[i]),
            "error_rms_dbfs": _db(err[i]),
            "whole_excerpt_sdr_db": _ratio_db(ref[i], err[i]) if ref[i] >= ACTIVE_POWER else None,
            "estimate_reference_rms_ratio": _ratio(np.sqrt(est[i]), np.sqrt(ref[i])),
            "signed_target_gain": _ratio(cross[i], ref[i]),
            "target_correlation": _ratio(cross[i], np.sqrt(ref[i] * est[i])),
        } for i, source in enumerate(SOURCES)}

    # A Hann window reduces excerpt-edge leakage. Integrate actual waveform
    # energy within +/-1Hz, rather than calling a phase-zero seam a tone.
    hann = np.hanning(samples)
    correction = float(np.mean(hann ** 2))
    hann_ref, hann_est = np.fft.rfft(references * hann), np.fft.rfft(estimates * hann)
    tones = {}
    total_error_power = power(hann_est - hann_ref, np.ones(frequency.size, dtype=bool)) / correction
    for period in PERIODS:
        hz = SAMPLE_RATE / period
        keep = np.abs(frequency - hz) <= 1.
        ref, est = power(hann_ref, keep) / correction, power(hann_est, keep) / correction
        err = power(hann_est - hann_ref, keep) / correction
        tones[str(period)] = {"center_hz": hz, "half_width_hz": 1., "per_stem": {
            source: {"reference_rms_dbfs": _db(ref[i]), "estimate_rms_dbfs": _db(est[i]),
                     "error_rms_dbfs": _db(err[i]),
                     "estimate_to_reference_db": _ratio_db(est[i], ref[i]) if ref[i] >= POWER_FLOOR else None,
                     "error_to_total_error_db": _ratio_db(err[i], total_error_power[i]) if total_error_power[i] >= POWER_FLOOR else None}
            for i, source in enumerate(SOURCES)}}

    windows = samples // SAMPLE_RATE
    ref_power = (references[..., :windows * SAMPLE_RATE].reshape(4, 2, windows, SAMPLE_RATE) ** 2).mean(axis=(1, 3))
    est_power = (estimates[..., :windows * SAMPLE_RATE].reshape(4, 2, windows, SAMPLE_RATE) ** 2).mean(axis=(1, 3))
    absence = {}
    for i, source in enumerate(SOURCES):
        absent = ref_power[i] < ACTIVE_POWER
        absence[source] = {"reference_quiet_seconds": int(absent.sum()),
                           "output_rms_dbfs_when_reference_quiet": _db(est_power[i, absent].mean()) if absent.any() else None}
    return {"reference_start": reference_start, "samples": samples,
            "reconstruction_max_abs": float(np.max(np.abs(estimates.sum(axis=0) - mixture))),
            "reference_sum_max_abs": float(np.max(np.abs(references.sum(axis=0) - mixture))),
            "bands": bands, "narrowband_waveform": tones, "absence": absence,
            "seams": {str(period): seam_metrics(error, reference_start, period) for period in PERIODS}}


def _read(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != SAMPLE_RATE or values.shape[1] != 2:
        raise ValueError(f"expected 44.1kHz stereo: {path}")
    return values.T


def score_directory(directory: Path) -> dict[str, Any]:
    rows = []
    for alignment_path in sorted(directory.glob("*/excerpt-*/alignment.json")):
        folder = alignment_path.parent
        alignment = json.loads(alignment_path.read_text())
        mixture = _read(folder / "mixture.wav")
        refs = np.stack([_read(folder / f"reference-{source}.wav") for source in SOURCES])
        ests = np.stack([_read(folder / f"estimate-{source}.wav") for source in SOURCES])
        if alignment["reference_end"] - alignment["reference_start"] != mixture.shape[-1]:
            raise ValueError(f"alignment length differs from WAV: {folder}")
        row = score_excerpt(ests, refs, mixture, int(alignment["reference_start"]))
        row.update({"id": str(folder.relative_to(directory)), "alignment": alignment,
                    "reference_sha256": hashlib.sha256(refs.tobytes() + mixture.tobytes()).hexdigest()})
        rows.append(row)
    if not rows:
        raise ValueError(f"no direct.evaluate exports found under {directory}")

    def mean(values: Sequence[float | None]) -> float | None:
        valid = [value for value in values if value is not None]
        return float(np.mean(valid)) if valid else None

    summary = {source: {
        "low_20_250_whole_excerpt_sdr_db": mean([row["bands"]["20_250"][source]["whole_excerpt_sdr_db"] for row in rows]),
        "low_20_250_rms_recovery_ratio": mean([row["bands"]["20_250"][source]["estimate_reference_rms_ratio"] for row in rows]),
        "seams": {str(period): {
            "mean_exact_error_d1_rms": mean([row["seams"][str(period)]["per_stem"][source]["exact_error_d1_rms"] for row in rows]),
            "mean_phase0_error_d1_power_over_median_db": mean([row["seams"][str(period)]["per_stem"][source]["phase0_error_d1_power_over_median_db"] for row in rows]),
            "mean_narrowband_error_rms_dbfs": mean([row["narrowband_waveform"][str(period)]["per_stem"][source]["error_rms_dbfs"] for row in rows]),
        } for period in PERIODS},
    } for source in SOURCES}
    return {"schema_version": 1, "audio_dir": str(directory.resolve()),
            "definition": {"alignment": "already aligned audio; seam phase uses absolute reference_start",
                           "aggregate": "equal excerpt arithmetic mean; SDR is whole-excerpt, not standard evaluator SDR",
                           "display_floor_dbfs": -120, "quiet_reference_threshold_dbfs": -60,
                           "phase0_statistic": "error derivative power at callback phase zero / median other phases; not tone power",
                           "narrowband": "Hann-windowed waveform energy, +/-1Hz; can include source leakage, not a perceptual verdict",
                           "decision": "diagnostic only; assess recovery, absolute error, all stems and listening together"},
            "excerpt_count": len(rows), "summary": summary, "excerpts": rows}


def make_probes() -> list[dict[str, Any]]:
    """32 seconds: silence, off-grid notes, exact hop tones, harmonic bass notes."""
    probes = []
    specifications = [
        ("silence", [(), (), (), ()], [0., 0., 0., 0.]),
        ("bass_notes", [(41.203444,), (55.,), (82.406889,), (110.,)], [.2] * 4),
        ("hop_tones", [(SAMPLE_RATE / 512,), (SAMPLE_RATE / 256,),
                       (SAMPLE_RATE / 512,), (SAMPLE_RATE / 256,)], [.2, .2, .02, .02]),
        ("harmonic_bass", [(41.203444, 82.406889, 123.610333), (55., 110., 165.),
                           (65.406391, 130.812783, 196.219174), (82.406889, 164.813778, 247.220668)], [.2] * 4),
    ]
    for name, notes, amplitudes in specifications:
        mixture = np.zeros((2, 8 * SAMPLE_RATE), dtype=np.float64)
        segments = []
        for i, (frequencies, amplitude) in enumerate(zip(notes, amplitudes)):
            t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
            envelope = np.minimum(1., t / .05) * np.minimum(1., (2. - t) / .1)
            # Off-grid note changes and different channel phases expose joins.
            for channel, phase in enumerate((.23, .71)):
                for harmonic, frequency in enumerate(frequencies, 1):
                    mixture[channel, i * 2 * SAMPLE_RATE:(i + 1) * 2 * SAMPLE_RATE] += (
                        amplitude / harmonic ** 2 * envelope * np.sin(2 * np.pi * frequency * t + phase))
            segments.append({"start": round((2 * i + .4) * SAMPLE_RATE),
                             "end": round((2 * i + 1.8) * SAMPLE_RATE),
                             "frequencies_hz": list(frequencies)})
        probes.append({"id": name, "mixture": mixture.astype(np.float32), "segments": segments})
    return probes


def score_tone_outputs(estimates: np.ndarray, probe: dict[str, Any]) -> dict[str, Any]:
    """Fit input frequencies per stem; do not assume a sine belongs in Bass."""
    mixture = np.asarray(probe["mixture"], dtype=np.float64)
    estimates = np.asarray(estimates, dtype=np.float64)
    _validate(estimates, np.zeros_like(estimates), mixture)
    rows = []
    for segment in probe["segments"]:
        start, end = segment["start"], segment["end"]
        t = np.arange(start, end) / SAMPLE_RATE
        basis = [np.ones_like(t)]
        for hz in segment["frequencies_hz"]:
            basis.extend((np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)))
        basis = np.stack(basis, axis=1)
        output = estimates[..., start:end]
        coefficients = np.linalg.lstsq(basis, output.reshape(8, -1).T, rcond=None)[0]
        fitted = (basis @ coefficients).T.reshape(output.shape)
        unexplained = np.mean((output - fitted) ** 2, axis=(1, 2))
        output_power = np.mean(output ** 2, axis=(1, 2))
        rows.append({**segment, "per_stem": {source: {
            "output_rms_dbfs": _db(output_power[i]),
            "unexplained_rms_dbfs": _db(unexplained[i]),
            "unexplained_to_output_db": _ratio_db(unexplained[i], output_power[i]) if output_power[i] >= POWER_FLOOR else None,
            "dc_offset_rms": float(np.sqrt(np.mean(coefficients[0, 2 * i:2 * i + 2] ** 2))),
        } for i, source in enumerate(SOURCES)}})
    return {"id": probe["id"], "segments": rows,
            "reconstruction_max_abs": float(np.max(np.abs(estimates.sum(axis=0) - mixture)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-probes", type=Path)
    args = parser.parse_args()
    if not args.audio_dir and not args.write_probes:
        parser.error("provide --audio-dir and --output, or --write-probes")
    if args.audio_dir:
        if args.output is None:
            parser.error("--audio-dir requires --output")
        report = score_directory(args.audio_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "excerpt_count": report["excerpt_count"], "summary": report["summary"]}))
    if args.write_probes:
        args.write_probes.mkdir(parents=True, exist_ok=True)
        for probe in make_probes():
            sf.write(args.write_probes / f"{probe['id']}.wav", probe["mixture"].T, SAMPLE_RATE, subtype="FLOAT")
            metadata = {key: value for key, value in probe.items() if key != "mixture"}
            metadata.update({"sample_rate": SAMPLE_RATE, "source_routing": "unspecified; fit each output's input frequencies"})
            (args.write_probes / f"{probe['id']}.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
