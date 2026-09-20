"""Historical per-window, per-track, then per-stem aggregation."""
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from .metrics import (MetricConfig, SOURCE_ORDER, band_sdr, mean_or_none,
                      mixture_consistency, projection_metrics, windowed_sdr)

def _weighted(results: Iterable[Mapping[str, Any]], value: str, weight: str) -> float | None:
    pairs = [
        (float(item[value]), int(item[weight]))
        for item in results
        if item.get(value) is not None and int(item.get(weight, 0)) > 0
    ]
    total = sum(item_weight for _, item_weight in pairs)
    if total == 0:
        return None
    return float(sum(item_value * item_weight for item_value, item_weight in pairs) / total)


def _mean_matrix(matrices: Sequence[Sequence[Sequence[float | None]]]) -> list[list[float | None]]:
    if not matrices:
        return []
    rows, columns = len(matrices[0]), len(matrices[0][0])
    return [
        [mean_or_none(matrix[row][column] for matrix in matrices)
         for column in range(columns)]
        for row in range(rows)
    ]


def _score_track(
    name: str,
    intervals: Sequence[Mapping[str, int | str]],
    mixtures: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    estimates: Sequence[np.ndarray],
    metric_config: MetricConfig,
) -> dict[str, Any]:
    full = [[] for _ in SOURCE_ORDER]
    bands = {
        band_name: [[] for _ in SOURCE_ORDER]
        for band_name in metric_config.bands_hz
    }
    projections = []
    consistencies = []
    for mixture, refs, ests in zip(mixtures, references, estimates):
        for source in range(len(SOURCE_ORDER)):
            full[source].append(windowed_sdr(refs[source], ests[source], metric_config))
            for band_name, limits in metric_config.bands_hz.items():
                bands[band_name][source].append(
                    band_sdr(refs[source], ests[source], limits, metric_config)
                )
        projections.append(projection_metrics(refs, ests, metric_config))
        consistencies.append(mixture_consistency(mixture, ests, metric_config))

    per_stem: dict[str, Any] = {}
    for source, source_name in enumerate(SOURCE_ORDER):
        projection_items = [item["per_stem"][source] for item in projections]
        per_stem[source_name] = {
            "full_sdr_db": _weighted(full[source], "db", "active_windows"),
            "band_sdr_db": {
                band_name: _weighted(bands[band_name][source], "db", "active_windows")
                for band_name in metric_config.bands_hz
            },
            "sir_db": _weighted(projection_items, "sir_db", "active_windows"),
            "absent_fp_dbfs": _weighted(
                projection_items, "absent_fp_dbfs", "absent_windows"
            ),
            "absent_fp_ratio_db": _weighted(
                projection_items, "absent_fp_ratio_db", "absent_windows"
            ),
            "active_windows": sum(item["active_windows"] for item in full[source]),
            "absent_windows": sum(item["absent_windows"] for item in projection_items),
        }

    return {
        "name": name,
        "excerpts": [dict(interval) for interval in intervals],
        "full_sdr_db": mean_or_none(
            per_stem[source]["full_sdr_db"] for source in SOURCE_ORDER
        ),
        "low_sdr_db": mean_or_none(
            per_stem[source]["band_sdr_db"]["low_20_250"]
            for source in SOURCE_ORDER
        ),
        "band_sdr_db": {
            band_name: mean_or_none(
                per_stem[source]["band_sdr_db"][band_name]
                for source in SOURCE_ORDER
            )
            for band_name in metric_config.bands_hz
        },
        "bleed_sir_db": mean_or_none(
            per_stem[source]["sir_db"] for source in SOURCE_ORDER
        ),
        "per_stem": per_stem,
        "projection_attribution_db": _mean_matrix(
            [item["projection_attribution_db"] for item in projections]
        ),
        "mixture_consistency_db": mean_or_none(
            item["db"] for item in consistencies
        ),
        "mixture_consistency_error_rms": mean_or_none(
            item["error_rms"] for item in consistencies
        ),
    }


def _aggregate_tracks(tracks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_stem: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        per_stem[source] = {
            "full_sdr_db": mean_or_none(
                track["per_stem"][source]["full_sdr_db"] for track in tracks
            ),
            "band_sdr_db": {
                band: mean_or_none(
                    track["per_stem"][source]["band_sdr_db"][band]
                    for track in tracks
                )
                for band in tracks[0]["per_stem"][source]["band_sdr_db"]
            },
            "sir_db": mean_or_none(
                track["per_stem"][source]["sir_db"] for track in tracks
            ),
            "absent_fp_dbfs": mean_or_none(
                track["per_stem"][source]["absent_fp_dbfs"] for track in tracks
            ),
            "absent_fp_ratio_db": mean_or_none(
                track["per_stem"][source]["absent_fp_ratio_db"] for track in tracks
            ),
        }
    primary_values = [
        value
        for source in SOURCE_ORDER
        for value in (
            per_stem[source]["full_sdr_db"],
            per_stem[source]["band_sdr_db"]["low_20_250"],
            per_stem[source]["sir_db"],
        )
    ]
    complete = all(value is not None for value in primary_values)
    full_sdr = mean_or_none([
        per_stem[source]["full_sdr_db"] for source in SOURCE_ORDER
    ])
    low_sdr = mean_or_none([
        per_stem[source]["band_sdr_db"]["low_20_250"] for source in SOURCE_ORDER
    ])
    bleed_sir = mean_or_none([
        per_stem[source]["sir_db"] for source in SOURCE_ORDER
    ])
    aggregate: dict[str, Any] = {
        "val_score": (0.50 * full_sdr + 0.25 * low_sdr + 0.25 * bleed_sir) if complete else None,
        "primary_metrics_complete": complete,
        "full_sdr_db": full_sdr,
        "low_sdr_db": low_sdr,
        "band_sdr_db": {
            band: mean_or_none(
                per_stem[source]["band_sdr_db"][band] for source in SOURCE_ORDER
            )
            for band in tracks[0]["band_sdr_db"]
        },
        "bleed_sir_db": bleed_sir,
        "per_stem": per_stem,
        "projection_attribution_db": _mean_matrix(
            [track["projection_attribution_db"] for track in tracks]
        ),
        "mixture_consistency_db": mean_or_none(
            track["mixture_consistency_db"] for track in tracks
        ),
        "mixture_consistency_error_rms": mean_or_none(
            track["mixture_consistency_error_rms"] for track in tracks
        ),
    }
    return aggregate
