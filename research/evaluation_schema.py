"""Dependency-free validation for evaluator outputs consumed by ``decide.py``.

The evaluator and orchestrator both call this module.  Keeping the decision
input contract here prevents a successful run from producing a summary which
only fails later, when the replicated decision engine tries to derive its 40
robust guardrails.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SOURCE_ORDER = ("drums", "bass", "vocals", "other")
DECISION_GUARDRAIL_COUNT = 40
ABSENT_FP_RATIO_POLICY = "required_finite"
SILENT_MATRIX_POLICY = "nullable_cells_finite_when_present"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    parsed = float(value)
    _require(math.isfinite(parsed), f"{label} must be finite")
    return parsed


def _finite_matrix(value: Any, label: str, *, nullable: bool) -> tuple[int, int]:
    _require(
        isinstance(value, list) and len(value) == len(SOURCE_ORDER),
        f"{label} must have four rows",
    )
    finite_count = 0
    null_count = 0
    for row_index, row in enumerate(value):
        _require(
            isinstance(row, list) and len(row) == len(SOURCE_ORDER),
            f"{label}[{row_index}] must have four columns",
        )
        for column_index, cell in enumerate(row):
            cell_label = f"{label}[{row_index}][{column_index}]"
            if cell is None:
                _require(nullable, f"{cell_label} must be finite, not null")
                null_count += 1
            else:
                _finite(cell, cell_label)
                finite_count += 1
    return finite_count, null_count


def _finite_vector(value: Any, label: str) -> None:
    _require(
        isinstance(value, list) and len(value) == len(SOURCE_ORDER),
        f"{label} must contain four values",
    )
    for index, cell in enumerate(value):
        _finite(cell, f"{label}[{index}]")


def validate_decision_inputs(aggregate: Any) -> dict[str, Any]:
    """Validate every evaluator field needed to derive the 40 guardrails.

    Active mixed/probe matrices and desired-head metrics are mandatory and
    finite.  Per-stem absent ratios are also mandatory and finite for this
    frozen validation set.  Only the two silent-input diagnostic matrices
    permit nullable cells, and every present cell is finite.
    """

    _require(isinstance(aggregate, Mapping), "aggregate must be an object")
    for key in ("val_score", "full_sdr_db", "low_sdr_db", "bleed_sir_db"):
        _require(key in aggregate, f"aggregate.{key} is missing")
        _finite(aggregate[key], f"aggregate.{key}")

    per_stem = aggregate.get("per_stem")
    _require(isinstance(per_stem, Mapping), "aggregate.per_stem is missing")
    absent_finite_count = 0
    absent_null_count = 0
    for source in SOURCE_ORDER:
        record = per_stem.get(source)
        _require(
            isinstance(record, Mapping),
            f"aggregate.per_stem.{source} is missing",
        )
        bands = record.get("band_sdr_db")
        _require(
            isinstance(bands, Mapping),
            f"aggregate.per_stem.{source}.band_sdr_db is missing",
        )
        for field, value in (
            ("full_sdr_db", record.get("full_sdr_db")),
            ("band_sdr_db.low_20_250", bands.get("low_20_250")),
            ("sir_db", record.get("sir_db")),
        ):
            _finite(value, f"aggregate.per_stem.{source}.{field}")
        _require(
            "absent_fp_ratio_db" in record,
            f"aggregate.per_stem.{source}.absent_fp_ratio_db is missing",
        )
        absent = record["absent_fp_ratio_db"]
        _finite(absent, f"aggregate.per_stem.{source}.absent_fp_ratio_db")
        absent_finite_count += 1

    projection_finite, projection_null = _finite_matrix(
        aggregate.get("projection_attribution_db"),
        "aggregate.projection_attribution_db",
        nullable=False,
    )
    assert projection_null == 0

    single = aggregate.get("single_source")
    _require(isinstance(single, Mapping), "aggregate.single_source is missing")
    track_names = single.get("track_names")
    _require(
        isinstance(track_names, list)
        and bool(track_names)
        and all(isinstance(name, str) and name for name in track_names)
        and len(set(track_names)) == len(track_names),
        "aggregate.single_source.track_names must be a non-empty unique string list",
    )
    _require(
        single.get("track_count") == len(track_names),
        "aggregate.single_source.track_count disagrees with track_names",
    )
    _require(
        single.get("matrix_rows") == "isolated_input_source"
        and single.get("matrix_columns") == "estimated_head",
        "aggregate.single_source matrix orientation changed",
    )

    active_matrix_finite = 0
    for matrix_name in ("output_to_input_db", "low_output_to_input_db"):
        finite_count, null_count = _finite_matrix(
            single.get(matrix_name),
            f"aggregate.single_source.{matrix_name}",
            nullable=False,
        )
        assert null_count == 0
        active_matrix_finite += finite_count

    desired = single.get("desired_head_retention")
    activity = single.get("input_activity")
    _require(
        isinstance(desired, Mapping),
        "aggregate.single_source.desired_head_retention is missing",
    )
    _require(
        isinstance(activity, Mapping),
        "aggregate.single_source.input_activity is missing",
    )
    desired_finite_count = 0
    for source in SOURCE_ORDER:
        desired_record = desired.get(source)
        _require(
            isinstance(desired_record, Mapping),
            f"aggregate.single_source.desired_head_retention.{source} is missing",
        )
        for key in ("sdr_db", "low_sdr_db", "gain_db"):
            _finite(
                desired_record.get(key),
                f"aggregate.single_source.desired_head_retention.{source}.{key}",
            )
            desired_finite_count += 1

        activity_record = activity.get(source)
        _require(
            isinstance(activity_record, Mapping),
            f"aggregate.single_source.input_activity.{source} is missing",
        )
        for key in (
            "active_excerpts",
            "silent_excerpts",
            "low_active_excerpts",
            "low_silent_excerpts",
        ):
            count = activity_record.get(key)
            _require(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                f"aggregate.single_source.input_activity.{source}.{key} "
                "must be a nonnegative integer",
            )
        _require(
            activity_record["active_excerpts"] > 0
            and activity_record["low_active_excerpts"] > 0,
            f"aggregate.single_source.input_activity.{source} has no active probe",
        )

    silent_finite_count = 0
    silent_null_count = 0
    for matrix_name in ("silent_input_fp_dbfs", "low_silent_input_fp_dbfs"):
        finite_count, null_count = _finite_matrix(
            single.get(matrix_name),
            f"aggregate.single_source.{matrix_name}",
            nullable=True,
        )
        silent_finite_count += finite_count
        silent_null_count += null_count
    for vector_name in (
        "silent_input_fp_dbfs_by_output",
        "low_silent_input_fp_dbfs_by_output",
    ):
        _finite_vector(
            single.get(vector_name),
            f"aggregate.single_source.{vector_name}",
        )

    return {
        "guardrail_count": DECISION_GUARDRAIL_COUNT,
        "source_order": list(SOURCE_ORDER),
        "required_projection_matrix_finite_cells": projection_finite,
        "required_active_matrix_finite_cells": active_matrix_finite,
        "required_desired_metric_finite_values": desired_finite_count,
        "absent_fp_ratio_policy": ABSENT_FP_RATIO_POLICY,
        "absent_fp_ratio_finite_values": absent_finite_count,
        "absent_fp_ratio_null_values": absent_null_count,
        "silent_matrix_policy": SILENT_MATRIX_POLICY,
        "silent_matrix_finite_cells": silent_finite_count,
        "silent_matrix_null_cells": silent_null_count,
    }
