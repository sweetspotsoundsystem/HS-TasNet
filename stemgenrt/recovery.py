"""Check repeated training updates against an authenticated earlier journal."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


TIMING_FIELDS = {"data_wait_seconds", "compute_and_audit_seconds", "elapsed_seconds", "peak_vram_gib"}


def scientific_row(row):
    return {key: value for key, value in row.items() if key not in TIMING_FIELDS}


def compare_recorded_update(actual, expected):
    if scientific_row(actual) != scientific_row(expected):
        raise ValueError("Recovered scientific update differs at step " + str(expected.get("step")))


class ReplayVerifier:
    """Verify every recorded unsaved update before journaling or checkpointing.

    The original file is read once and never changed. An unterminated final
    line is treated as an uncommitted write; all complete lines must be valid.
    """

    def __init__(self, path, *, sha256, resume_step, schedule_steps):
        self.path = Path(path)
        contents = self.path.read_bytes()
        if not isinstance(sha256, str) or hashlib.sha256(contents).hexdigest() != sha256:
            raise ValueError("Replay journal SHA-256 differs")
        if type(resume_step) is not int or type(schedule_steps) is not int or not 0 <= resume_step < schedule_steps:
            raise ValueError("Invalid replay schedule")
        lines = contents.splitlines(keepends=True)
        self.incomplete_tail_ignored = bool(lines and not lines[-1].endswith(b"\n"))
        if self.incomplete_tail_ignored:
            lines.pop()
        def invalid_constant(value):
            raise ValueError("Nonfinite replay journal value: " + value)
        def finite_float(value):
            parsed = float(value)
            if not math.isfinite(parsed):
                invalid_constant(value)
            return parsed
        rows = [json.loads(line, parse_constant=invalid_constant, parse_float=finite_float) for line in lines]
        if not rows or any(not isinstance(row, dict) or type(row.get("step")) is not int for row in rows):
            raise ValueError("Replay journal requires completed update rows")
        first, last = rows[0]["step"], rows[-1]["step"]
        if not (1 <= first <= resume_step + 1 <= last <= schedule_steps):
            raise ValueError("Replay journal does not cover the unsaved updates")
        if [row["step"] for row in rows] != list(range(first, last + 1)):
            raise ValueError("Replay journal has missing, duplicate or unordered updates")
        self.sha256 = sha256
        self.resume_step, self.schedule_steps = resume_step, schedule_steps
        self.rows = {row["step"]: row for row in rows if row["step"] > resume_step}
        self.last_recorded_step = last
        self.next_step = resume_step + 1
        self.verified_steps = []

    def compare(self, row):
        step = row.get("step")
        if type(step) is not int or step != self.next_step or step > self.schedule_steps:
            raise ValueError("Replay verification skipped or reordered an update")
        if step <= self.last_recorded_step:
            compare_recorded_update(row, self.rows[step])
            self.verified_steps.append(step)
        self.next_step += 1

    def verify_source(self):
        with self.path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != self.sha256:
            raise ValueError("Replay journal changed during continuation")

    def report(self):
        return {"journal": str(self.path), "sha256": self.sha256,
                "resumed_from_step": self.resume_step, "last_recorded_step": self.last_recorded_step,
                "verified_steps": list(self.verified_steps),
                "ignored_fields": sorted(TIMING_FIELDS),
                "incomplete_tail_ignored": self.incomplete_tail_ignored}
