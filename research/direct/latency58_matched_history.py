"""Training views with identical scored coordinates and a shared teacher history."""
from __future__ import annotations

from research.direct.latency58_sdr_context import render_scored_context
from research.direct.train_latency58 import require

VERSION = "latency58-shared-teacher-eight-student-two-or-eight-v1"


def render_history_view(model, mixture, *, student_history_samples, teacher_history_samples, scored_samples):
    require(type(student_history_samples) is int and type(teacher_history_samples) is int
            and 0 < student_history_samples <= teacher_history_samples
            and student_history_samples % 128 == teacher_history_samples % 128 == 0
            and type(scored_samples) is int and scored_samples > 0
            and mixture.shape[-1] == teacher_history_samples + scored_samples,
            "Invalid matched history geometry")
    start = teacher_history_samples - student_history_samples
    output = render_scored_context(model, mixture[..., start:],
                                   warmup_samples=student_history_samples, carry_state=True)
    require(output.warmup_samples == student_history_samples and output.scored_samples == scored_samples
            and output.initial_state_detached, "Student history or detached state differs")
    return output
