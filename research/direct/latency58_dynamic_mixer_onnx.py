"""Literal-hop export using the existing recurrent state for both head features."""
from __future__ import annotations

import torch

from research.direct.latency58 import HOP, PUBLIC_FUSION_SCALE
from research.direct.latency58_dynamic_mixer import apply_correction, coefficients
from research.direct.latency58_residual_onnx import (
    INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES, STATE_SHAPES,
    make_export_copy as residual_export_copy, verify_onnx,
)


def make_export_copy(model):
    class DynamicWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        @property
        def model(self):
            return self.base.model

        def forward(self, audio_chunk, audio_history, fusion_hidden, spectral_numerator_tail, waveform_tail):
            output = self.base(audio_chunk, audio_history, fusion_hidden, spectral_numerator_tail, waveform_tail)
            previous = fusion_hidden[-1] / PUBLIC_FUSION_SCALE
            current = output[2][-1] / PUBLIC_FUSION_SCALE
            features = torch.stack((previous, current), 1)
            delta = coefficients(self.model.dynamic_mixer_head, features, self.model.dynamic_mixer_scale)
            deployed = apply_correction(output[0], audio_history[..., -HOP:], delta, self.model.dynamic_mixer_ramp)
            return (deployed, *output[1:])

    return DynamicWrapper(residual_export_copy(model)).eval()
