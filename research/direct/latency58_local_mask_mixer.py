"""Untrained, zero-initialized spectral-logit correction for a future experiment.

The same small nonlinear network processes each stereo channel independently.
It mixes the four sources and real/imaginary components across three adjacent
frequency bins. Every time frame is independent; no new streaming state exists.
"""
from __future__ import annotations

import torch
from torch import nn

from research.direct.latency58 import CHANNELS, MASK_BINS, SOURCES, require

VERSION = "latency58-stereo-shared-local-spectral-mask-mixer-v1"
HIDDEN = 16


class LocalSpectralMaskMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv1d(2 * SOURCES, HIDDEN, 3, padding=1)
        self.out = nn.Conv1d(HIDDEN, 2 * SOURCES, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, logits):
        require(logits.ndim == 3 and logits.shape[-1] == CHANNELS * MASK_BINS * 2 * SOURCES,
                "Expected flat [batch, frames, stereo * frequency * RI * sources] logits")
        batch, frames, _ = logits.shape
        # Fold time and stereo into the batch axis: neither is convolved or
        # normalized together. FP32 preserves the existing mask precision.
        with torch.autocast(logits.device.type, enabled=False):
            x = logits.float().reshape(batch, frames, CHANNELS, MASK_BINS, 2, SOURCES)
            x = x.permute(0, 1, 2, 4, 5, 3).reshape(batch * frames * CHANNELS, 2 * SOURCES, MASK_BINS)
            correction = self.out(torch.nn.functional.silu(self.local(x)))
            correction = correction.reshape(batch, frames, CHANNELS, 2, SOURCES, MASK_BINS)
            correction = correction.permute(0, 1, 2, 5, 3, 4).reshape(batch, frames, -1)
            return logits.float() + correction


class SpectralHeadWithLocalMixer(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.mixer = LocalSpectralMaskMixer()

    def forward(self, features):
        return self.mixer(self.head(features))
