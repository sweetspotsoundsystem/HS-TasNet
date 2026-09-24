"""Causal past-frame correction using the existing source masks."""
import torch
from torch import nn
from torch.nn import functional as F


class SharedMaskPastFilter(nn.Module):
    def __init__(self, *, features=500, bins=513, channels=2, sources=4,
                 lags=(1, 2)):
        super().__init__()
        if (not all(type(v) is int and v > 0 for v in
                    (features, bins, channels, sources)) or bins < 3 or sources < 2
                or type(lags) is not tuple or not lags
                or any(type(v) is not int or v <= 0 for v in lags)
                or tuple(sorted(set(lags))) != lags):
            raise ValueError('Require positive geometry and ordered distinct past lags')
        self.features, self.bins = features, bins
        self.channels, self.sources, self.lags = channels, sources, lags
        self.gate = nn.Linear(features, channels * len(lags) * 2 * sources, bias=False)
        nn.init.zeros_(self.gate.weight)

    def initial_history(self, batch):
        if type(batch) is not int or batch <= 0:
            raise ValueError('Require a positive batch size')
        return self.gate.weight.new_zeros(
            (batch, self.channels, max(self.lags), self.bins, 2))

    def _joined(self, carrier, history):
        if (carrier.ndim != 5 or carrier.shape[0] <= 0 or carrier.shape[2] <= 0
                or carrier.shape[1] != self.channels
                or carrier.shape[-2:] != (self.bins, 2)
                or history.shape != (carrier.shape[0], self.channels,
                                     max(self.lags), self.bins, 2)
                or self.gate.weight.dtype not in (torch.float32, torch.float64)
                or any(v.dtype != self.gate.weight.dtype
                       or v.device != self.gate.weight.device for v in (carrier, history))):
            raise ValueError('Invalid carrier or history')
        return torch.cat((history, carrier), dim=2)

    def advance_history(self, carrier, history):
        """Cache-only advancement; this is not a separator warmup implementation."""
        return self._joined(carrier, history)[:, :, -max(self.lags):].clone()

    def forward(self, features, masks, carrier, history):
        joined = self._joined(carrier, history)
        batch, _, frames, _, _ = carrier.shape
        if (features.shape != (batch, frames, self.features)
                or masks.shape != (batch, self.channels, frames, self.bins, 2, self.sources)
                or any(v.dtype != carrier.dtype or v.device != carrier.device
                       for v in (features, masks))):
            raise ValueError('Invalid causal feature or source-mask geometry')
        gates = self.gate(features).tanh().reshape(
            batch, frames, self.channels, len(self.lags), 2, self.sources)
        correction = None
        for tap, lag in enumerate(self.lags):
            gate = gates[:, :, :, tap].permute(0, 2, 1, 3, 4).unsqueeze(3)
            coefficients = masks[:, :, :, 1:-1] * gate
            coefficients = coefficients - coefficients.mean(-1, keepdim=True)
            past = joined[:, :, max(self.lags)-lag:max(self.lags)-lag+frames, 1:-1]
            real = past[..., 0, None] * coefficients[..., 0, :] - past[..., 1, None] * coefficients[..., 1, :]
            imag = past[..., 0, None] * coefficients[..., 1, :] + past[..., 1, None] * coefficients[..., 0, :]
            contribution = torch.stack((real, imag), dim=-2)
            correction = contribution if correction is None else correction + contribution
        correction = F.pad(correction, (0, 0, 0, 0, 1, 1))
        return correction, joined[:, :, -max(self.lags):].clone()
