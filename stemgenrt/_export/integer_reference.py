"""Independent CPU arithmetic for signed integer projection verification."""
import torch
from torch import nn
from .helpers import require

class PreciseProjection(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("weight", original.centered_weight.detach().clone())
        self.register_buffer("scale", original.scale.detach().double().clone())
        self.register_buffer("bias", None if original.bias is None else original.bias.detach().double().clone())

    def forward(self, values):
        require(values.dtype == torch.float64 and values.device.type == "cpu", "Require CPU FP64 values")
        x = values.float()
        minimum, maximum = x.min().clamp_max(0), x.max().clamp_min(0)
        scale = (maximum - minimum) / 255.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(-minimum / scale).clamp(0, 255).to(torch.int32)
        quantized = (torch.round(x / scale) + zero).clamp(0, 255).to(torch.int32)
        integer = (quantized.reshape(1, -1) - zero) @ self.weight
        output = integer.float() * (scale * self.scale.float())
        if self.bias is not None:
            output = output + self.bias.float()
        return output.double().reshape(*x.shape[:-1], self.weight.shape[1])

class IntegerGRU(nn.Module):

    def forward(self, x, hidden):
        current, returned = (x[:, 0], [])
        for layer in range(2):
            ir, iz, inn = self.input_maps[layer](current).chunk(3, dim=-1)
            hr, hz, hn = self.hidden_maps[layer](hidden[layer]).chunk(3, dim=-1)
            reset, update = ((ir + hr).sigmoid(), (iz + hz).sigmoid())
            candidate = (inn + reset * hn).tanh()
            current = candidate + update * (hidden[layer] - candidate)
            returned.append(current)
        return (current[:, None], torch.stack(returned))
