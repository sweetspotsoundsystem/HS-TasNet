"""Independent complex-arithmetic reference for the FP32 past-filter path."""
import torch
from torch.nn import functional as F
from .helpers import require


def correction(weight, features, masks, carrier, history):
    require(history.shape == (1, 2, 2, 513, 2) and history.dtype == torch.float32
            and history.device.type == 'cpu', 'Invalid past-carrier reference state')
    gates = F.linear(features.float(), weight.float()).tanh().reshape(1, 1, 2, 2, 2, 4)
    residual = torch.zeros((1, 2, 1, 513, 2, 4), dtype=torch.float32)
    for index in range(2):
        gate = gates[:, :, :, index].permute(0, 2, 1, 3, 4).unsqueeze(3)
        coefficients = masks[:, :, :, 1:-1] * gate
        coefficients = coefficients - coefficients.mean(-1, keepdim=True)
        coefficient = torch.complex(coefficients[..., 0, :], coefficients[..., 1, :])
        previous = history[:, :, 1-index:2-index, 1:-1]
        value = torch.complex(previous[..., 0], previous[..., 1]).unsqueeze(-1) * coefficient
        residual[:, :, :, 1:-1] += torch.stack((value.real, value.imag), dim=-2)
    return residual, torch.cat((history[:, :, 1:], carrier), dim=2)
