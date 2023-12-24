"""
Quad-input activation functions

Each custom activation function is limited to 4 input channels
because I wanted them to be JIT compatible.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PowerProductPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, channel_width = x.size()
        assert channels == 4

        x_1, x_2, x_3, x_4 = torch.unbind(x, dim=1)
        return (x_1 + 1) * (x_2 + 1) * (x_3 + 1) * (x_4 + 1) - 1


class PairwisePowerProductPool(nn.Module):

    def __init__(self, channel_width: int):
        super().__init__()

        self.channel_width = channel_width
        self.coefficients = nn.Parameter(torch.ones(1, 6*channel_width), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, channel_width = x.size()
        assert channels == 4

        x_1, x_2, x_3, x_4 = torch.unbind(x, dim=1)
        return torch.hstack([
            (x_1 + 1) * (x_2 + 1) - 1,
            (x_1 + 1) * (x_3 + 1) - 1,
            (x_1 + 1) * (x_4 + 1) - 1,
            (x_2 + 1) * (x_3 + 1) - 1,
            (x_2 + 1) * (x_4 + 1) - 1,
            (x_3 + 1) * (x_4 + 1) - 1
        ]) * self.coefficients
