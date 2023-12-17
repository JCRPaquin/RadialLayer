import torch
import torch.nn as nn


class PowerLayer(nn.Module):

    input_width: int
    power: int

    def __init__(self, input_width: int, power: int):
        super().__init__()

        self.input_width = input_width
        assert power >= 1
        self.power = power

        self.multipliers = nn.Parameter(torch.zeros(1, input_width*power), requires_grad=True)
        self.bn = nn.BatchNorm1d(input_width*power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        powers = torch.hstack([
            x**i for i in range(0, self.power)
        ])
        scaled_powers = powers * (1+self.multipliers)

        return self.bn(scaled_powers)