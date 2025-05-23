import torch
from torch import nn

class MSLELoss(nn.Module):
    """ Implements Mean Squared Log Error Loss """
    def __init__(self, delta=1):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        # TODO: review old approaches

        # return torch.mean(torch.div(torch.abs(input - target), torch.abs(target) + self.delta))
        # return torch.mean((torch.log(torch.clamp(input, min=(-self.delta + 1)) + self.delta) - torch.log(torch.clamp(target, min=(-self.delta + 1)) + self.delta))**2)
        # return torch.mean(torch.div((input - target)**4, target**2 + self.eps))
        # return torch.mean(torch.div((input - target)**2, target**2 + 1e-2))
        # return torch.mean(torch.log((input - target)**2))
        # return torch.mean((input - target)**2)
        # return torch.mean(torch.log(torch.cosh(input - target)))

        # 1) Shift centipawns score to a non-negative scale by adding the maximum (absolute) value
        # 2) Add 1 to make the score positive
        # 3) Take the logarithm
        # 4) Take the mean of the squared differences
        return torch.mean((input.add(1 + self.delta).log() - target.add(1 + self.delta).log()).square())

