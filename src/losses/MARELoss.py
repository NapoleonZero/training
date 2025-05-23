import torch
from torch import nn

def mare_loss(input, target, delta=1.0):
        return torch.mean(torch.abs(input - target).div(target.abs().clamp(min=delta)))

def smare_loss(input, target, delta=1.0):
        return torch.mean(torch.abs(input - target).div( (input.abs() + target.abs()).div(2).clamp(min=delta)))

class MARELoss(nn.Module):
    """ Implements Mean Absolute Relative Error Loss """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        return mare_loss(input, target, delta=self.delta)

class SMARELoss(MARELoss):
    def forward(self, input, target):
        return smare_loss(input, target, delta=self.delta)
