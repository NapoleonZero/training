import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss

class WDLLoss(nn.Module):
    """ Implements Win-Draw-Loss loss function"""
    def __init__(self, input_scale: float = 410.0, target_scale: float = 410.0):
        super().__init__()
        self.input_scale = input_scale
        self.target_scale = target_scale
        self.bce = BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce(input / self.input_scale, torch.sigmoid(target / self.target_scale))

