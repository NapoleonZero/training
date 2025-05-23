import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss

class WDLLoss(nn.Module):
    """ Implements Win-Draw-Loss loss function"""
    def __init__(self, input_scale: float = 0.4, target_scale: float = 0.4):
        super().__init__()
        self.input_scale = input_scale
        self.target_scale = target_scale
        self.bce = BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce(input * self.input_scale, torch.sigmoid(target * self.target_scale))


class WDLMseLoss(WDLLoss):
    # TODO: change w1, w2 names
    def __init__(self, *args, w1 = 0.5, w2 = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.w1 = w1
        self.w2 = w2
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.w1 * self.mse(input, target) + self.w2 * super().forward(input, target)
