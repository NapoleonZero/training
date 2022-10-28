import torch
from torch import nn

class ClippedReLU(nn.Module):
    def __init__(self, max_val=1.0, *args, **kwargs):
        super().__init__()
        self.activation = nn.Hardtanh(min_val=0, max_val=max_val, *args, **kwargs)

    def forward(self, x):
        return self.activation(x)

