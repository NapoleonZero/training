import torch
from torch import nn

class MSRELoss(nn.Module):
    """ Implements Mean Squared Relative Error Loss """
    def __init__(self, delta=1):
        super(MSRELoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        # return torch.mean(torch.div(torch.abs(input - target), torch.abs(target) + self.delta))
        return torch.mean((torch.log(torch.clamp(input, min=(-self.delta + 1)) + self.delta) - torch.log(torch.clamp(target, min=(-self.delta + 1)) + self.delta))**2)

        # return torch.mean(torch.div((input - target)**4, target**2 + self.eps))
        # return torch.mean(torch.div((input - target)**2, target**2 + 1e-2))
        # return torch.mean(torch.log((input - target)**2))
        # return torch.mean((input - target)**2)
        # return torch.mean(torch.log(torch.cosh(input - target)))
