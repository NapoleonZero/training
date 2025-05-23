import torch
from torch import Tensor
from torch import nn

class WMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, features: Tensor, input: Tensor, target: Tensor) -> Tensor:
        # features sum: Bx12x8x8 -> B
        # input: B
        # target: B
        assert features.shape[1:] == torch.Size([12, 8, 8])
        assert input.shape[1:] == torch.Size([])
        assert target.shape[1:] == torch.Size([])

        scale = features.sum(dim=(1,2,3))
        # return torch.square(scale * (input - target)).mean() # scale before squaring
        return torch.mean(scale * self.mse(input, target)) # scale after squaring
