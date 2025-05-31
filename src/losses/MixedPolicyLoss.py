import torch
from torch import Tensor
from torch import nn
from typing import Literal

def policy_accuracy(pred_logits: Tensor, target_action: Tensor, top_k: int = -1) -> Tensor:
    preds = torch.argmax(pred_logits, dim=-1)
    correct = (preds[:, :top_k] == target_action[:, :top_k]).float()
    return correct.mean()

def policy_loss(pred_logits: Tensor, target_action: Tensor, weight: float | list[float] = 1.0) -> Tensor | Literal[0]:
    if isinstance(weight, float):
        weight = [weight] * pred_logits.shape[1]

    policy_losses = [
        torch.nn.functional.cross_entropy(pred_logits[:, i], target_action[:, i]) * weight[i]
        for i in range(target_action.shape[1])
    ]

    return sum(policy_losses)

class MixedPolicyLoss(nn.Module):
    def __init__(self, w_mse: float = 1.0, w_cat: float | list[float] = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_mse = w_mse
        self.w_cat = w_cat
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred_score: Tensor, target_score: Tensor, pred_logits: Tensor, target_action: Tensor) -> Tensor:
        # pred_score: B
        # target_score: B
        # pred_logits: Bx10x4096 (unnormalized logits)
        # target_action: Bx10 (integral categorical type, for memory efficiency)

        assert pred_score.shape[1:] == torch.Size([])
        assert target_score.shape[1:] == torch.Size([])

        cat_loss = policy_loss(pred_logits, target_action, self.w_cat)
        return self.w_mse * self.mse(pred_score, target_score) + cat_loss

        # return self.cross_entropy(pred_logits, target_action)
