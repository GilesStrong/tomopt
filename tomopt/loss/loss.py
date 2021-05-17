from typing import Optional
import torch
from torch import nn, Tensor

from ..volume import Volume

__all__ = ["DetectorLoss"]


class DetectorLoss(nn.Module):
    def __init__(self, cost_coef: float):
        super().__init__()
        self.cost_coef = cost_coef
        self.pred_loss: Optional[Tensor] = None
        self.cost_loss: Optional[Tensor] = None

    def forward(self, pred_x0: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        self.pred_loss = torch.mean((pred_x0 - true_x0).pow(2) * pred_weight)  # MSE/variance
        self.cost_loss = self.cost_coef * volume.get_cost()
        return self.pred_loss + self.cost_loss
