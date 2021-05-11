import torch
from torch import nn, Tensor

from ..volume import Volume

__all__ = ["DetectorLoss"]


class DetectorLoss(nn.Module):
    def __init__(self, cost_coef: float):
        super().__init__()
        self.cost_coef = cost_coef

    def forward(self, pred_x0: Tensor, pred_weights: Tensor, volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        pred_loss = torch.mean((pred_x0 - true_x0).pow(2) * pred_weights)  # MSE/variance
        det_loss = self.cost_coef * volume.get_cost()
        return pred_loss + det_loss
