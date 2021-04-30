from torch import nn, Tensor
import torch.nn.functional as F

from ..volume import Volume

__all__ = ["DetectorLoss"]


class DetectorLoss(nn.Module):
    def __init__(self, cost_coef: float):
        super().__init__()
        self.cost_coef = cost_coef

    def forward(self, pred_x0: Tensor, volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        return F.mse_loss(pred_x0, true_x0) + (self.cost_coef * volume.get_cost())
