from typing import Dict
import torch
from torch import nn, Tensor

from ...volume import Volume

__all__ = ["DetectorLoss"]


class DetectorLoss(nn.Module):
    def __init__(self, cost_coef: float):
        super().__init__()
        self.cost_coef = cost_coef
        self.sub_losses: Dict[str, Tensor] = {}  # Store subcomponents in dict for telemetry

    def forward(self, pred_x0: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        self.sub_losses["error"] = torch.mean((pred_x0 - true_x0).pow(2) * pred_weight)  # SE/variance
        self.sub_losses["cost"] = self.cost_coef * volume.get_cost()
        return self.sub_losses["error"] + self.sub_losses["cost"]
