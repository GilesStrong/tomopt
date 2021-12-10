from typing import Dict, Optional, Union
import numpy as np
from abc import abstractmethod, ABCMeta

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ...volume import Volume

__all__ = ["VoxelX0Loss", "VoxelClassLoss", "VolumeClassLoss"]


class AbsDetectorLoss(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        target_budget: Optional[float],
        budget_smoothing: float = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        steep_budget: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.target_budget, self.budget_smoothing, self.cost_coef, self.steep_budget, self.debug = (
            target_budget,
            budget_smoothing,
            cost_coef,
            steep_budget,
            debug,
        )
        self.sub_losses: Dict[str, Tensor] = {}  # Store subcomponents in dict for telemetry

    def _get_budget_coef(self, cost: Tensor) -> Tensor:
        r"""Switch-on near target budget, plus linear increase above budget"""

        if self.target_budget is None:
            return cost.new_zeros(1)

        if self.steep_budget:
            d = self.budget_smoothing * (cost - self.target_budget) / self.target_budget
            if d <= 0:
                return 2 * torch.sigmoid(d)
            else:
                return 1 + (d / 2)
        else:
            d = cost - self.target_budget
            return (2 * torch.sigmoid(self.budget_smoothing * d / self.target_budget)) + (F.relu(d) / self.target_budget)

    def _compute_cost_coef(self, inference: Tensor) -> None:
        self.cost_coef = inference.detach().clone()
        print(f"Automatically setting cost coefficient to {self.cost_coef}")

    @abstractmethod
    def _get_inference_loss(self, pred: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        pass

    def _get_cost_loss(self, volume: Volume) -> Tensor:
        if self.cost_coef is None:
            self._compute_cost_coef(self.sub_losses["error"])
        cost = volume.get_cost()
        cost_loss = self._get_budget_coef(cost) * self.cost_coef
        if self.debug:
            print(
                f'cost {cost}, cost coef {self.cost_coef}, budget coef {self._get_budget_coef(cost)}. error loss {self.sub_losses["error"]}, cost loss {self.sub_losses["cost"]}'
            )
        return cost_loss

    def forward(self, pred: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        self.sub_losses = {}
        self.sub_losses["error"] = self._get_inference_loss(pred, pred_weight, volume)
        self.sub_losses["cost"] = self._get_cost_loss(volume)
        return self.sub_losses["error"] + self.sub_losses["cost"]


class VoxelX0Loss(AbsDetectorLoss):
    """MSE on X0 float predictions per voxel"""

    def _get_inference_loss(self, pred: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        return torch.mean(F.mse_loss(pred, true_x0, reduction="none") / pred_weight)


class AbsMaterialClassLoss(AbsDetectorLoss):
    """Predictions are class IDs, targets might be float X0"""

    def __init__(
        self,
        *,
        x02id: Dict[float, int],
        target_budget: float,
        budget_smoothing: float = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        steep_budget: bool = True,
        debug: bool = False,
    ):
        super().__init__(target_budget=target_budget, budget_smoothing=budget_smoothing, cost_coef=cost_coef, steep_budget=steep_budget, debug=debug)
        self.x02id = x02id


class VoxelClassLoss(AbsMaterialClassLoss):
    """Predictions are classes per voxel"""

    def _get_inference_loss(self, pred: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        for x0 in true_x0.unique():
            true_x0[true_x0 == x0] = self.x02id[min(self.x02id, key=lambda x: abs(x - x0))]
        true_x0 = true_x0.long().flatten()[None]
        return torch.mean(F.nll_loss(pred, true_x0, reduction="none") / pred_weight)


class VolumeClassLoss(AbsMaterialClassLoss):
    """Predictions are classes of whole volume"""

    def _get_inference_loss(self, pred: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        targ = volume.target.clone()
        for x0 in targ.unique():
            targ[targ == x0] = self.x02id[min(self.x02id, key=lambda x: abs(x - x0))]
        loss = F.nll_loss(pred, targ.long(), reduction="none") if pred.shape[1] > 1 else F.binary_cross_entropy(pred, targ[:, None], reduction="none")
        return torch.mean(loss / pred_weight)
