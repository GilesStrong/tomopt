from typing import Dict, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ...volume import Volume

__all__ = ["DetectorLoss"]


class DetectorLoss(nn.Module):
    sub_losses: Dict[str, Tensor]  # Store subcomponents in dict for telemetry

    def __init__(
        self,
        target_budget: Union[Tensor, float],
        budget_smoothing: Union[Tensor, float] = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.target_budget, self.budget_smoothing, self.cost_coef, self.debug_mode = target_budget, budget_smoothing, cost_coef, debug_mode

    def _get_budget_coef(self, cost: Tensor) -> Tensor:
        r"""Switch-on near target budget, plus linear increase above budget"""
        d = cost - self.target_budget
        return (2 * torch.sigmoid(self.budget_smoothing * d / self.target_budget)) + (F.relu(d) / self.target_budget)

    def _compute_cost_coef(self, cost: Tensor, inference: Tensor) -> None:
        self.cost_coef = inference.detach() / cost.detach()
        print(f"Automatically setting cost coefficient to {self.cost_coef}")

    def forward(self, pred_x0: Tensor, pred_weight: Tensor, volume: Volume) -> Tensor:
        self.sub_losses = {}
        true_x0 = volume.get_rad_cube()
        inference = torch.mean((pred_x0 - true_x0).pow(2) / pred_weight)
        self.sub_losses["error"] = inference
        cost = volume.get_cost()
        if self.cost_coef is None:
            self._compute_cost_coef(cost, inference)
        self.sub_losses["cost"] = self._get_budget_coef(cost) * self.cost_coef * cost
        if self.debug_mode:
            print(
                f'cost {cost}, cost coef {self.cost_coef}, budget coef {self._get_budget_coef(cost)}. error loss {self.sub_losses["error"]}, cost loss {self.sub_losses["cost"]}'
            )
        return self.sub_losses["error"] + self.sub_losses["cost"]
