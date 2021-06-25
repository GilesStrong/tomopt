from typing import Dict, Optional, List
import torch
from torch import nn, Tensor

from ...volume import Volume

__all__ = ["DetectorLoss"]


class DetectorLoss(nn.Module):
    def __init__(self, cost_coef: Optional[float] = None):
        super().__init__()
        self.cost_coef = cost_coef
        self.sub_losses: Dict[str, Tensor] = {}  # Store subcomponents in dict for telemetry

    def _compute_cost_coef(self, cost: Tensor, inference: Tensor) -> None:
        self.cost_coef = inference.detach() / cost.detach()

    def _compute_weight(self, weight_comps: Dict[str, List[Tensor]]) -> Tensor:
        eff = torch.stack(weight_comps["efficiency"], dim=0)
        prob = torch.stack(weight_comps["scatter_prob"], dim=0)
        unc = torch.stack(weight_comps["x0_variance"], dim=0)
        wgt = (prob * eff) / ((1e-17) + unc)
        return wgt.sum(0)

    def forward(self, pred_x0: Tensor, weight_comps: Dict[str, List[Tensor]], volume: Volume) -> Tensor:
        true_x0 = volume.get_rad_cube()
        wgt = self._compute_weight(weight_comps)
        print(wgt, torch.stack(weight_comps["weight"], dim=0).sum(0))
        inference = torch.mean((pred_x0 - true_x0).pow(2) * wgt)  # SE*efficiency/variance TODO fix this
        self.sub_losses["error"] = inference
        cost = volume.get_cost()
        if self.cost_coef is None:
            self._compute_cost_coef(cost, inference)
        self.sub_losses["cost"] = self.cost_coef * cost
        return self.sub_losses["error"] + self.sub_losses["cost"]
