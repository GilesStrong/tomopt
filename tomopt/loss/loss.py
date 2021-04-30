from torch import nn, Tensor

from ..volume import Volume


class DetectorLoss(nn.MSELoss):
    def __init__(self, cost_coef: float):
        super().__init__()
        self.cost_coef = cost_coef

    def forward(self, pred_x0: Tensor, volume: Volume) -> Tensor:
        return super().forward(pred_x0, volume.get_rad_cube()) + (self.cost_coef * volume.get_cost())
