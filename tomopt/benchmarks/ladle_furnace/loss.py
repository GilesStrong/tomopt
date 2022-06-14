from typing import Optional, Union

import torch
from torch import Tensor

from ...volume import Volume
from ...optimisation.loss import VolumeIntClassLoss


__all__ = ["LadleFurnaceIntClassLoss"]


class LadleFurnaceIntClassLoss(VolumeIntClassLoss):
    def __init__(
        self,
        *,
        pred_int_start: int = 0,
        use_mse: bool,
        target_budget: float,
        budget_smoothing: float = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        steep_budget: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            targ2int=self._targ2int,
            pred_int_start=pred_int_start,
            use_mse=use_mse,
            target_budget=target_budget,
            budget_smoothing=budget_smoothing,
            cost_coef=cost_coef,
            steep_budget=steep_budget,
            debug=debug,
        )

    @staticmethod
    def _targ2int(targs: Tensor, volume: Volume) -> Tensor:
        return (
            torch.div((targs - volume.get_passive_z_range()[0]), volume.passive_size, rounding_mode="floor") - 1
        )  # -1 due to conversion to layer ID, instead of fill height
