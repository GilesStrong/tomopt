from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from ...optimisation.callbacks import Callback
from ...optimisation.loss import VolumeIntClassLoss
from ...volume import Volume

__all__ = ["LadleFurnaceIntClassLoss", "SpreadRangeLoss"]


class LadleFurnaceIntClassLoss(VolumeIntClassLoss):
    r"""
    Research tested only: no unit tests
    """

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


class SpreadRangeLoss(Callback):
    r"""
    Research tested only: no unit tests
    """

    def on_volume_batch_begin(self) -> None:
        self._preds: Dict[float, List[Tensor]] = defaultdict(list)

    def on_x0_pred_end(self) -> None:
        self._preds[self.wrapper.volume.target.cpu().item()].append(self.wrapper.fit_params.pred)

    def on_volume_batch_end(self) -> None:
        stds = []
        means = []
        for preds in self._preds.values():
            if len(preds) < 2:
                continue
            p = torch.cat(preds)
            stds.append(torch.std(p))
            means.append(torch.mean(p))

        spread = torch.mean(torch.stack(stds))
        range_ = torch.std(torch.stack(means))

        loss = spread / range_

        self.wrapper.fit_params.mean_loss = loss[None]
