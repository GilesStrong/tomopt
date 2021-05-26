from typing import List

import torch
from torch import Tensor

from .callback import Callback

__all__ = ["ScatterRecord"]


class ScatterRecord(Callback):
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.record: List[Tensor] = []

    def on_train_begin(self) -> None:
        super().on_train_begin()
        self._reset()

    def on_pred_begin(self) -> None:
        super().on_pred_begin()
        self._reset()

    def on_scatter_end(self) -> None:
        self.record.append(self.wrapper.fit_params.sb.location.detach().clone())

    def get_scatter_record(self) -> Tensor:
        return torch.cat(self.record, dim=0)
