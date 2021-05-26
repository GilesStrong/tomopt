from typing import List, Union
import pandas as pd
import numpy as np

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
        self.record.append(self.wrapper.fit_params.sb.location[self.wrapper.fit_params.sb.get_scatter_mask()].detach().cpu().clone())

    def get_scatter_record(self, as_df: bool = False) -> Union[Tensor, pd.DataFrame]:
        scatters = torch.cat(self.record, dim=0)
        if as_df:
            df = pd.DataFrame(scatters.numpy(), columns=["x", "y", "z"])
            df["layer"] = pd.cut(
                self.wrapper.volume.h - df.z,
                np.linspace(*self.wrapper.volume.get_passive_z_range(), 1 + len(self.wrapper.volume.get_passives())).squeeze(),
                labels=False,
            )
            return df
        else:
            return scatters
