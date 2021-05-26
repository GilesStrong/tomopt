from typing import List, Union
import pandas as pd
import numpy as np

import torch
from torch import Tensor

from .callback import Callback

__all__ = ["ScatterRecord", "HitRecord"]


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

    def _to_df(self, record: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(record.numpy(), columns=["x", "y", "z"])
        df["layer"] = pd.cut(
            self.wrapper.volume.h - df.z,
            np.linspace(*self.wrapper.volume.get_passive_z_range(), 1 + len(self.wrapper.volume.get_passives())).squeeze(),
            labels=False,
        )
        return df

    def get_record(self, as_df: bool = False) -> Union[Tensor, pd.DataFrame]:
        record = torch.cat(self.record, dim=0)
        if as_df:
            return self._to_df(record)
        else:
            return record


class HitRecord(ScatterRecord):
    def on_scatter_end(self) -> None:
        hits = torch.stack(
            [
                self.wrapper.fit_params.sb.xa0.detach().cpu().clone(),
                self.wrapper.fit_params.sb.xa1.detach().cpu().clone(),
                self.wrapper.fit_params.sb.xb1.detach().cpu().clone(),
                self.wrapper.fit_params.sb.xb0.detach().cpu().clone(),
            ],
            dim=1,
        )
        self.record.append(hits)

    def _to_df(self, record: Tensor) -> pd.DataFrame:
        df = pd.DataFrame(record.reshape(-1, 3).numpy(), columns=["x", "y", "z"])
        df["layer"] = (self.wrapper.volume.h - df.z).astype("category").cat.codes  # df ordered by reshapeing hits
        return df
