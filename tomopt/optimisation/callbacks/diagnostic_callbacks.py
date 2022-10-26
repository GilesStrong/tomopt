from typing import List, Union
import pandas as pd
import numpy as np

import torch
from torch import Tensor

from .callback import Callback

r"""
Provides callbacks deigned to record diagnostic information
"""

__all__ = ["ScatterRecord", "HitRecord"]


class ScatterRecord(Callback):
    r"""
    Records the PoCAs of the muons which are located inside the passive volume.
    Once recorded, the PoCAs can be retrieved via the :meth:`~tomopt.optimisation.callbacks.diagnostic_callbacks.ScatterRecord.get_record` method.
    :meth:`~tomopt.plotting.diagnostics.plot_scatter_density` may be used to plot the scatter record.

    .. warning::
        Currently this callback makes no distinciton between different volume layouts, and is designed to used over a single volume layout.

    # TODO extend these to create one record per volume
    """

    def __init__(self) -> None:
        r"""
        Initialises the callback and prepares to record scatters
        """

        self._reset()

    def on_train_begin(self) -> None:
        r"""
        Prepares to record scatters
        """

        super().on_train_begin()
        self._reset()

    def on_pred_begin(self) -> None:
        r"""
        Prepares to record scatters
        """

        super().on_pred_begin()
        self._reset()

    def on_scatter_end(self) -> None:
        r"""
        Saves the PoCAs of the latest muon batch.
        """

        self.record.append(self.wrapper.fit_params.sb.poca_xyz[self.wrapper.fit_params.sb.get_scatter_mask()].detach().cpu().clone())

    def _to_df(self, record: Tensor) -> pd.DataFrame:
        r"""
        Converts the saved PoCAs to a Pandas DataFrame

        Arguments:
            record: (N,xyz) tensor of PoCAs

        Returns:
            DataFrame of PoCAs
        """

        df = pd.DataFrame(record.numpy(), columns=["x", "y", "z"])
        dw, up = self.wrapper.volume.get_passive_z_range()
        df["layer"] = pd.cut(
            self.wrapper.volume.h.detach().cpu().item() - df.z,
            np.linspace(dw.detach().cpu().item(), up.detach().cpu().item(), 1 + len(self.wrapper.volume.get_passives())).squeeze(),
            labels=False,
        )
        return df

    def get_record(self, as_df: bool = False) -> Union[Tensor, pd.DataFrame]:
        r"""
        Accessthe recorded PoCAs.

        Arguments:
            as_df: if True, will return a Pandas DataFrame, otherwise will return a Tensor

        Returns:
            a Pandas DataFrame or a Tensor of recorded PoCAs
        """
        record = torch.cat(self.record, dim=0)
        if as_df:
            return self._to_df(record)
        else:
            return record

    def _reset(self) -> None:
        r"""
        Prepares to record scatters
        """

        self.record: List[Tensor] = []


class HitRecord(ScatterRecord):
    r"""
    Records the hits of the muons.
    Once recorded, the hits can be retrieved via the :meth:`~tomopt.optimisation.callbacks.diagnostic_callbacks.HitRecord.get_record` method.
    :meth:`~tomopt.plotting.diagnostics.plot_hit_density` may be used to plot the hit record.

    .. warning::
        Currently this callback makes no distinciton between different volume layouts, and is designed to used over a single volume layout.

    # TODO extend these to create one record per volume
    """

    def on_scatter_end(self) -> None:
        r"""
        Saves the hits of the latest muon batch.
        """

        hits = self.wrapper.fit_params.sb._reco_hits.detach().cpu().clone()
        self.record.append(hits)

    def _to_df(self, record: Tensor) -> pd.DataFrame:
        r"""
        Converts the saved hits to a Pandas DataFrame

        Arguments:
            record: (N,xyz) tensor of hits

        Returns:
            DataFrame of hits
        """

        df = pd.DataFrame(record.reshape(-1, 3).numpy(), columns=["x", "y", "z"])
        df["layer"] = (self.wrapper.volume.h.detach().cpu().item() - df.z).astype("category").cat.codes  # df ordered by reshapeing hits
        return df
