from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .appearance import H_MID, LBL_COL, LBL_SZ

__all__ = ["plot_scatter_density", "plot_hit_density"]


def plot_scatter_density(scatter_df: pd.DataFrame, savename: Optional[str] = None) -> None:
    r"""
    Plots the position of PoCAs in the passive volume, as recorded using :class:`~tomopt.optimisation.callbacks.diagnostic_callbacks.ScatterRecord`.

    Arguments:
        scatter_df: Dataframe of recorded PoCAs, as returned by :meth:`~tomopt.optimisation.callbacks.diagnostic_callbacks.ScatterRecord.get_record`
        savename: optional savename to save the plot
    """

    with sns.axes_style(style="whitegrid", rc={"patch.edgecolor": "none"}):
        zs = sorted(scatter_df.layer.unique())
        n = len(zs)
        fig, axs = plt.subplots(n, 1, figsize=(2 * H_MID / n, 2 * H_MID))
        for i, z in enumerate(zs):
            sns.histplot(data=scatter_df[(scatter_df.layer == z)], x="x", y="y", cmap="viridis", ax=axs[i], cbar=True)
            axs[i].set_ylabel(f"AbsLayer {z}", fontsize=LBL_SZ, color=LBL_COL)
            axs[i].set_xlabel("", fontsize=LBL_SZ, color=LBL_COL)
        if savename is not None:
            plt.savefig(savename, bbox_inches="tight")
        plt.show()


def plot_hit_density(hit_df: pd.DataFrame, savename: Optional[str] = None) -> None:
    r"""
    Plots the position of muon hits in the detectors, as recorded using :class:`~tomopt.optimisation.callbacks.diagnostic_callbacks.HitRecord`.

    Arguments:
        hit_df: Dataframe of recorded hits, as returned by :meth:`~tomopt.optimisation.callbacks.diagnostic_callbacks.HitRecord.get_record`
        savename: optional savename to save the plot
    """

    plot_scatter_density(hit_df, savename)
