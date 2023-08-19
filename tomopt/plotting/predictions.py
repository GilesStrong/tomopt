from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .appearance import H_MID, LBL_COL, LBL_SZ, W_MID

__all__ = ["plot_pred_true_x0"]


def plot_pred_true_x0(pred: np.ndarray, true: np.ndarray, savename: Optional[str] = None) -> None:
    r"""
    Plots the predicted voxelwise X0s compared to the true values of the X0s.
    2D plots are produced in xy for layers in z in order of increasing z, i.e. the bottom most layer is the first to be plotted.
    TODO: revise this ordering to make it more intuitive

    Arguments:
        pred: (z,x,y) array of predicted X0s
        true: (z,x,y) array of true X0s
        savename: optional savename for saving the plot
    """

    with sns.axes_style(style="whitegrid", rc={"patch.edgecolor": "none"}):
        fig, axs = plt.subplots(len(pred), 2, figsize=(H_MID, W_MID))
        pred_cbar_ax = fig.add_axes([0.45, 0.25, 0.03, 0.5])
        true_cbar_ax = fig.add_axes([0.90, 0.25, 0.03, 0.5])

        for plot_idx in range(len(pred)):
            layer_idx = len(pred) - 1 - plot_idx
            sns.heatmap(
                pred[layer_idx],
                ax=axs[plot_idx][0],
                cmap="viridis",
                square=True,
                cbar=(plot_idx == 0),
                vmin=np.nanmin(pred),
                vmax=np.nanmax(pred),
                cbar_ax=pred_cbar_ax if plot_idx == 0 else None,
            )
            sns.heatmap(
                true[layer_idx],
                ax=axs[plot_idx][1],
                cmap="viridis",
                square=True,
                cbar=(plot_idx == 0),
                vmin=true.min(),
                vmax=true.max(),
                cbar_ax=true_cbar_ax if plot_idx == 0 else None,
            )
            axs[plot_idx][0].set_ylabel(f"AbsLayer {layer_idx}", fontsize=LBL_SZ, color=LBL_COL)
        axs[-1][0].set_xlabel("Prediction", fontsize=LBL_SZ, color=LBL_COL)
        axs[-1][1].set_xlabel("True", fontsize=LBL_SZ, color=LBL_COL)
        if savename is not None:
            plt.savefig(savename, bbox_inches="tight")
        plt.show()
