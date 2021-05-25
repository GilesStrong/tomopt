from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_pred_true_x0"]


TK_SZ = 16
TK_COL = "black"
LBL_SZ = 24
LBL_COL = "black"
LEG_SZ = 16
CAT_PALETTE = "tab10"
STYLE = {"style": "whitegrid", "rc": {"patch.edgecolor": "none"}}
H_MID = 8
W_MID = H_MID * 16 / 9


def plot_pred_true_x0(pred: np.ndarray, true: np.ndarray, savename: Optional[str] = None) -> None:
    with sns.axes_style(style="whitegrid", rc={"patch.edgecolor": "none"}):
        fig, axs = plt.subplots(len(pred), 2, figsize=(H_MID, W_MID))
        pred_cbar_ax = fig.add_axes([0.45, 0.25, 0.03, 0.5])
        true_cbar_ax = fig.add_axes([0.90, 0.25, 0.03, 0.5])

        for i in range(len(pred)):
            sns.heatmap(
                pred[i],
                ax=axs[i][0],
                cmap="viridis",
                square=True,
                cbar=(i == 0),
                vmin=np.nanmin(pred),
                vmax=np.nanmax(pred),
                cbar_ax=pred_cbar_ax if i == 0 else None,
            )
            sns.heatmap(
                true[i], ax=axs[i][1], cmap="viridis", square=True, cbar=(i == 0), vmin=true.min(), vmax=true.max(), cbar_ax=true_cbar_ax if i == 0 else None
            )

            axs[i][0].set_ylabel(f"Layer {i}", fontsize=LBL_SZ, color=LBL_COL)
        axs[-1][0].set_xlabel("Prediction", fontsize=LBL_SZ, color=LBL_COL)
        axs[-1][1].set_xlabel("True", fontsize=LBL_SZ, color=LBL_COL)
        if savename is not None:
            plt.savefig(savename, bbox_inches="tight")
        plt.show()
