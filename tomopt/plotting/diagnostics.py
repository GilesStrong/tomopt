from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .appearance import H_MID, LBL_COL, LBL_SZ

__all__ = ["plot_scatter_density"]


def plot_scatter_density(scatter_df: pd.DataFrame, savename: Optional[str] = None) -> None:
    with sns.axes_style(style="whitegrid", rc={"patch.edgecolor": "none"}):
        zs = sorted(scatter_df.layer.unique())
        n = len(zs)
        fig, axs = plt.subplots(n, 1, figsize=(H_MID / n, H_MID))
        for i, z in enumerate(zs):
            sns.histplot(data=scatter_df[(scatter_df.layer == z)], x="x", y="y", cmap="viridis", ax=axs[i], cbar=True)
            axs[i].set_ylabel(f"Layer {z}", fontsize=LBL_SZ, color=LBL_COL)
            axs[i].set_xlabel("", fontsize=LBL_SZ, color=LBL_COL)
        if savename is not None:
            plt.savefig(savename, bbox_inches="tight")
        plt.show()
