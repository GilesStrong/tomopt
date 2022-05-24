from __future__ import annotations
from fastprogress.fastprogress import IN_NOTEBOOK
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict
import os
import imageio

if IN_NOTEBOOK:
    from IPython.display import display

from .callback import Callback
from .eval_metric import EvalMetric
from ...volume import VoxelDetectorLayer, PanelDetectorLayer

if TYPE_CHECKING:
    from ...optimisation.wrapper.volume_wrapper import AbsVolumeWrapper

r"""
This MetricLogger is a modified version of the MetricLogger in LUMIN (https://github.com/GilesStrong/lumin/blob/v0.7.2/lumin/nn/callbacks/monitors.py#L125), distributed under the following lincence:
    Copyright 2018 onwards Giles Strong

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Usage is compatible with the AGPL licence underwhich TomOpt is distributed.
Stated changes: adaption to work with `VolumeWrapper` class, replacement of the telemtry plots with task specific information.
"""

__all__ = ["MetricLogger", "VoxelMetricLogger", "PanelMetricLogger"]


class MetricLogger(Callback):
    r"""
    Provides live feedback during training showing a variety of metrics to help highlight problems or test hyper-parameters without completing a full training.
    If `show_plots` is false, will instead print training and validation losses at the end of each epoch.
    The full history is available as a dictionary by calling `MetricLogger.get_loss_history`.
    """

    tk_sz = 16
    tk_col = "black"
    lbl_sz = 24
    lbl_col = "black"
    leg_sz = 16
    cat_palette = "tab10"
    style = {"style": "whitegrid", "rc": {"patch.edgecolor": "none"}}
    h_mid = 8
    w_mid = h_mid * 16 / 9

    def __init__(self, gif_filename: Optional[str] = None, show_plots: bool = IN_NOTEBOOK):
        self.gif_filename, self.show_plots = gif_filename, show_plots

    def _reset(self) -> None:
        self.loss_vals: Dict[str, List[float]] = {"Training": [], "Validation": []}
        self.sub_losses: Dict[str, List[float]] = defaultdict(list)
        self.best_loss: float = math.inf
        self.val_epoch_results: Optional[Tuple[float, Optional[float]]] = None
        self.metric_cbs: List[EvalMetric] = []
        self.n_trn_batches = len(self.wrapper.fit_params.trn_passives) // self.wrapper.fit_params.passive_bs
        self._buffer_files: List[str] = []

        self.metric_vals: List[List[float]] = [[] for _ in self.wrapper.fit_params.metric_cbs]
        self.main_metric_idx: Optional[int] = None
        self.lock_to_metric: bool = False
        if len(self.wrapper.fit_params.metric_cbs) > 0:
            self.main_metric_idx = 0
            for i, c in enumerate(self.wrapper.fit_params.metric_cbs):
                if c.main_metric:
                    self.main_metric_idx = i
                    self.lock_to_metric = True
                    break
        self._prep_plots()
        if self.show_plots:
            self.display = display(self.fig, display_id=True)

    def _build_grid_spec(self) -> GridSpec:
        return self.fig.add_gridspec(3 + (self.main_metric_idx is None), 1)

    def _prep_plots(self) -> None:
        if self.show_plots:
            with sns.axes_style(**self.style):
                self.fig = plt.figure(figsize=(self.w_mid, self.w_mid), constrained_layout=True)
                self.grid_spec = self._build_grid_spec()
                self.loss_ax = self.fig.add_subplot(self.grid_spec[:3, :])
                self.sub_loss_ax = self.fig.add_subplot(self.grid_spec[3:4, :])
                if self.main_metric_idx is not None:
                    self.metric_ax = self.fig.add_subplot(self.grid_spec[4:5, :])
                for ax in [self.loss_ax, self.sub_loss_ax]:
                    ax.tick_params(axis="x", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                    ax.tick_params(axis="y", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                self.sub_loss_ax.set_xlabel("Epoch", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                self.loss_ax.set_ylabel("Loss", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                self.sub_loss_ax.set_ylabel("Loss Composition", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                if self.main_metric_idx is not None:
                    self.metric_ax.tick_params(axis="x", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                    self.metric_ax.tick_params(axis="y", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                    self.metric_ax.set_xlabel("Epoch", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                    self.metric_ax.set_ylabel(self.wrapper.fit_params.metric_cbs[self.main_metric_idx].name, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

    def on_train_begin(self) -> None:
        r"""
        Prepare for new training
        """

        super().on_train_begin()
        self._reset()

    def _snapshot_monitor(self) -> None:
        self._buffer_files.append(self.wrapper.fit_params.cb_savepath / f"temp_monitor_{len(self._buffer_files)}.png")
        self.fig.savefig(self._buffer_files[-1], bbox_inches="tight")

    def on_epoch_begin(self) -> None:
        r"""
        Prepare to track new loss
        """

        self.tmp_loss, self.batch_cnt, self.volume_cnt = 0.0, 0, 0
        self.tmp_sub_losses: Dict[str, float] = defaultdict(float)
        if self.gif_filename is not None and self.wrapper.fit_params.state == "train":
            self._snapshot_monitor()

    def on_volume_end(self) -> None:
        if self.wrapper.fit_params.state == "valid" and self.wrapper.loss_func is not None and hasattr(self.wrapper.loss_func, "sub_losses"):
            if self.wrapper.fit_params.pred is not None:  # Was able to scan volume
                for k, v in self.wrapper.loss_func.sub_losses.items():
                    self.tmp_sub_losses[k] += v.data.item()
                self.volume_cnt += 1
            else:
                for k, v in self.wrapper.loss_func.sub_losses.items():
                    self.tmp_sub_losses[k] += 0  # Create sub loss at 0 or add zero if exists

    def on_backwards_end(self) -> None:
        if self.wrapper.fit_params.state == "train":
            self.loss_vals["Training"].append(self.wrapper.fit_params.mean_loss.data.item() if self.wrapper.fit_params.mean_loss is not None else math.inf)

    def on_volume_batch_end(self) -> None:
        if self.wrapper.fit_params.state == "valid":
            self.tmp_loss += self.wrapper.fit_params.mean_loss.data.item() if self.wrapper.fit_params.mean_loss is not None else math.inf
            self.batch_cnt += 1

    def on_epoch_end(self) -> None:
        r"""
        If validation epoch finished, record validation losses, compute info and update plots
        """

        if self.wrapper.fit_params.state == "valid":
            self.loss_vals["Validation"].append(self.tmp_loss / self.batch_cnt)
            for k, v in self.tmp_sub_losses.items():
                self.sub_losses[k].append(v / (self.loss_vals["Validation"][-1] * self.volume_cnt))  # Fractional components

            for i, c in enumerate(self.wrapper.fit_params.metric_cbs):
                self.metric_vals[i].append(c.get_metric())
            if self.loss_vals["Validation"][-1] <= self.best_loss:
                self.best_loss = self.loss_vals["Validation"][-1]

            if self.show_plots:
                self.update_plot()
                self.display.update(self.fig)
            else:
                self.print_losses()

            m = None
            if self.lock_to_metric:
                m = self.metric_vals[self.main_metric_idx][-1]
                if not self.wrapper.fit_params.metric_cbs[self.main_metric_idx].lower_metric_better:
                    m *= -1
            self.val_epoch_results = self.loss_vals["Validation"][-1], m

    def print_losses(self) -> None:
        r"""
        Print training and validation losses for the last epoch
        """

        p = f'Epoch {len(self.loss_vals["Validation"])}: '
        p += f'Training = {np.mean(self.loss_vals["Training"][-self.n_trn_batches:]):.2E} | '
        p += f'Validation = {self.loss_vals["Validation"][-1]:.2E}'
        for m, v in zip(self.wrapper.fit_params.metric_cbs, self.metric_vals):
            p += f" {m.name} = {v[-1]:.2E}"
        print(p)

    def update_plot(self) -> None:
        r"""
        Updates the plot(s).
        """

        # Loss
        self.loss_ax.clear()
        self.sub_loss_ax.clear()
        with sns.axes_style(**self.style), sns.color_palette(self.cat_palette):
            self.loss_ax.plot(
                (1 / self.n_trn_batches)
                + np.linspace(0, len(self.loss_vals["Validation"]), self.n_trn_batches * len(self.loss_vals["Validation"]), endpoint=False),
                self.loss_vals["Training"],
                label="Training",
            )
            x = range(1, len(self.loss_vals["Validation"]) + 1)
            self.loss_ax.plot(x, self.loss_vals["Validation"], label="Validation")
            keys = sorted([k for k in self.sub_losses])
            self.sub_loss_ax.stackplot(x, *[self.sub_losses[k] for k in keys], labels=keys)
            self.loss_ax.plot([1 / self.n_trn_batches, x[-1]], [self.best_loss, self.best_loss], label=f"Best = {self.best_loss:.3E}", linestyle="--")
            self.loss_ax.legend(loc="upper right", fontsize=0.8 * self.leg_sz)
            self.sub_loss_ax.legend(loc="upper left", fontsize=0.8 * self.leg_sz)
            for ax in [self.loss_ax, self.sub_loss_ax]:
                ax.grid(True, which="both")
                ax.set_xlim(1 / self.n_trn_batches, x[-1])
            self.sub_loss_ax.set_xlabel("Epoch", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
            self.loss_ax.set_ylabel("Loss", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
            self.sub_loss_ax.set_ylabel("Loss Composition", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

        if len(self.loss_vals["Validation"]) > 1:
            # Metrics
            if self.main_metric_idx is not None:
                self.metric_ax.clear()
                with sns.axes_style(**self.style), sns.color_palette(self.cat_palette) as palette:
                    x = range(self.n_trn_batches, self.n_trn_batches * len(self.loss_vals["Validation"]) + 1, self.n_trn_batches)
                    y = self.metric_vals[self.main_metric_idx]
                    self.metric_ax.plot(x, y, color=palette[1])
                    best = np.nanmin(y) if self.wrapper.fit_params.metric_cbs[self.main_metric_idx].lower_metric_better else np.nanmax(y)
                    self.metric_ax.plot([1, x[-1]], [best, best], label=f"Best = {best:.3E}", linestyle="--", color=palette[2])
                    self.metric_ax.legend(loc="upper left", fontsize=0.8 * self.leg_sz)
                    self.metric_ax.grid(True, which="both")
                    self.metric_ax.set_xlim(1 / self.n_trn_batches, x[-1])
                    self.metric_ax.set_xlabel("Epoch", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                    self.metric_ax.set_ylabel(self.wrapper.fit_params.metric_cbs[self.main_metric_idx].name, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

    def _create_gif(self) -> None:
        with imageio.get_writer(self.wrapper.fit_params.cb_savepath / self.gif_filename, mode="I") as writer:
            for filename in self._buffer_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self._buffer_files):
            os.remove(filename)

    def on_train_end(self) -> None:
        if self.gif_filename is not None:
            self._snapshot_monitor()
            self._create_gif()
        plt.clf()  # prevent plot be shown twice
        self.metric_cbs = self.wrapper.fit_params.metric_cbs  # Copy referenece since fit_params gets set to None at end of training

    def get_loss_history(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        r"""
        Get the current history of losses and metrics

        Returns:
            history: tuple of ordered dictionaries: first with losses, second with validation metrics
        """

        history: Tuple[Dict[str, List[float]], Dict[str, List[float]]] = ({}, {})
        history[0]["Training"] = self.loss_vals["Training"]
        history[0]["Validation"] = self.loss_vals["Validation"]
        for v, c in zip(self.metric_vals, self.metric_cbs):
            history[1][c.name] = v
        return history

    def get_results(self, loaded_best: bool) -> Dict[str, float]:
        idx: int
        if loaded_best:
            if self.lock_to_metric:
                idx = int(
                    np.nanargmin(self.metric_vals[self.main_metric_idx])
                    if self.metric_cbs[self.main_metric_idx].lower_metric_better
                    else np.nanargmax(self.metric_vals[self.main_metric_idx])
                )
            else:
                idx = int(np.nanargmin(self.loss_vals["Validation"]))
        else:
            idx = -1

        results = {}
        results["loss"] = self.loss_vals["Validation"][idx]
        if len(self.metric_cbs) > 0:
            for c, v in zip(self.metric_cbs, np.array(self.metric_vals)[:, idx]):
                results[c.name] = v
        return results


class VoxelMetricLogger(MetricLogger):
    def set_wrapper(self, wrapper: AbsVolumeWrapper) -> None:
        super().set_wrapper(wrapper)
        self.dets: List[VoxelDetectorLayer] = []
        for l in self.wrapper.get_detectors():
            if not isinstance(l, VoxelDetectorLayer):
                raise ValueError(f"Detector {l} is not a VoxelDetectorLayer")
            self.dets.append(l)

    def _build_grid_spec(self) -> GridSpec:
        self.n_dets = len(self.dets)
        return self.fig.add_gridspec(5 + (self.main_metric_idx is None), self.n_dets)

    def _set_axes_labels(self) -> None:
        for i in range(self.n_dets):
            self.eff_axes[i].set_xlabel(f"Det. {i}", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
        self.res_axes[0].set_ylabel("Resolution", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
        self.eff_axes[0].set_ylabel("Efficiency", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

    def _prep_plots(self) -> None:
        super()._prep_plots()
        if self.show_plots:
            with sns.axes_style(**self.style):
                self.res_axes = [self.fig.add_subplot(self.grid_spec[-2:-1, i : i + 1]) for i in range(self.n_dets)]
                self.eff_axes = [self.fig.add_subplot(self.grid_spec[-1:, i : i + 1]) for i in range(self.n_dets)]
                self.res_cbar_ax = self.fig.add_axes([1.0, 0.04, 0.03, 0.31])
                self.eff_cbar_ax = self.fig.add_axes([1.1, 0.04, 0.03, 0.31])

    def update_plot(self) -> None:
        super().update_plot()
        with sns.axes_style(**self.style):
            res = np.array([l.resolution.data.cpu().numpy() for l in self.dets])
            eff = np.array([l.efficiency.data.cpu().numpy() for l in self.dets])
            res_min, res_max = res.min(), res.max()
            eff_min, eff_max = eff.min(), eff.max()

            for i, l in enumerate(self.dets):
                self.res_axes[i].clear()
                self.eff_axes[i].clear()
                sns.heatmap(
                    res[i],
                    ax=self.res_axes[i],
                    cmap="viridis",
                    square=True,
                    cbar=(i == 0),
                    vmin=res_min,
                    vmax=res_max,
                    cbar_ax=self.res_cbar_ax if i == 0 else None,
                )
                sns.heatmap(
                    eff[i],
                    ax=self.eff_axes[i],
                    cmap="plasma",
                    square=True,
                    cbar=(i == 0),
                    vmin=eff_min,
                    vmax=eff_max,
                    cbar_ax=self.eff_cbar_ax if i == 0 else None,
                )
            self._set_axes_labels()


class PanelMetricLogger(MetricLogger):
    def _build_grid_spec(self) -> GridSpec:
        self.n_dets = len(self.wrapper.get_detectors())
        return self.fig.add_gridspec(5 + (self.main_metric_idx is None), 3)

    def _set_axes_labels(self) -> None:
        for ax, x in zip(self.below_det, ["x", "y", "x"]):
            ax.set_xlabel(x, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
        for i, (ax, x) in enumerate(zip(self.above_det, ["z", "z", "y"])):
            if i == 0:
                x = "Above, " + x
            ax.set_ylabel(x, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
        for i, (ax, x) in enumerate(zip(self.below_det, ["z", "z", "y"])):
            if i == 0:
                x = "Below, " + x
            ax.set_ylabel(x, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

        for ax, det in zip((self.above_det, self.below_det), self.wrapper.get_detectors()):
            lw, z = det.lw.detach().cpu(), det.z.detach().cpu()
            ax[0].set_xlim(-lw[0] / 2, 1.5 * lw[0])
            ax[1].set_xlim(-lw[1] / 2, 1.5 * lw[1])
            ax[2].set_xlim(-lw[0] / 2, 1.5 * lw[0])
            ax[0].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[1].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[2].set_ylim(-lw[1] / 2, 1.5 * lw[1])
            ax[2].set_aspect("equal", "box")

    def _prep_plots(self) -> None:
        super()._prep_plots()
        if self.show_plots:
            with sns.axes_style(**self.style):
                self.above_det = [self.fig.add_subplot(self.grid_spec[-2:-1, i : i + 1]) for i in range(3)]
                self.below_det = [self.fig.add_subplot(self.grid_spec[-1:, i : i + 1]) for i in range(3)]
                self._set_axes_labels()

    def update_plot(self) -> None:
        super().update_plot()
        with sns.axes_style(**self.style), sns.color_palette(self.cat_palette) as palette:
            for axes, det in zip([self.above_det, self.below_det], self.wrapper.get_detectors()):
                l, s = [], []
                if not isinstance(det, PanelDetectorLayer):
                    raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
                for p in det.panels:
                    if det.type_label == "heatmap":
                        l_val = np.concatenate((p.mu.detach().cpu().numpy().mean(axis=0), p.z.detach().cpu().numpy()))
                        s_val = p.sig.detach().cpu().numpy().mean(axis=0)
                        l.append(l_val)
                        s.append(s_val)
                    else:
                        l.append(np.concatenate((p.xy.detach().cpu().numpy(), p.z.detach().cpu().numpy())))
                        s.append(p.xy_span.detach().cpu().numpy())
                loc, span = np.array(l), np.array(s)

                for ax in axes:
                    ax.clear()
                for p in range(len(loc)):
                    axes[0].add_line(
                        mlines.Line2D((loc[p, 0] - (span[p, 0] / 2), loc[p, 0] + (span[p, 0] / 2)), (loc[p, 2], loc[p, 2]), linewidth=2, color=palette[p])
                    )  # xz
                    axes[1].add_line(
                        mlines.Line2D((loc[p, 1] - (span[p, 1] / 2), loc[p, 1] + (span[p, 1] / 2)), (loc[p, 2], loc[p, 2]), linewidth=2, color=palette[p])
                    )  # yz
                    axes[2].add_patch(
                        patches.Rectangle(
                            (loc[p, 0] - (span[p, 0] / 2), loc[p, 1] - (span[p, 1] / 2)),
                            span[p, 0],
                            span[p, 1],
                            linewidth=1,
                            edgecolor=palette[p],
                            facecolor="none",
                        )
                    )  # xy

            self._set_axes_labels()
