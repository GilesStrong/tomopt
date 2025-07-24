from typing import List, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec
from torch import Tensor

from tomopt.optimisation.callbacks import Callback, MetricLogger, PostWarmupCallback
from tomopt.volume import AbsDetectorLayer

from .layer import SegmentedPanelDetectorLayer
from .panel import SegmentedSigmoidDetectorPanel

"""_DESCRIPTION_

Defines customized callbacks inheriting from the tomopt callbacks.
The callbacks are compatible with the SegmentedPanelDetectorLayer
and the SegmentedSigmoidDetectorPanel classes.
"""

__all__ = ["PanelMetricLogger", "HitRecordEpoch", "PredHitRecord", "PredPocaRecord", "LossGradientRecorder", "NoMoreNaNs", "SigmoidPanelSmoothnessSchedule"]


class PanelMetricLogger(MetricLogger):
    r"""
    Logger for use with :class:`~TomOpt_SegmentedPanels.panel.SegmentedSigmoidDetectorPanel` s

    Arguments:
        gif_filename: optional savename for recording a gif of the optimisation process (None -> no gif)
            The savename will be appended to the callback savepath
        gif_length: If saving gifs, controls the total length in seconds
        show_plots: whether to provide live plots during optimisation in notebooks
    """

    def _reset(self) -> None:
        det = self.wrapper.volume.get_detectors()[0]
        if isinstance(det, SegmentedPanelDetectorLayer):
            self.uses_sigmoid_panels = isinstance(det.panels[0], SegmentedSigmoidDetectorPanel)
        else:
            self.uses_sigmoid_panels = False
        super()._reset()

    def _prep_plots(self) -> None:
        r"""
        Creates the plots for a new optimisation
        """

        super()._prep_plots()
        if self.show_plots:
            with sns.axes_style(**self.style):
                self.above_det = [self.fig.add_subplot(self.grid_spec[-2:-1, i : i + 1]) for i in range(3)]
                self.below_det = [self.fig.add_subplot(self.grid_spec[-1:, i : i + 1]) for i in range(3)]
                if self.uses_sigmoid_panels:
                    self.panel_smoothness = self.fig.add_subplot(self.grid_spec[-2:-1, -1:])
                self._set_axes_labels()

    def update_plot(self) -> None:
        r"""
        Updates the plot(s).
        """

        # super().update_plot()

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
            # self.loss_ax.set_yscale("log")
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

        with sns.axes_style(**self.style), sns.color_palette(self.cat_palette) as palette:
            for axes, det in zip([self.above_det, self.below_det], self.wrapper.get_detectors()):
                l, s, g = [], [], []
                if not isinstance(det, AbsDetectorLayer):
                    raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
                for p in det.panels:
                    if det.type_label == "heatmap":
                        l_val = np.concatenate((p.mu.detach().cpu().numpy().mean(axis=0), p.z.detach().cpu().numpy()))
                        s_val = p.sig.detach().cpu().numpy().mean(axis=0)
                        l.append(l_val)
                        s.append(s_val)
                    else:
                        l.append(np.concatenate((p.xy.detach().cpu().numpy(), p.z.detach().cpu().numpy())))
                        s.append(p.get_scaled_xy_span().detach().cpu().numpy())
                        g.append(p.gap_size.detach().cpu().numpy())
                loc, span = np.array(l), np.array(s)

                for ax in axes:
                    ax.clear()

                lw = self.wrapper.volume.lw.detach().cpu().numpy()
                axes[2].add_patch(patches.Rectangle((0, 0), lw[0], lw[1], linewidth=1, edgecolor="black", facecolor="none", hatch="x"))  # volume
                # zs = [p.z.detach().cpu().numpy() for p in det.panels]
                # z_margin = 0.01 # margin for panel z positions
                # axes[0].set_ylim(min(zs) - z_margin, max(zs) + z_margin)
                # axes[1].set_ylim(min(zs) - z_margin, max(zs) + z_margin)

                for p in range(len(loc)):
                    panel = det.panels[p]
                    n_panels = panel.n_panels
                    panel_size = span[p] / n_panels
                    gap_size = panel.gap_size.cpu().detach().numpy()
                    center_x = loc[p][0]
                    center_y = loc[p][1]
                    z = panel.z.cpu().detach().numpy()

                    indices = np.arange(n_panels) - (n_panels - 1) / 2
                    centers_x = center_x + indices * (panel_size[0] + gap_size)
                    centers_y = center_y + indices * (panel_size[0] + gap_size)

                    starts_x = []
                    starts_y = []
                    ends_x = []
                    ends_y = []

                    for cx in centers_x:
                        start = cx - panel_size[0] / 2
                        end = cx + panel_size[0] / 2
                        starts_x.append(start)
                        ends_x.append(end)
                        axes[0].add_line(mlines.Line2D((start, end), (z, z), linewidth=2, color=palette[p]))  # xz

                    for cy in centers_y:
                        start = cy - panel_size[0] / 2
                        end = cy + panel_size[0] / 2
                        starts_y.append(start)
                        ends_y.append(end)
                        axes[1].add_line(mlines.Line2D((start, end), (z, z), linewidth=2, color=palette[p]))  # yz

                    for i in range(n_panels):
                        for j in range(n_panels):
                            axes[2].add_patch(
                                patches.Rectangle(
                                    (starts_x[i], starts_y[j]),
                                    ends_x[i] - starts_x[i],
                                    ends_y[j] - starts_y[j],
                                    linewidth=1,
                                    edgecolor=palette[p],
                                    facecolor="none",
                                )
                            )  # xy

                if self.uses_sigmoid_panels:
                    self.panel_smoothness.clear()
                    with torch.no_grad():
                        panel = det.panels[0]
                        width = panel.get_scaled_xy_span()[0].cpu().item()
                        panel.xy[0].cpu().item()
                        x_range = torch.linspace(-width, width, 50)[:, None]
                        y = panel.sig_model(x)[:, 0]
                        self.panel_smoothness.plot(2 * x_range / width, y)

            self._set_axes_labels()

    def _build_grid_spec(self) -> GridSpec:
        r"""
        Returns:
            The layout object for the plots
        """

        self.n_dets = len(self.wrapper.get_detectors())
        return self.fig.add_gridspec(5 + (self.main_metric_idx is None), 3 + self.uses_sigmoid_panels)

    def _set_axes_labels(self) -> None:
        r"""
        Adds styling to plots after they are cleared
        """

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
            if not isinstance(det, SegmentedPanelDetectorLayer):
                raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
            lw, z = det.lw.detach().cpu(), det.z.detach().cpu()
            sizes = torch.stack([p.get_scaled_xy_span().detach().cpu() for p in det.panels], dim=0) / 2
            poss = torch.stack([p.xy.detach().cpu() for p in det.panels], dim=0)
            xy_min, xy_max = (poss - sizes).min(0).values, (poss + sizes).max(0).values
            margin = lw.max() / 2

            ax[0].set_xlim(min([1, xy_min[0].item()]) - (lw[0] / 2), max([lw[0].item(), xy_max[0].item()]) + (lw[0] / 2))
            ax[1].set_xlim(min([1, xy_min[1].item()]) - (lw[1] / 2), max([lw[1].item(), xy_max[1].item()]) + (lw[1] / 2))
            ax[2].set_xlim(xy_min.min() - margin, xy_max.max() + margin)
            ax[0].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[1].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[2].set_ylim(xy_min.min() - margin, xy_max.max() + margin)
            ax[2].set_aspect("equal", "box")

        if self.uses_sigmoid_panels:
            self.panel_smoothness.set_xlim((-2, 2))
            self.panel_smoothness.set_xlabel("Panel model (arb. pos.)", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)


class HitRecordEpoch(Callback):
    r"""
    Extended hit recorder. Used in training mode.
    Records reconstructed hits, generated hits, and hit uncertainties,
    separated into above and below panels, for each epoch.
    Also distinguishes between training and validation data.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize lists to accumulate hits during the epoch
        self.epoch_reco_hits_train: List[Tensor] = []
        self.epoch_reco_hits_valid: List[Tensor] = []
        self.epoch_gen_hits_train: List[Tensor] = []
        self.epoch_gen_hits_valid: List[Tensor] = []
        self.epoch_hit_uncs_train: List[Tensor] = []
        self.epoch_hit_uncs_valid: List[Tensor] = []

    def on_volume_begin(self) -> None:
        """
        Initialize data structures for the epoch.
        This method is called at the start of each epoch.
        """
        # Clear the temporary lists for the epoch
        self.epoch_reco_hits: List[Tensor] = []
        self.epoch_gen_hits: List[Tensor] = []
        self.epoch_hit_uncs: List[Tensor] = []

    def on_x0_pred_end(self) -> None:
        """
        Save the hits, generated hits, and uncertainties for the entire epoch.
        This method is called at the end of each epoch.
        """
        reco_hits_epoch = torch.cat(self.epoch_reco_hits, dim=0)
        gen_hits_epoch = torch.cat(self.epoch_gen_hits, dim=0)
        hit_uncs_epoch = torch.cat(self.epoch_hit_uncs, dim=0)

        # Check if the current state is "train" or "valid"
        if self.wrapper.fit_params.state == "train":
            # Convert accumulated hits for the training epoch into tensors
            self.epoch_reco_hits_train.append(reco_hits_epoch)
            self.epoch_gen_hits_train.append(gen_hits_epoch)
            self.epoch_hit_uncs_train.append(hit_uncs_epoch)

            print(f"Training epoch ends with {len( self.epoch_reco_hits_train)} reco hits, {len(self.epoch_gen_hits_train)} gen hits")

        elif self.wrapper.fit_params.state == "valid":
            # Convert accumulated hits for the validation epoch into tensors
            self.epoch_reco_hits_valid.append(reco_hits_epoch)
            self.epoch_gen_hits_valid.append(gen_hits_epoch)
            self.epoch_hit_uncs_valid.append(hit_uncs_epoch)

            print(f"Validation epoch ends with {len(self.epoch_reco_hits_valid)} reco hits, {len(self.epoch_gen_hits_valid)} gen hits")

    def on_scatter_end(self) -> None:
        """
        Saves the hits, generated hits, and uncertainties of the latest muon batch.
        This method is called at the end of each scatter event.
        """
        self.n_above = self.wrapper.fit_params.sb.n_hits_above

        # Record reconstructed hits
        reco_hits = self.wrapper.fit_params.sb._reco_hits.detach().cpu().clone()
        self.epoch_reco_hits.append(reco_hits)

        # Record generated hits
        gen_hits = self.wrapper.fit_params.sb._gen_hits.detach().cpu().clone()
        self.epoch_gen_hits.append(gen_hits)

        # Record hit uncertainties
        hit_uncs = self.wrapper.fit_params.sb._hit_uncs.detach().cpu().clone()
        self.epoch_hit_uncs.append(hit_uncs)

    def split_above_below(self, record: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Splits the record into above and below hits.

        Arguments:
            record: (n_muons, n_panels (above + below), xyz)

        Returns:
            Tuple of (above_hits, below_hits)
        """
        above_hits = record[:, : self.n_above]
        below_hits = record[:, self.n_above :]
        return above_hits, below_hits


class PredHitRecord(Callback):
    r"""
    Extended hit recorder. Used in prediction mode.
    Records reconstructed hits, generated hits, and hit uncertainties,
    separated into above and below panels.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize lists to accumulate hits for the volumes in the
        # passive batch.
        self.reco_hits_batch: List[Tensor] = []
        self.gen_hits_batch: List[Tensor] = []
        self.hit_uncs_batch: List[Tensor] = []

    def on_volume_begin(self) -> None:
        """
        Initializes the lists for the new volume.
        This method is called at the beginning of each volume prediction.
        """
        self.reco_hits: List[Tensor] = []
        self.gen_hits: List[Tensor] = []
        self.hit_uncs: List[Tensor] = []

    def on_scatter_end(self) -> None:
        """
        Saves the hits, generated hits, and uncertainties of the latest muon batch.
        This method is called at the end of each scatter event.
        """
        self.n_above = self.wrapper.fit_params.sb.n_hits_above

        # Record reconstructed hits
        reco_hits = self.wrapper.fit_params.sb._reco_hits.detach().cpu().clone()
        self.reco_hits.append(reco_hits)

        # Record generated hits
        gen_hits = self.wrapper.fit_params.sb._gen_hits.detach().cpu().clone()
        self.gen_hits.append(gen_hits)

        # Record hit uncertainties
        hit_uncs = self.wrapper.fit_params.sb._hit_uncs.detach().cpu().clone()
        self.hit_uncs.append(hit_uncs)

    def on_x0_pred_end(self) -> None:
        """
        Called at the end of the volume prediction.
        This method is called after all muon batches have been processed.
        """
        # Concatenate hits from all batches
        reco_hits = torch.cat(self.reco_hits, dim=0)
        gen_hits = torch.cat(self.gen_hits, dim=0)
        hit_uncs = torch.cat(self.hit_uncs, dim=0)

        self.reco_hits_batch.append(reco_hits)
        self.gen_hits_batch.append(gen_hits)
        self.hit_uncs_batch.append(hit_uncs)

    def split_above_below(self, record: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Splits the record into above and below hits.

        Arguments:
            record: (n_muons, n_panels (above + below), xyz)

        Returns:
            Tuple of (above_hits, below_hits)
        """
        above_hits = record[:, : self.n_above]
        below_hits = record[:, self.n_above :]
        return above_hits, below_hits


class PredPocaRecord(Callback):
    r"""
    Scattering poca point recorder. Used in prediction mode.
    Records poca xyz positions and uncertainties, and their scattering
    angles and their uncertainties.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize lists to accumulate hits for the volumes in the
        # passive batch.
        self.poca_xyz_batch: List[Tensor] = []
        self.poca_xyz_unc_batch: List[Tensor] = []
        self.poca_theta_mcs_batch: List[Tensor] = []
        self.poca_theta_mcs_unc_batch: List[Tensor] = []

    def on_volume_begin(self) -> None:
        """
        Initializes the lists for the new volume.
        This method is called at the beginning of each volume prediction.
        """
        self.poca_xyz: List[Tensor] = []
        self.poca_xyz_unc: List[Tensor] = []
        self.poca_theta_mcs: List[Tensor] = []
        self.poca_theta_mcs_unc: List[Tensor] = []

    def on_scatter_end(self) -> None:
        """
        Saves the hits, generated hits, and uncertainties of the latest muon batch.
        This method is called at the end of each scatter event.
        """
        self.n_above = self.wrapper.fit_params.sb.n_hits_above

        # Record poca xyz positions
        poca_xyz = self.wrapper.fit_params.sb.poca_xyz.detach().cpu().clone()
        self.poca_xyz.append(poca_xyz)

        # Record poca xyz uncertainties
        poca_xyz_unc = self.wrapper.fit_params.sb.poca_xyz_unc.detach().cpu().clone()
        self.poca_xyz_unc.append(poca_xyz_unc)

        # Record scattering angles
        poca_theta_mcs = self.wrapper.fit_params.sb.total_scatter.detach().cpu().clone()
        self.poca_theta_mcs.append(poca_theta_mcs)

        # Record scattering angles uncertainties
        poca_theta_mcs_unc = self.wrapper.fit_params.sb.total_scatter_unc.detach().cpu().clone()
        self.poca_theta_mcs_unc.append(poca_theta_mcs_unc)

    def on_x0_pred_end(self) -> None:
        """
        Called at the end of the volume prediction.
        This method is called after all muon batches have been processed.
        """
        # Concatenate hits from all batches
        poca_xyz = torch.cat(self.poca_xyz, dim=0)
        self.poca_xyz_batch.append(poca_xyz)

        poca_xyz_unc = torch.cat(self.poca_xyz_unc, dim=0)
        self.poca_xyz_unc_batch.append(poca_xyz_unc)

        poca_theta_mcs = torch.cat(self.poca_theta_mcs, dim=0)
        self.poca_theta_mcs_batch.append(poca_theta_mcs)

        poca_theta_mcs_unc = torch.cat(self.poca_theta_mcs_unc, dim=0)
        self.poca_theta_mcs_unc_batch.append(poca_theta_mcs_unc)

    def split_above_below(self, record: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Splits the record into above and below hits.

        Arguments:
            record: (n_muons, n_panels (above + below), xyz)

        Returns:
            Tuple of (above_hits, below_hits)
        """
        above_hits = record[:, : self.n_above]
        below_hits = record[:, self.n_above :]
        return above_hits, below_hits


class LossGradientRecorder(Callback):
    r"""
    Callback for recording the gradients of loss wrt panel xyz positions across epochs,
    as well as the values of these parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss: List[Tensor] = []
        self.zgradients: List[list] = []
        self.xgradients: List[list] = []
        self.ygradients: List[list] = []
        self.gap_gradients: List[list] = []
        self.zvals: List[list] = []
        self.xvals: List[list] = []
        self.yvals: List[list] = []
        self.gap_vals: List[list] = []

    def on_backwards_end(self) -> None:
        zgrads: List[float] = []
        xgrads: List[float] = []
        ygrads: List[float] = []
        gap_grads: List[Tensor] = []
        zvals: List[float] = []
        xvals: List[float] = []
        yvals: List[float] = []
        gap_vals: List[float] = []

        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, SegmentedPanelDetectorLayer):
                gap_grads.append(det.gap_size.grad.detach().cpu().clone().numpy())
                gap_vals.append(det.gap_size.detach().clone().cpu().numpy())
                for p in det.panels:
                    zgrads.append(p.z.grad.detach().cpu().clone().numpy())
                    xgrads.append(p.xy.grad[0].detach().cpu().clone().numpy())
                    ygrads.append(p.xy.grad[1].detach().cpu().clone().numpy())

                    zvals.append(p.z.detach().cpu().clone().numpy())
                    xvals.append(p.xy[0].detach().cpu().clone().numpy())
                    yvals.append(p.xy[1].detach().cpu().clone().numpy())

        self.zgradients.append(zgrads)
        self.xgradients.append(xgrads)
        self.ygradients.append(ygrads)
        self.gap_gradients.append(gap_grads)
        self.zvals.append(zvals)
        self.xvals.append(xvals)
        self.yvals.append(yvals)
        self.gap_vals.append(gap_vals)
        self.loss.append(self.wrapper.fit_params.loss_val.detach().numpy())


class NoMoreNaNs(Callback):
    r"""
    Prior to parameter updates, this callback will check and set any NaN gradients to zero.
    Updates based on NaN gradients will set the parameter value to NaN.

    .. important::
        As new parameters are introduced, e.g. through new detector models, this callback will need to be updated.
    """

    def on_backwards_end(self) -> None:
        r"""
        Prior to optimiser updates, parameter gradients are checked for NaNs.
        """

        if hasattr(self.wrapper.volume, "budget_weights"):
            torch.nan_to_num_(self.wrapper.volume.budget_weights.grad, 0)
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, SegmentedPanelDetectorLayer):
                torch.nan_to_num_(l.gap_size.grad, 0)
                for p in l.panels:
                    torch.nan_to_num_(p.xy.grad, 0)
                    torch.nan_to_num_(p.z.grad, 0)
                    torch.nan_to_num_(p.xy_span.grad, 0)
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l)}")


class SigmoidPanelSmoothnessSchedule(PostWarmupCallback):
    r"""
    Creates an annealing schedule for the smooth attribute of :class:`~TomOpt_SegmentedPanels.layer.SegmentedPanelDetectorLayer`.
    This can be used to move from smooth, unphysical panel with high sensitivity outside the physical panel boundaries,
    to one with sharper decrease in resolution | efficiency at the edge, and so more closely resembles a physical panel, whilst still being differentiable.

    Arguments:
        smooth_range: tuple of initial and final values for the smooth attributes of all panels in the volume.
            A base-10 log schedule used over the number of epochs-total number of warmup epochs.
    """

    def __init__(self, smooth_range: Tuple[float, float]):
        self.smooth_range = smooth_range

    def _activate(self) -> None:
        r"""
        When the schedule begins, computes the appropriate smooth value at each up-coming epoch.
        """

        super()._activate()
        self.offset = self.wrapper.fit_params.epoch - 1
        self.smooth = torch.logspace(np.log10(self.smooth_range[0]), np.log10(self.smooth_range[1]), self.wrapper.fit_params.n_epochs - self.offset)

    def on_train_begin(self) -> None:
        r"""
        Sets all :class:`~tomopt.volume.panel.SigmoidDetectorPanel` s to their initial smooth values.
        """

        super().on_train_begin()
        self._set_smooth(Tensor([self.smooth_range[0]]))

    def _set_smooth(self, smooth: Tensor) -> None:
        r"""
        Sets the smooth values for all :class:`~tomopt.volume.panel.SigmoidDetectorPanel  in the detector.

        Arguments:
            smooth: smooth values for every :class:`~tomopt.volume.panel.SigmoidDetectorPanel` in the volume.
        """

        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, SegmentedPanelDetectorLayer):
                for p in det.panels:
                    if isinstance(p, SegmentedSigmoidDetectorPanel):
                        p.smooth = smooth

    def on_epoch_begin(self) -> None:
        r"""
        At the start of each training epoch, will anneal the :class:`~tomopt.volume.panel.SigmoidDetectorPanel` s' smooth attributes, if the callback is active.
        """

        super().on_epoch_begin()
        if self.active:
            if self.wrapper.fit_params.state == "train":
                self._set_smooth(self.smooth[self.wrapper.fit_params.epoch - self.offset - 1])
