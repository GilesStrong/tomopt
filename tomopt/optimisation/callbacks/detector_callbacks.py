from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ...volume import PanelDetectorLayer, SigmoidDetectorPanel
from .callback import Callback
from .warmup_callbacks import PostWarmupCallback

r"""
Set of callbacks that affect the detectors during optimisation
"""

__all__ = ["PanelUpdateLimiter", "SigmoidPanelSmoothnessSchedule"]


class SigmoidPanelSmoothnessSchedule(PostWarmupCallback):
    r"""
    Creates an annealing schedule for the smooth attribute of :class:`~tomopt.volume.panel.SigmoidDetectorPanel`.
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
            if isinstance(det, PanelDetectorLayer):
                for p in det.panels:
                    if isinstance(p, SigmoidDetectorPanel):
                        p.smooth = smooth

    def on_epoch_begin(self) -> None:
        r"""
        At the start of each training epoch, will anneal the :class:`~tomopt.volume.panel.SigmoidDetectorPanel` s' smooth attributes, if the callback is active.
        """

        super().on_epoch_begin()
        if self.active:
            if self.wrapper.fit_params.state == "train":
                self._set_smooth(self.smooth[self.wrapper.fit_params.epoch - self.offset - 1])


class PanelUpdateLimiter(Callback):
    r"""
    Limits the maximum difference that optimisers can make to panel parameters, to prevent them from being affected by large updates from anomolous gradients.
    This is enacted by a hard-clamping based on the initial and final parameter values before/after each update step.

    Arguments:
        max_xy_step: maximum update in xy position of panels
        max_z_step: maximum update in z position of panels
        max_xy_span_step: maximum update in xy_span position of panels
    """

    def __init__(
        self, max_xy_step: Optional[Tuple[float, float]] = None, max_z_step: Optional[float] = None, max_xy_span_step: Optional[Tuple[float, float]] = None
    ):
        self.max_xy_step = Tensor(max_xy_step) if max_xy_step is not None else None
        self.max_z_step = Tensor([max_z_step]) if max_z_step is not None else None
        self.max_xy_span_step = Tensor(max_xy_span_step) if max_xy_span_step is not None else None

    def on_backwards_end(self) -> None:
        r"""
        Records the current paramaters of each panel before they are updated.
        """

        self.panel_params: List[Dict[str, Tensor]] = []
        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, PanelDetectorLayer):
                for panel in det.panels:
                    self.panel_params.append({"xy": panel.xy.detach().clone(), "z": panel.z.detach().clone(), "xy_span": panel.xy_span.detach().clone()})

    def on_step_end(self) -> None:
        r"""
        After the update step, goes through and hard-clamps parameter updates based on the difference between their current values
        and values before the update step.
        """

        with torch.no_grad():
            panel_idx = 0
            for det in self.wrapper.volume.get_detectors():
                if isinstance(det, PanelDetectorLayer):
                    for panel in det.panels:
                        if self.max_xy_step is not None:
                            delta = panel.xy - self.panel_params[panel_idx]["xy"]
                            panel.xy.data = torch.where(
                                delta.abs() > self.max_xy_step, self.panel_params[panel_idx]["xy"] + (torch.sign(delta) * self.max_xy_step), panel.xy
                            )

                        if self.max_z_step is not None:
                            delta = panel.z - self.panel_params[panel_idx]["z"]
                            panel.z.data = torch.where(
                                delta.abs() > self.max_z_step, self.panel_params[panel_idx]["z"] + (torch.sign(delta) * self.max_z_step), panel.z
                            )

                        if self.max_xy_span_step is not None:
                            delta = panel.xy_span - self.panel_params[panel_idx]["xy_span"]
                            panel.xy_span.data = torch.where(
                                delta.abs() > self.max_xy_span_step,
                                self.panel_params[panel_idx]["xy_span"] + (torch.sign(delta) * self.max_xy_span_step),
                                panel.xy_span,
                            )
                        panel_idx += 1


class PanelCentring(Callback):
    """
    Callback class for panel centring in the optimisation process.

    This callback is used to centre the panels of PanelDetectorLayer objects
    by setting their xy coordinates to the mean xy value of all panels in the layer.

    This update takes place after the panel positions have been updated in the optimisation process.
    """

    def on_step_end(self) -> None:
        """
        Updates the xy coordinates of all panels in the PanelDetectorLayer objects after they have be updated, based on their current mean xy position.
        """
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, PanelDetectorLayer):
                xy = []
                for p in l.panels:
                    xy.append(p.xy.detach().cpu().numpy())
                mean_xy = Tensor(np.mean(xy, 0), device=self.wrapper.device)
                for p in l.panels:
                    p.xy.data = mean_xy
            else:
                raise NotImplementedError(f"PanelCentring does not yet support {type(l)}")
