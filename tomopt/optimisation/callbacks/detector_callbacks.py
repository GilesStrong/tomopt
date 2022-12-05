from typing import Tuple
import numpy as np

import torch
from torch import Tensor

from .warmup_callbacks import PostWarmupCallback
from ...volume import PanelDetectorLayer, SigmoidDetectorPanel

r"""
Set of callbacks that affect the detectors during optimisation
"""

__all__ = ["SigmoidPanelSmoothnessSchedule"]


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
        Sets all :class:`~tomopt.volume.panel.SigmoidDetectorPanel`s to their initial smooth values.
        """

        super().on_train_begin()
        self._set_smooth(Tensor([self.smooth_range[0]]))

    def _set_smooth(self, smooth: Tensor) -> None:
        r"""
        Sets the smooth values for all :class:`~tomopt.volume.panel.SigmoidDetectorPanel`s in the detector.

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
        At the start of each training epoch, will anneal the :class:`~tomopt.volume.panel.SigmoidDetectorPanel`s' smooth attributes, if the callback is active.
        """

        super().on_epoch_begin()
        if self.active:
            if self.wrapper.fit_params.state == "train":
                self._set_smooth(self.smooth[self.wrapper.fit_params.epoch - self.offset - 1])
