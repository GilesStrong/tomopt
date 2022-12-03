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
    def __init__(self, smooth_range: Tuple[float, float]):
        self.smooth_range = smooth_range

    def _activate(self) -> None:
        super()._activate()
        self.offset = self.wrapper.fit_params.epoch - 1
        self.smooth = torch.logspace(np.log10(self.smooth_range[0]), np.log10(self.smooth_range[1]), self.wrapper.fit_params.n_epochs - self.offset)

    def on_train_begin(self) -> None:
        super().on_train_begin()
        self._set_smooth(Tensor([self.smooth_range[0]]))

    def _set_smooth(self, smooth: Tensor) -> None:
        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, PanelDetectorLayer):
                for p in det.panels:
                    if isinstance(p, SigmoidDetectorPanel):
                        p.smooth = smooth

    def on_epoch_begin(self) -> None:
        super().on_epoch_begin()
        if self.active:
            if self.wrapper.fit_params.state == "train":
                self._set_smooth(self.smooth[self.wrapper.fit_params.epoch - self.offset - 1])
