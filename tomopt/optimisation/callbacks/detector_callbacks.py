from typing import Tuple, Dict, List, Optional

import torch
from torch import Tensor

from .callback import Callback
from ...volume import PanelDetectorLayer

r"""
Set of callbacks that affect the detectors during optimisation
"""

__all__ = ["PanelUpdateLimiter"]


class PanelUpdateLimiter(Callback):
    def __init__(
        self, max_xy_step: Optional[Tuple[float, float]] = None, max_z_step: Optional[float] = None, max_xy_span_step: Optional[Tuple[float, float]] = None
    ):
        self.max_xy_step = Tensor(max_xy_step) if max_xy_step is not None else None
        self.max_z_step = Tensor([max_z_step]) if max_z_step is not None else None
        self.max_xy_span_step = Tensor(max_xy_span_step) if max_xy_span_step is not None else None

    def on_backwards_end(self) -> None:
        self.panel_params: List[Dict[str, Tensor]] = []
        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, PanelDetectorLayer):
                for panel in det.panels:
                    self.panel_params.append({"xy": panel.xy.detach().clone(), "z": panel.z.detach().clone(), "xy_span": panel.xy_span.detach().clone()})

    def on_step_end(self) -> None:
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
