from ...volume import PanelDetectorLayer
from .callback import Callback

import torch

r"""
Provides callbacks for affecting optimisation gradients
"""

__all__ = ["NoMoreNaNs"]


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
            if isinstance(l, PanelDetectorLayer):
                for p in l.panels:
                    if l.type_label == "heatmap":
                        torch.nan_to_num_(p.mu.grad, 0)
                        torch.nan_to_num_(p.norm.grad, 0)
                        torch.nan_to_num_(p.sig.grad, 0)
                        torch.nan_to_num_(p.z.grad, 0)
                    else:
                        torch.nan_to_num_(p.xy.grad, 0)
                        torch.nan_to_num_(p.z.grad, 0)
                        torch.nan_to_num_(p.xy_span.grad, 0)
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l)}")
