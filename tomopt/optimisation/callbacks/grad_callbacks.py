from ...volume import VoxelDetectorLayer, PanelDetectorLayer
from .callback import Callback

import torch

__all__ = ["NoMoreNaNs"]


class NoMoreNaNs(Callback):
    def on_backwards_end(self) -> None:
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, VoxelDetectorLayer):
                torch.nan_to_num_(l.resolution.grad, 0)
                torch.nan_to_num_(l.efficiency.grad, 0)
            elif isinstance(l, PanelDetectorLayer):
                for p in l.panels:
                    torch.nan_to_num_(p.xy.grad, 0)
                    torch.nan_to_num_(p.z.grad, 0)
                    torch.nan_to_num_(p.xy_span.grad, 0)
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l)}")
