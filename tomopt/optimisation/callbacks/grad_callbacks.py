from .callback import Callback

import torch

__all__ = ["NoMoreNaNs"]


class NoMoreNaNs(Callback):
    def on_backwards_end(self) -> None:
        for l in self.wrapper.volume.get_detectors():
            torch.nan_to_num_(l.resolution.grad, 0)
            torch.nan_to_num_(l.efficiency.grad, 0)
