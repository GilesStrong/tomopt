from __future__ import annotations
from typing import TYPE_CHECKING

from .callback import Callback

if TYPE_CHECKING:
    from ...optimisation import AbsVolumeWrapper

__all__ = []


class AbsOptSchedule(Callback):
    def __init__(self, opt_name: str) -> None:
        self.opt_name = opt_name

    def set_wrapper(self, wrapper: AbsVolumeWrapper) -> None:
        super().set_wrapper(wrapper)
        if self.opt_name not in self.wrapper.opts:
            raise ValueError(f"{self.opt_name} not found in VolumeWrapper")
