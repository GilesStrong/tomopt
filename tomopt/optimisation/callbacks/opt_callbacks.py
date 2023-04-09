from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union
from abc import ABCMeta, abstractmethod
import numpy as np

from .warmup_callbacks import PostWarmupCallback

if TYPE_CHECKING:
    from ...optimisation.wrapper import AbsVolumeWrapper

__all__ = ["OneCycle"]


class AbsOptSchedule(PostWarmupCallback, metaclass=ABCMeta):
    def __init__(
        self,
        lr_range: Optional[Union[Tuple[float, float], Tuple[float, float, float]]],
        mom_range: Optional[Union[Tuple[float, float], Tuple[float, float, float]]],
        opt_name: str,
    ) -> None:
        self.lr_range = lr_range
        self.mom_range = mom_range
        self.opt_name = opt_name

    def set_wrapper(self, wrapper: AbsVolumeWrapper) -> None:
        super().set_wrapper(wrapper)
        if self.opt_name not in self.wrapper.opts:
            raise ValueError(f"{self.opt_name} not found in VolumeWrapper")

    def _activate(self) -> None:
        super()._activate()
        self.n_iters_per_epoch = len(self.wrapper.fit_params.trn_passives) // self.wrapper.fit_params.passive_bs
        self.n_epochs_expected = self.wrapper.fit_params.n_epochs - self.wrapper.fit_params.epoch + 1

    @abstractmethod
    def schedule(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Compute LR and momentum as a function of inter_cnt, according to defined ranges.
        """

        pass

    def on_train_begin(self) -> None:
        super().on_train_begin()
        if self.lr_range is not None:
            self.wrapper.set_opt_lr(self.lr_range[0], self.opt_name)
        if self.mom_range is not None:
            self.wrapper.set_opt_mom(self.mom_range[0], self.opt_name)
        self.iter_cnt = 0

    def on_step_end(self) -> None:
        if self.active:
            self.iter_cnt += 1
            lr, mom = self.schedule()
            if self.lr_range is not None:
                self.wrapper.set_opt_lr(lr, self.opt_name)
            if self.mom_range is not None:
                self.wrapper.set_opt_mom(mom, self.opt_name)


class OneCycle(AbsOptSchedule):
    def __init__(
        self,
        warmup_length: int,
        lr_range: Optional[Union[Tuple[float, float], Tuple[float, float, float]]],
        mom_range: Optional[Union[Tuple[float, float], Tuple[float, float, float]]],
        opt_name: str,
    ) -> None:
        self.warmup_length = warmup_length
        self.lr_range = lr_range
        self.mom_range = mom_range
        self.opt_name = opt_name

    def on_epoch_end(self) -> None:
        if self.wrapper.fit_params.epoch - 1 == self.warmup_length:
            self.iter_cnt = 0
            self.warming_up = False
            self.length = self.n_epochs_expected - self.wrapper.fit_params.epoch + 1
            self.scale = self.length * self.n_iters_per_epoch
            print("scale", self.scale)
            print(self.wrapper.fit_params.epoch, self.warmup_length, "reset")

    def _activate(self) -> None:
        super()._activate()
        self.warming_up = True
        self.length = self.warmup_length
        self.scale = self.length * self.n_iters_per_epoch
        print("scale", self.scale)

    def schedule(self) -> Tuple[Optional[float], Optional[float]]:
        x = np.cos(np.pi * (self.iter_cnt) / self.scale) + 1

        def get_param(param_range: Union[Tuple[float, float], Tuple[float, float, float]]) -> float:
            if self.warming_up:
                params = (param_range[0], param_range[1])
            else:
                if len(param_range) == 3:
                    params = (param_range[1], param_range[0])
                else:
                    params = (param_range[1], param_range[0])
            dx = (params[1] - params[0]) * x / 2
            return params[1] - dx

        if self.lr_range is None:
            lr = None
        else:
            lr = get_param(self.lr_range)
        if self.mom_range is None:
            mom = None
        else:
            mom = get_param(self.mom_range)
        return lr, mom
