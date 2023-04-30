from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
from abc import ABCMeta, abstractmethod
import numpy as np

from .warmup_callbacks import PostWarmupCallback

if TYPE_CHECKING:
    from ...optimisation.wrapper import AbsVolumeWrapper

__all__ = ["OneCycle"]


class AbsOptSchedule(PostWarmupCallback, metaclass=ABCMeta):
    def __init__(
        self,
        opt_name: str,
        init_lr: Optional[float] = None,
        init_mom: Optional[float] = None,
    ) -> None:
        self.init_lr = init_lr
        self.init_mom = init_mom
        self.opt_name = opt_name

    def set_wrapper(self, wrapper: AbsVolumeWrapper) -> None:
        super().set_wrapper(wrapper)
        print(self.wrapper)
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
        if self.init_lr is not None:
            self.wrapper.set_opt_lr(self.init_lr, self.opt_name)
        if self.init_mom is not None:
            self.wrapper.set_opt_mom(self.init_mom, self.opt_name)
        self.iter_cnt = 0

    def on_step_end(self) -> None:
        if self.active:
            self.iter_cnt += 1
            lr, mom = self.schedule()
            if lr is not None:
                self.wrapper.set_opt_lr(lr, self.opt_name)
            if mom is not None:
                self.wrapper.set_opt_mom(mom, self.opt_name)


class OneCycle(AbsOptSchedule):
    def __init__(
        self,
        opt_name: str,
        warmup_length: int,
        init_lr: Optional[float] = None,
        init_mom: Optional[float] = None,
        mid_lr: Optional[float] = None,
        mid_mom: Optional[float] = None,
        final_lr: Optional[float] = None,
        final_mom: Optional[float] = None,
    ) -> None:
        super().__init__(opt_name=opt_name, init_lr=init_lr, init_mom=init_mom)
        self.warmup_length = warmup_length
        self.mid_lr = mid_lr
        self.mid_mom = mid_mom
        self.final_lr = final_lr if final_lr is not None else init_lr
        self.final_mom = final_mom if final_mom is not None else init_mom

    def on_epoch_end(self) -> None:
        if self.wrapper.fit_params.epoch - 1 == self.warmup_length:
            self.iter_cnt = 0
            self.warming_up = False
            self.length = self.n_epochs_expected - self.wrapper.fit_params.epoch + 1
            self.scale = self.length * self.n_iters_per_epoch
            print("scale", self.scale)

    def _activate(self) -> None:
        super()._activate()
        self.warming_up = True
        self.length = self.warmup_length
        self.scale = self.length * self.n_iters_per_epoch

    def schedule(self) -> Tuple[Optional[float], Optional[float]]:
        x = np.cos(np.pi * (self.iter_cnt) / self.scale) + 1

        def get_param(init: float, mid: float, final: float) -> float:
            if self.warming_up:
                params = (init, mid)
            else:
                params = (final, mid)
            dx = (params[1] - params[0]) * x / 2
            return params[1] - dx

        if self.init_lr is None:
            lr = None
        else:
            lr = get_param(self.init_lr, self.mid_lr, self.final_lr)
        if self.init_mom is None:
            mom = None
        else:
            mom = get_param(self.init_mom, self.mid_mom, self.final_mom)
        return lr, mom
