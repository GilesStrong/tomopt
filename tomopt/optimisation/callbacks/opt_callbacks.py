from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .callback import Callback
from .warmup_callbacks import PostWarmupCallback

if TYPE_CHECKING:
    from ...optimisation.wrapper import AbsVolumeWrapper

__all__ = ["OneCycle", "EpochSave"]


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
        if self.opt_name not in self.wrapper.opts:
            raise ValueError(f"{self.opt_name} not found in VolumeWrapper")

    def _activate(self) -> None:
        super()._activate()
        self.n_iters_per_epoch = len(self.wrapper.fit_params.trn_passives) // self.wrapper.fit_params.passive_bs
        self.n_epochs_expected = self.wrapper.fit_params.n_epochs - self.wrapper.fit_params.epoch + 1

    @abstractmethod
    def schedule(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Compute LR and momentum as a function of iter_cnt, according to defined ranges.
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
    r"""
    Callback implementing Smith 1-cycle evolution for lr and momentum (beta_1) https://arxiv.org/abs/1803.09820

    In the warmup phase:
        Learning rate is increased from `init_lr` to `mid_lr`,
        Momentum is decreased from `init_mom` to `mid_mom`, to stabilise the use of high LRs

    In the convergence phase:
        Learning rate is decreased from `mid_lr` to `final_lr`,
        Momentum is increased from `mid_mom` to `final_mom`

    Setting the learning rate or momentum here will override the values specified when instantiating the `VolumeWrapper`.
    learning rate or momentum arguments can be `None` to avoid annealing or overriding their values.

    Arguments:
        opt_name: name of optimiser that should be affected by the scheduler
        warmup_length: number of epochs to use for the warmup phase
        init_lr: initial learning rate (low)
        init_mom: initial momentum (high)
        mid_lr: nominal learning rate (high),
        mid_mom: nominal momentum (moderate),
        final_lr: final learning rate (low),
        final_mom: final momentum (high)
    """

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


class EpochSave(Callback):
    r"""
    Saves the state of the volume at the end of each training epoch to a unique file.
    This can be used to load a specifc state to either be used, or to resume training.
    """

    def on_epoch_end(self) -> None:
        if self.wrapper.fit_params.state == "train":
            self.wrapper.save(self.wrapper.fit_params.cb_savepath / f"volume_state_{self.wrapper.fit_params.epoch}.pt")
