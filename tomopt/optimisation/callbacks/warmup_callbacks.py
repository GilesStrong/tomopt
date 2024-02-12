from typing import Dict, List

import numpy as np
from prettytable import PrettyTable

from .callback import Callback

r"""
Provides callbacks that act at the start of training to freeze the optimisation and adjust themselves to the initial state of the detectors
"""

__all__ = ["WarmupCallback", "CostCoefWarmup", "OptConfig", "PostWarmupCallback"]


class WarmupCallback(Callback):
    r"""
    Warmup callbacks act at the start of training to track and set parameters based on the initial state of the detector.
    During warmup, optimisation of the detector is prevented, via a flag.
    If multiple warmup callbacks are present, they will wait to warmup according to the order they are provided in.
    Once the last warmup callback finished, the flag will be set to allow the detectors to be optimised.
    When a `WarmupCallback` is warming up, its `warmup_active` attribute will be True.

    .. important::
        When inheriting from `WarmupCallback`, the super methods of `on_train_begin`, `on_epoch_begin`, and `on_epoch_end` must be called.

    Arguments:
        n_warmup: number of training epochs over-which to warmup
    """

    def __init__(self, n_warmup: int):
        self.n_warmup = n_warmup

    def on_train_begin(self) -> None:
        r"""
        Prepares to warmup
        """

        super().on_train_begin()
        self._reset()

    def check_warmups(self) -> None:
        r"""
        If a `WarmupCallback` has finished, then its `warmup_active` is set to False,
        and the next `WarmupCallback` will have its `warmup_active` is set to True.
        If the finishing callback was the last `WarmupCallback`, then the "skip optimisation" flag is unset.
        """

        for i, c in enumerate(self.wrapper.fit_params.warmup_cbs):
            if c.warmup_active and c.epoch_cnt >= c.n_warmup:
                c.warmup_active = False
                if i < len(self.wrapper.fit_params.warmup_cbs) - 1:
                    self.wrapper.fit_params.warmup_cbs[i + 1].warmup_active = True
                    break
                else:
                    self.wrapper.fit_params.skip_opt_step = False

    def on_epoch_begin(self) -> None:
        r"""
        Ensures that when one `WarmupCallback` has finished, either the next is called, or the detectors are set to be optimised.
        """

        if self.wrapper.fit_params.state == "train":
            self.check_warmups()

    def on_epoch_end(self) -> None:
        r"""
        After a training epoch is finished, increments the number of epochs that the callback has been warming up, provided it is active.
        """

        if not self.warmup_active:
            return
        if self.wrapper.fit_params.state == "train":
            self.epoch_cnt += 1

    def _reset(self) -> None:
        r"""
        Prepares the callback to warmup, and ensures that only the first `WarmupCallback` is active.
        """

        self.epoch_cnt = 0
        self.warmup_active = False
        self.wrapper.fit_params.skip_opt_step = True
        self.wrapper.fit_params.warmup_cbs[0].warmup_active = True


class CostCoefWarmup(WarmupCallback):
    r"""
    Sets a more stable cost coefficient in the :class:`~tomopt.optimisation.loss.loss.AbsDetectorLoss`
    by averaging the inference-error component for several epochs.
    During this warm-up monitoring phase, the detectors will be kept fixed.

    Arguments:
            n_warmup: number of training epochs to wait before setting the cost coefficient
    """

    def _reset(self) -> None:
        super()._reset()
        self.errors: List[np.ndarray] = []

    def on_volume_end(self) -> None:
        r"""
        If training, grabs the inference-error for the latest volume
        """

        if not self.warmup_active:
            return
        if self.wrapper.fit_params.state == "train" and self.wrapper.fit_params.pred is not None:
            self.errors.append(self.wrapper.loss_func.sub_losses["error"].detach().cpu().numpy())

    def on_epoch_end(self) -> None:
        r"""
        If enough epochs have past, the overall median inference-error is computed and used to set the cost coefficient in the loss.
        """

        if not self.warmup_active:
            return
        super().on_epoch_end()
        if self.wrapper.fit_params.state == "train":
            if self.epoch_cnt >= self.n_warmup:
                avg = np.median(self.errors)
                print(f"{type(self).__name__}: Warmed up, average error = {avg}")
                self.wrapper.loss_func.cost_coef = avg


class OptConfig(WarmupCallback):
    r"""
    Allows the user to specify the desired update steps for parameters in physical units.
    Over the course of several warm-up epochs the gradients on the parameters are monitored, after which suitable learning rates for the optimisers are set,
    such that the parameters will move by the desired amount every update.
    During the warm-up, the detectors will not be updated as optimiser learning rates will be set to zero.

    The calculation here does not account for the effect of the optimiser's momentum, nor scheduling and adaption of learning rates, and so the actual update rates may be different from the desired ones.

    Arguments:
        n_warmup: number of training epochs to wait before setting learning rates
        rates: dictionary of desired update rates for the parameters
            The keys are the names of the optimisers specified in the optimiser dictionary of the wrapper.
            The values are the desired update rates for the parameters in physical units.
            For example, if the optimiser is SGD, and the parameter is the xy position of a panel, then the update rate should be in metres.
            The parameters that are being optimisered are expected to be found in the zeroth parameter group of the optimiser, i.e. `wrapper.opts[opt].param_groups[0]['params']`.
            This implies that the optimiser is expected to have only one parameter group.

    Example::
        >>> OptConfig(n_warmup=2, rates={'xy_pos_opt':xy_pos_rate, 'z_pos_opt':z_pos_rate, 'xy_span_opt':xy_span_rate})
    """

    def __init__(
        self,
        n_warmup: int,
        rates: Dict[str, float],
    ):
        super().__init__(n_warmup=n_warmup)
        self.rates = rates

    def on_backwards_end(self) -> None:
        r"""
        Grabs training gradients from detector parameters
        """

        if not self.warmup_active:
            return
        if self.wrapper.fit_params.state == "train":
            for opt in self.rates:
                for param in self.wrapper.opts[opt].param_groups[0]["params"]:
                    self.stats[opt].append(param.grad.abs().cpu().numpy())

    def on_epoch_end(self) -> None:
        r"""
        When enough training epochs have passed, sets suitable learning rates for the optimisers based on the median gradients and desired update rates
        """

        if not self.warmup_active:
            return
        super().on_epoch_end()
        if self.wrapper.fit_params.state == "train":
            if self.epoch_cnt >= self.n_warmup:
                self.lrs = {}
                print(f"{type(self).__name__}: Optimiser warm-up completed")
                pt = PrettyTable(["Param", "Median Grad", "LR"])
                for k, v in self.stats.items():  # Allow optimisation
                    avg = np.nanmedian(v)
                    lr = self.rates[k] / avg
                    pt.add_row([k, avg, lr])
                    self.wrapper.set_opt_lr(lr, k)
                    self.lrs[k] = lr
                print(pt)

    def _reset(self) -> None:
        super()._reset()
        self.stats: Dict[str, List[np.ndarray]] = {k: [] for k in self.rates}


class PostWarmupCallback(Callback):
    r"""
    Callback class that waits for all :class:`~tomopt.optimisation.callbacks.warmup_callbacks.WarmupCallback` obejcts to finish their warmups before activating.
    """

    def on_train_begin(self) -> None:
        r"""
        Prepares for new training
        """

        super().on_train_begin()
        self.active = False

    def check_warmups(self) -> None:
        r"""
        When all WarmupCallbacks have finished, sets the callback to be active.
        """

        if self.active:
            return
        if len(self.wrapper.fit_params.warmup_cbs) == 0 or np.all([c.warmup_active is False for c in self.wrapper.fit_params.warmup_cbs]):
            self._activate()

    def _activate(self) -> None:
        self.active = True

    def on_epoch_begin(self) -> None:
        r"""
        Checks to see whether the callback should be active.
        """

        if self.wrapper.fit_params.state == "train":
            self.check_warmups()
