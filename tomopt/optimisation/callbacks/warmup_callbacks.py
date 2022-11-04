from prettytable import PrettyTable
import numpy as np
from typing import Dict, List, Optional

from .callback import Callback
from ...volume import PanelDetectorLayer

r"""
Provides callbacks that act at the start of training to freeze the optimisation and adjust themselves to the initial state of the detectors
"""

__all__ = ["WarmupCallback", "CostCoefWarmup", "PanelOptConfig"]


class WarmupCallback(Callback):
    def __init__(self, n_warmup: int):
        self.n_warmup = n_warmup

    def on_train_begin(self) -> None:
        super().on_train_begin()
        self._reset()

    def check_warmups(self) -> None:
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
        Runs when a new training or validations epoch begins.
        """

        if self.wrapper.fit_params.state == "train":
            self.check_warmups()

    def on_epoch_end(self) -> None:
        r"""
        Runs when a training or validations epoch ends.
        """

        if not self.warmup_active:
            return
        if self.wrapper.fit_params.state == "train":
            self.epoch_cnt += 1

    def _reset(self) -> None:
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
        If training, grabs the inference-error for the latest volume and adds to the running sum
        """

        if not self.warmup_active:
            return
        if self.wrapper.fit_params.state == "train" and self.wrapper.fit_params.pred is not None:
            self.errors.append(self.wrapper.loss_func.sub_losses["error"].detach().cpu().numpy())

    def on_epoch_end(self) -> None:
        r"""
        If training, adds the epoch mean inference error to a new running sum of errors.
        If then enough epochs have past, the overall mean inference-error is computed and used to set the cost coefficient in the loss,
        and the learning rates of the optimisers are set back to their original non-zero values.
        """

        if not self.warmup_active:
            return
        super().on_epoch_end()
        if self.wrapper.fit_params.state == "train":
            if self.epoch_cnt >= self.n_warmup:
                avg = np.median(self.errors)
                print(f"{type(self).__name__}: Warmed up, average error = {avg}")
                self.wrapper.loss_func.cost_coef = avg


class PanelOptConfig(WarmupCallback):
    r"""
    Allows the user to specify the desired update steps for :class:`~tomopt.volume.layer.PanelDetectorLayer`s in physical units.
    Over the course of several warm-up epochs the gradients on the parameters are monitored, after which suitable learning rates for the optimisation are set.
    During the warm-up, the detectors will not be updated as optimiser learning rates will be set to zero.

    Arguments:
        n_warmup: number of training epochs to wait before setting learning rates
        xy_pos_rate: desired distance in metres the panels should move in xy every update
        z_pos_rate: desired distance in metres the panels should move in z every update
        xy_span_rate: desired distance in metres the panels should expand in xy every update
        budget_rate: desired rate at which the fractional budget assignments should change every update
    """

    def __init__(
        self,
        n_warmup: int,
        xy_pos_rate: Optional[float] = None,
        z_pos_rate: Optional[float] = None,
        xy_span_rate: Optional[float] = None,
        budget_rate: Optional[float] = None,
    ):
        super().__init__(n_warmup=n_warmup)
        self.rates: Dict[str, float] = {}
        if xy_pos_rate is not None and xy_pos_rate != 0:
            self.rates["xy_pos_opt"] = xy_pos_rate
        if z_pos_rate is not None and z_pos_rate != 0:
            self.rates["z_pos_opt"] = z_pos_rate
        if xy_span_rate is not None and xy_span_rate != 0:
            self.rates["xy_span_opt"] = xy_span_rate
        if budget_rate is not None and budget_rate != 0:
            self.rates["budget_opt"] = budget_rate

    def on_backwards_end(self) -> None:
        r"""
        Grabs training gradients from detector parameters
        """

        if not self.warmup_active:
            return
        if self.wrapper.fit_params.state == "train":
            if "budget_opt" in self.rates:
                self.stats["budget_opt"].append(self.wrapper.volume.budget_weights.grad.abs().cpu().numpy())
            for l in self.wrapper.volume.get_detectors():
                if isinstance(l, PanelDetectorLayer):
                    for p in l.panels:
                        if "xy_pos_opt" in self.rates:
                            self.stats["xy_pos_opt"].append(p.xy.grad.abs().cpu().numpy())
                        if "z_pos_opt" in self.rates:
                            self.stats["z_pos_opt"].append(p.z.grad.abs().cpu().numpy())
                        if "xy_span_opt" in self.rates:
                            self.stats["xy_span_opt"].append(p.xy_span.grad.abs().cpu().numpy())

    def on_epoch_end(self) -> None:
        r"""
        When enough training epochs have passed, sets suitable learning rates for the optimisers based on the median gradients and desired update rates
        """

        if not self.warmup_active:
            return
        super().on_epoch_end()
        if self.wrapper.fit_params.state == "train":
            if self.epoch_cnt >= self.n_warmup:
                print(f"{type(self).__name__}: Optimiser warm-up completed")
                pt = PrettyTable(["Param", "Median Grad", "LR"])
                for k, v in self.stats.items():  # Allow optimisation
                    avg = np.nanmedian(v)
                    lr = self.rates[k] / avg
                    pt.add_row([k, avg, lr])
                    self.wrapper.set_opt_lr(lr, k)
                print(pt)

    def _reset(self) -> None:
        super()._reset()
        self.stats: Dict[str, List[np.ndarray]] = {k: [] for k in self.rates}
