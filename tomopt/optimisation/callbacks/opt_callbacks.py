from prettytable import PrettyTable
import numpy as np
from typing import Dict, List, Optional

from .callback import Callback
from ...volume import PanelDetectorLayer

r"""
Provides callbacks that affect the optimisers
"""

__all__ = ["PanelOptConfig"]


class PanelOptConfig(Callback):
    r"""
    Allows the user to specify the desired update steps for :class:`~tomopt.volume.layer.PanelDetectorLayer`s in physical units.
    Over the course of several warm-up epochs the gradients on the parameters are monitored, after which suitable learning rates for the optimisation are set.
    During the warm-up, the detectors will not be updated as optimiser learning rates will be set to zero.
    """

    def __init__(
        self,
        n_warmup: int,
        xy_pos_rate: Optional[float] = None,
        z_pos_rate: Optional[float] = None,
        xy_span_rate: Optional[float] = None,
        budget_rate: Optional[float] = None,
    ):
        r"""
        Arguments:
            n_warmup: number of training epochs to wait before setting learning rates
            xy_pos_rate: desired distance in metres the panels should move in xy every update
            z_pos_rate: desired distance in metres the panels should move in z every update
            xy_span_rate: desired distance in metres the panels should expand in xy every update
            budget_rate: desired rate at which the fractional budget assignments should change every update
        """

        self.n_warmup = n_warmup
        self.rates: Dict[str, float] = {}
        if xy_pos_rate is not None and xy_pos_rate != 0:
            self.rates["xy_pos_opt"] = xy_pos_rate
        if z_pos_rate is not None and z_pos_rate != 0:
            self.rates["z_pos_opt"] = z_pos_rate
        if xy_span_rate is not None and xy_span_rate != 0:
            self.rates["xy_span_opt"] = xy_span_rate
        if budget_rate is not None and budget_rate != 0:
            self.rates["budget_opt"] = budget_rate

    def on_train_begin(self) -> None:
        r"""
        Prepares to begin tracking losses, and sets optimiser learning rates to zero
        """

        super().on_train_begin()
        self.epoch_cnt = 0
        self.tracking = True
        self.stats: Dict[str, List[np.ndarray]] = {k: [] for k in self.rates}
        print(f"{type(self).__name__}: Freezing optimisation for {self.n_warmup} epochs")
        for o in self.stats:  # Prevent updates during warmup
            self.wrapper.set_opt_lr(0.0, o)

    def on_backwards_end(self) -> None:
        r"""
        Grabs training gradients from detector parameters
        """

        if self.tracking and self.wrapper.fit_params.state == "train":
            if "budget_opt" in self.rates:
                self.stats["budget_opt"].append(self.wrapper.volume.budget_weights.grad.cpu().numpy())
            for l in self.wrapper.volume.get_detectors():
                if isinstance(l, PanelDetectorLayer):
                    for p in l.panels:
                        if "xy_pos_opt" in self.rates:
                            self.stats["xy_pos_opt"].append(p.xy.grad.cpu().numpy())
                        if "z_pos_opt" in self.rates:
                            self.stats["z_pos_opt"].append(p.z.grad.cpu().numpy())
                        if "xy_span_opt" in self.rates:
                            self.stats["xy_span_opt"].append(p.xy_span.grad.cpu().numpy())

    def on_epoch_end(self) -> None:
        r"""
        When enough training epochs have passed, sets suitable learning rates for the optimisers based on the median gradients and desired update rates
        """

        if self.tracking and self.wrapper.fit_params.state == "train":
            self.epoch_cnt += 1
            if self.epoch_cnt == self.n_warmup:
                print(f"{type(self).__name__}: Optimiser warm-up completed")
                self.tracking = False
                pt = PrettyTable(["Param", "Median Grad", "LR"])
                for k, v in self.stats.items():  # Allow optimisation
                    avg = np.abs(np.median(v))
                    lr = self.rates[k] / avg
                    pt.add_row([k, avg, lr])
                    self.wrapper.set_opt_lr(lr, k)
                print(pt)
