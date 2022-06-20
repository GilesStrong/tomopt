from prettytable import PrettyTable
import numpy as np
from typing import Dict, List, Optional

from .callback import Callback
from ...volume import PanelDetectorLayer

__all__ = ["PanelOptConfig"]


class PanelOptConfig(Callback):
    def __init__(
        self,
        n_warmup: int,
        xy_pos_rate: Optional[float] = None,
        z_pos_rate: Optional[float] = None,
        xy_span_rate: Optional[float] = None,
        budget_rate: Optional[float] = None,
    ):
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
        super().on_train_begin()
        self.epoch_cnt = 0
        self.tracking = True
        self.stats: Dict[str, List[np.ndarray]] = {k: [] for k in self.rates}
        print(f"{type(self).__name__}: Freezing optimisation for {self.n_warmup} epochs")
        for o in self.stats:  # Prevent updates during warmup
            self.wrapper.set_opt_lr(0.0, o)

    def on_backwards_end(self) -> None:
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
