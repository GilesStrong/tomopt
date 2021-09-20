import torch

from .callback import Callback

__all__ = ["CostCoefWarmup"]


class CostCoefWarmup(Callback):
    r"""Sets a more stable cost coeficient in the DetectorLoss by averaging the inference-error component for several epochs"""

    def __init__(self, n_warmup: int):
        self.n_warmup = n_warmup

    def on_train_begin(self) -> None:
        super().on_train_begin()
        self.e_sum = torch.zeros(1, device=self.wrapper.device)
        self.epoch_cnt = 0
        self.tracking = True
        self.lrs = {}
        for o in self.wrapper.opts:  # Prevent updates during warmup (can't freeze due to grad in inference)
            self.lrs[o] = self.wrapper.get_opt_lr(o)
            self.wrapper.set_opt_lr(0.0, o)

    def on_epoch_begin(self) -> None:
        self.v_sum = torch.zeros(1, device=self.wrapper.device)
        self.volume_cnt = 0

    def on_volume_end(self) -> None:
        if self.tracking and self.wrapper.fit_params.state == "train" and self.wrapper.fit_params.pred is not None:
            self.v_sum += self.wrapper.loss_func.sub_losses["error"].detach().clone()
            self.volume_cnt += 1

    def on_epoch_end(self) -> None:
        if self.tracking and self.wrapper.fit_params.state == "train":
            self.e_sum += self.v_sum / self.volume_cnt
            self.epoch_cnt += 1
            if self.epoch_cnt == self.n_warmup:
                avg = self.e_sum / self.epoch_cnt
                print(f"Warmed up, average error = {avg}")
                self.wrapper.loss_func.cost_coef = avg
                self.tracking = False
                for o in self.wrapper.opts:  # Allow optimisation
                    self.wrapper.set_opt_lr(self.lrs[o], o)
