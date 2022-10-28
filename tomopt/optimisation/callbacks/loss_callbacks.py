import torch

from .callback import Callback

r"""
Provides callbacks that affect the loss functions or values
"""

__all__ = ["CostCoefWarmup"]


class CostCoefWarmup(Callback):
    r"""
    Sets a more stable cost coefficient in the :class:`~tomopt.optimisation.loss.loss.AbsDetectorLoss`
    by averaging the inference-error component for several epochs.
    During this warm-up monitoring phase, the detectors will be kept fixed.
    """

    def __init__(self, n_warmup: int):
        r"""
        Arguments:
            n_warmup: number of training epochs to wait before setting the cost coefficient
        """

        self.n_warmup = n_warmup

    def on_train_begin(self) -> None:
        r"""
        Prepares for warm-up by setting the learning rate of all optimisers to zero.
        After warm-up, the learning rates will be set back to their original values.
        """

        super().on_train_begin()
        self.e_sum = torch.zeros(1, device=self.wrapper.device)
        self.epoch_cnt = 0
        self.tracking = True
        self.lrs = {}
        print(f"{type(self).__name__}: Freezing optimisation for {self.n_warmup} epochs")
        for o in self.wrapper.opts:  # Prevent updates during warmup (can't freeze due to grad in inference)
            self.lrs[o] = self.wrapper.get_opt_lr(o)
            self.wrapper.set_opt_lr(0.0, o)

    def on_epoch_begin(self) -> None:
        r"""
        Prepares to compute the average inference-error for the epoch
        """

        self.v_sum = torch.zeros(1, device=self.wrapper.device)
        self.volume_cnt = 0

    def on_volume_end(self) -> None:
        r"""
        If training, grabs the inference-error for the latest volume and adds to the running sum
        """

        if self.tracking and self.wrapper.fit_params.state == "train" and self.wrapper.fit_params.pred is not None:
            self.v_sum += self.wrapper.loss_func.sub_losses["error"].detach().clone()
            self.volume_cnt += 1

    def on_epoch_end(self) -> None:
        r"""
        If training, adds the epoch mean inference error to a new running sum of errors.
        If then enough epochs have past, the overall mean inference-error is computed and used to set the cost coefficient in the loss,
        and the learning rates of the optimisers are set back to their original non-zero values.
        """

        if self.tracking and self.wrapper.fit_params.state == "train":
            self.e_sum += self.v_sum / self.volume_cnt
            self.epoch_cnt += 1
            if self.epoch_cnt == self.n_warmup:
                avg = self.e_sum / self.epoch_cnt
                print(f"{type(self).__name__}: Warmed up, average error = {avg}")
                self.wrapper.loss_func.cost_coef = avg
                self.tracking = False
                for o in self.wrapper.opts:  # Allow optimisation
                    self.wrapper.set_opt_lr(self.lrs[o], o)
