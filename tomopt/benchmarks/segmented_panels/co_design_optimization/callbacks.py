from typing import List, Optional

import torch

# tomopt imports
from tomopt.optimisation.callbacks import Callback


class HypothesisTestCallback(Callback):
    """Callback to perform hypothesis testing on passives after each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.is_signal_phase: bool = False

    def on_volume_batch_begin(self) -> None:
        """Initialize data collection for the current batch phase."""
        if not self.is_signal_phase:
            # Background phase - first call
            self.scatter_dists_bkg: Optional[List[torch.Tensor]] = []
            self.angular_res_means: Optional[List[torch.Tensor]] = []
        else:
            # Signal phase - second call
            self.scatter_dists_sig: List[torch.Tensor] = []

    def on_volume_begin(self) -> None:
        self.scatter_batch: List[torch.Tensor] = []
        self.sigma_theta: List[torch.Tensor] = []

    def on_scatter_end(self) -> None:
        self.scatter_batch.append(self.wrapper.fit_params.sb.total_scatter)
        delta_theta_in = self.wrapper.fit_params.sb.theta_in - self.wrapper.fit_params.sb.theta_in_gen  # type: ignore[attr-defined]
        delta_theta_out = self.wrapper.fit_params.sb.theta_out - self.wrapper.fit_params.sb.theta_out_gen  # type: ignore[attr-defined]
        self.sigma_theta.append((delta_theta_in.std() + delta_theta_out.std()) / 2)

    def on_volume_end(self) -> None:
        scatters = torch.cat(self.scatter_batch, dim=0)
        if not self.is_signal_phase:
            self.scatter_dists_bkg.append(scatters)
        else:
            self.scatter_dists_sig.append(scatters)
            self.angular_res_means.append(torch.mean(torch.stack(self.sigma_theta)))

    def on_volume_batch_end(self) -> None:
        # self.wrapper.loss_func.scatters_bkg = self.scatter_dists_bkg
        # self.wrapper.loss_func.scatters_sig = self.scatter_dists_sig
        # self.wrapper.loss_func.avg_scatter_bkg = torch.mean(torch.cat(self.scatter_dists_bkg, dim=0), dim=0)
        # self.wrapper.loss_func.angular_res = torch.mean(torch.stack(self.angular_res_means))
        self.wrapper.loss_func.set_scatter_data(
            scatters_bkg=self.scatter_dists_bkg,
            scatters_sig=self.scatter_dists_sig,
            avg_scatter_bkg=torch.mean(torch.cat(self.scatter_dists_bkg, dim=0), dim=0),
            angular_res=torch.mean(torch.stack(self.angular_res_means)),
        )
        self.is_signal_phase = False  # Reset for next epoch
