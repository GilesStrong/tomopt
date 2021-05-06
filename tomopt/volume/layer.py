from typing import Optional, Callable, Dict
import math
from abc import abstractmethod

import torch
from torch import nn, Tensor

from ..core import DEVICE, SCATTER_COEF_A
from ..muon import MuonBatch

__all__ = ["PassiveLayer", "DetectorLayer"]


class Layer(nn.Module):
    def __init__(self, lw: Tensor, z: float, size: float, device: torch.device = DEVICE):
        super().__init__()
        self.lw, self.z, self.size, self.device = lw, Tensor([z]), size, device
        self.rad_length: Optional[Tensor] = None

    def scatter_and_propagate(self, mu: MuonBatch, deltaz: float) -> None:
        """
        This function produces a model of multiple scattering through a layer of material
        of depth deltaz

        TODO: Expand to sum over traversed voxels
        """

        if self.rad_length is not None:
            mask = mu.get_xy_mask(self.lw)  # Only scatter muons inside volume
            n = mask.sum().cpu().numpy()
            xy_idx = self.mu_abs2idx(mu, mask)

            x0 = deltaz / (self.rad_length[xy_idx[:, 0], xy_idx[:, 1]] * torch.cos(mu.theta[mask]))
            z1 = torch.randn(n, device=self.device)
            z2 = torch.randn(n, device=self.device)

            theta0 = (SCATTER_COEF_A / mu.p[mask]) * torch.sqrt(x0)  # Ignore due to inversion problems * (1+(SCATTER_COEF_B*torch.log(x0)))
            theta_msc = math.sqrt(2) * z2 * theta0
            phi_msc = torch.rand(n, device=self.device) * 2 * math.pi
            dh_msc = deltaz * torch.sin(theta0) * ((z1 / math.sqrt(12)) + (z2 / 2))
            dx_msc = math.sqrt(2) * dh_msc * torch.cos(phi_msc) * torch.cos(mu.theta_x[mask])  # we need to account for direction of incident particle!
            dy_msc = math.sqrt(2) * dh_msc * torch.sin(phi_msc) * torch.cos(mu.theta_y[mask])  # ... so we project onto the surface of the layer

            # Update to position at scattering. Note that if a track incides on a layer
            # with angle theta_mu, the dx and dy displacements are relative to zero angle
            # (generation of MSC formulas are oblivious of angle of incidence) so we need
            # to rescale them by cos of thetax and thetay.
            # ---------------------------------------------------------------------------
            mu.x[mask] = mu.x[mask] + dx_msc
            mu.y[mask] = mu.y[mask] + dy_msc
            mu.propagate(deltaz)
            mu.theta_x[mask] = mu.theta_x[mask] + theta_msc * torch.cos(phi_msc)
            mu.theta_y[mask] = mu.theta_y[mask] + theta_msc * torch.sin(phi_msc)
        else:
            mu.propagate(deltaz)

    def mu_abs2idx(self, mu: MuonBatch, mask: Optional[Tensor] = None) -> Tensor:
        xy = mu.xy
        if mask is not None:
            xy = xy[mask]
        return self.abs2idx(xy)

    def abs2idx(self, xy: Tensor) -> Tensor:
        return torch.floor(xy / self.size).long()

    @abstractmethod
    def forward(self, mu: MuonBatch) -> None:
        pass


class PassiveLayer(Layer):
    def __init__(self, rad_length_func: Callable[..., Tensor], lw: Tensor, z: float, size: float, device: torch.device = DEVICE):
        super().__init__(lw=lw, z=z, size=size, device=device)
        self.rad_length = rad_length_func(z=self.z, lw=self.lw, size=self.size).to(self.device)

    def forward(self, mu: MuonBatch, n: int = 2) -> None:
        for _ in range(n):
            self.scatter_and_propagate(mu, deltaz=self.size / n)


class DetectorLayer(Layer):
    def __init__(
        self,
        pos: str,
        init_res: float,
        init_eff: float,
        lw: Tensor,
        z: float,
        size: float,
        eff_cost_func: Callable[[Tensor], Tensor],
        res_cost_func: Callable[[Tensor], Tensor],
        device: torch.device = DEVICE,
    ):
        super().__init__(lw=lw, z=z, size=size, device=device)
        self.pos = pos
        self.resolution = nn.Parameter(torch.zeros(list((self.lw / size).long()), device=self.device) + init_res)
        self.efficiency = nn.Parameter(torch.zeros(list((self.lw / size).long()), device=self.device) + init_eff)
        self.eff_cost_func, self.res_cost_func = eff_cost_func, res_cost_func

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:  # to dense and add precision
        mask = mu.get_xy_mask(self.lw)
        res, eff = torch.zeros(len(mu), device=self.device), torch.zeros(len(mu), device=self.device)  # Zero detection outside detector
        xy_idxs = self.mu_abs2idx(mu, mask)
        res[mask] = self.resolution[xy_idxs[:, 0], xy_idxs[:, 1]]
        eff[mask] = self.efficiency[xy_idxs[:, 0], xy_idxs[:, 1]]

        # TODO clamp deviation so that reco hit is always inside gen hit's detector element
        hits = {
            "xy": torch.stack(
                [
                    mu.x + (torch.randn(len(mu), device=self.device) / (torch.abs(res) + 1e-17)),  # Inverse resolution
                    mu.y + (torch.randn(len(mu), device=self.device) / (torch.abs(res) + 1e-17)),
                ],
                dim=1,
            ),
            "z": self.z.expand_as(mu.x)[:, None] - (self.size / 2),
        }
        return hits

    def forward(self, mu: MuonBatch) -> None:
        self.scatter_and_propagate(mu, self.size / 2)
        mu.append_hits(self.get_hits(mu), self.pos)
        self.scatter_and_propagate(mu, self.size / 2)

    def get_cost(self) -> Tensor:
        return self.eff_cost_func(self.efficiency).sum() + self.res_cost_func(self.resolution).sum()
