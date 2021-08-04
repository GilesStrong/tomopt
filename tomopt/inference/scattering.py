from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor, nn

from ..muon import MuonBatch
from ..volume import Volume
from ..utils import jacobian

__all__ = ["ScatterBatch"]


class ScatterBatch:
    def __init__(self, mu: MuonBatch, volume: Volume):
        self.mu, self.volume = mu, volume
        self.hits = self.mu.get_hits(self.volume.lw)
        self.compute_scatters()

    def compute_scatters(self) -> None:
        r"""
        Currently only handles 2 detectors above and below passive volume

        Scatter locations adapted from:
        @MISC {3334866,
            TITLE = {Closest points between two lines},
            AUTHOR = {Brian (https://math.stackexchange.com/users/72614/brian)},
            HOWPUBLISHED = {Mathematics Stack Exchange},
            NOTE = {URL:https://math.stackexchange.com/q/3334866 (version: 2019-08-26)},
            EPRINT = {https://math.stackexchange.com/q/3334866},
            URL = {https://math.stackexchange.com/q/3334866}
        }
        """

        self.xa0 = torch.cat([self.hits["above"]["xy"][:, 0], self.hits["above"]["z"][:, 0]], dim=-1)  # reco x, reco y, gen z
        self.xa1 = torch.cat([self.hits["above"]["xy"][:, 1], self.hits["above"]["z"][:, 1]], dim=-1)
        self.xb0 = torch.cat([self.hits["below"]["xy"][:, 1], self.hits["below"]["z"][:, 1]], dim=-1)
        self.xb1 = torch.cat([self.hits["below"]["xy"][:, 0], self.hits["below"]["z"][:, 0]], dim=-1)

        dets = self.volume.get_detectors()
        res = []
        for p, l, i in zip(("above", "above", "below", "below"), dets, (0, 1, 1, 0)):
            x = l.abs2idx(self.hits[p]["xy"][:, i])
            r = 1 / l.resolution[x[:, 0], x[:, 1]]
            res.append(torch.stack([r, r, torch.zeros_like(r)], dim=-1))
        self._hit_unc = torch.stack(res, dim=1)

        # Extrapolate muon-path vectors from hits
        v1 = self.xa1 - self.xa0
        v2 = self.xb1 - self.xb0

        # scatter locations
        v3 = torch.cross(v1, v2, dim=1)  # connecting vector perpendicular to both lines
        rhs = self.xb0 - self.xa0
        lhs = torch.stack([v1, -v2, v3], dim=1).transpose(2, 1)
        coefs = torch.linalg.solve(lhs, rhs)  # solve p1+t1*v1 + t3*v3 = p2+t2*v2 => p2-p1 = t1*v1 - t2*v2 + t3*v3

        q1 = self.xa0 + (coefs[:, 0:1] * v1)  # closest point on v1
        self._loc = q1 + (coefs[:, 2:3] * v3 / 2)  # Move halfway along v3 from q1
        self._loc_unc: Optional[Tensor] = None

        # Theta deviations
        self._theta_in = torch.arctan(v1[:, :2] / v1[:, 2:3])
        self._theta_out = torch.arctan(v2[:, :2] / v2[:, 2:3])
        self._dtheta = torch.abs(self._theta_in - self._theta_out)
        self._theta_in_unc: Optional[Tensor] = None
        self._theta_out_unc: Optional[Tensor] = None
        self._dtheta_unc: Optional[Tensor] = None

        # xy deviations
        self._dxy = coefs[:, 2:3] * v3[:, :2]
        self._dxy_unc: Optional[Tensor] = None

    def _compute_unc(self, var: Tensor, hits: List[Tensor], hit_uncs: List[Tensor]) -> Tensor:
        unc2_sum = None
        for i, (xi, unci) in enumerate(zip(hits, hit_uncs)):
            for j, (xj, uncj) in enumerate(zip(hits, hit_uncs)):
                if j < i:
                    continue
                dv_dx_2 = jacobian(var, xi).sum((2)) * jacobian(var, xj).sum((2)) if i != j else jacobian(var, xi).sum((2)) ** 2  # Muons, var_xyz, hit_xyz
                unc_2 = (dv_dx_2 * unci[:, None] * uncj[:, None]).sum(2)  # Muons, (x,y,z)
                if unc2_sum is None:
                    unc2_sum = unc_2
                else:
                    unc2_sum = unc2_sum + unc_2
        return torch.sqrt(unc2_sum)

    @property
    def location(self) -> Tensor:
        return self._loc

    @property
    def location_unc(self) -> Tensor:
        if self._loc_unc is None:
            self._loc_unc = self._compute_unc(
                self._loc, [self.xa0, self.xa1, self.xb0, self.xb1], [self._hit_unc[:, 0], self._hit_unc[:, 1], self._hit_unc[:, 2], self._hit_unc[:, 3]]
            )
        return self._loc_unc

    @property
    def dtheta(self) -> Tensor:
        return self._dtheta

    @property
    def dtheta_unc(self) -> Tensor:
        if self._dtheta_unc is None:
            self._dtheta_unc = self._compute_unc(
                self._dtheta, [self.xa0, self.xa1, self.xb0, self.xb1], [self._hit_unc[:, 0], self._hit_unc[:, 1], self._hit_unc[:, 2], self._hit_unc[:, 3]]
            )
        return self._dtheta_unc

    @property
    def dxy(self) -> Tensor:
        return self._dxy

    @property
    def dxy_unc(self) -> Tensor:
        if self._dxy_unc is None:
            self._dxy_unc = self._compute_unc(
                self._loc, [self.xa0, self.xa1, self.xb0, self.xb1], [self._hit_unc[:, 0], self._hit_unc[:, 1], self._hit_unc[:, 2], self._hit_unc[:, 3]]
            )
        return self._dxy_unc

    @property
    def theta_out(self) -> Tensor:
        return self._theta_out

    @property
    def theta_out_unc(self) -> Tensor:
        if self._theta_out_unc is None:
            self._theta_out_unc = self._compute_unc(self._loc, [self.xb0, self.xb1], [self._hit_unc[:, 2], self._hit_unc[:, 3]])
        return self._theta_out_unc

    @property
    def theta_in(self) -> Tensor:
        return self._theta_in

    @property
    def theta_in_unc(self) -> Tensor:
        if self._theta_in_unc is None:
            self._theta_in_unc = self._compute_unc(self._loc, [self.xa0, self.xa1], [self._hit_unc[:, 0], self._hit_unc[:, 1]])
        return self._theta_in_unc

    def plot_scatter(self, idx: int) -> None:
        x = np.hstack([self.hits["above"]["xy"][idx, :, 0].detach().cpu().numpy(), self.hits["below"]["xy"][idx, :, 0].detach().cpu().numpy()])
        y = np.hstack([self.hits["above"]["xy"][idx, :, 1].detach().cpu().numpy(), self.hits["below"]["xy"][idx, :, 1].detach().cpu().numpy()])
        z = np.hstack([self.hits["above"]["z"][idx, :, 0].detach().cpu().numpy(), self.hits["below"]["z"][idx, :, 0].detach().cpu().numpy()])
        scatter = self.location[idx].detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].scatter(x, z)
        axs[0].scatter(scatter[0], scatter[2], label=r"$\Delta\theta=" + f"{self.dtheta[idx,0]:.1e}$")
        axs[0].set_xlim(0, 1)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("z")
        axs[0].legend()
        axs[1].scatter(y, z)
        axs[1].scatter(scatter[1], scatter[2], label=r"$\Delta\theta=" + f"{self.dtheta[idx,1]:.1e}$")
        axs[1].set_xlim(0, 1)
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("z")
        axs[1].legend()
        plt.show()

    def get_scatter_mask(self) -> Tensor:
        z = self.volume.get_passive_z_range()
        return (
            (self.location[:, 0] >= 0)
            * (self.location[:, 0] < self.volume.lw[0])
            * (self.location[:, 1] >= 0)
            * (self.location[:, 1] < self.volume.lw[1])
            * (self.location[:, 2] >= z[0])
            * (self.location[:, 2] < z[1])
        )
