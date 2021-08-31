from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from ..muon import MuonBatch
from ..volume import Volume, DetectorPanel
from ..volume.layer import AbsDetectorLayer, VoxelDetectorLayer, PanelDetectorLayer
from ..utils import jacobian

__all__ = ["VoxelScatterBatch", "PanelScatterBatch"]


class AbsScatterBatch(metaclass=ABCMeta):
    track_in: Tensor
    track_out: Tensor
    track_start_in: Tensor
    track_start_out: Tensor
    above_hits: List[Tensor]
    below_hits: List[Tensor]
    above_gen_hits: List[Tensor]
    below_gen_hits: List[Tensor]
    above_hit_uncs: List[Tensor]
    below_hit_uncs: List[Tensor]
    _loc: Tensor
    _loc_unc: Optional[Tensor] = None
    _theta_in: Tensor
    _theta_out: Tensor
    _dtheta: Tensor
    _theta_in_unc: Optional[Tensor] = None
    _theta_out_unc: Optional[Tensor] = None
    _dtheta_unc: Optional[Tensor] = None
    _dxy: Tensor
    _dxy_unc: Optional[Tensor] = None

    def __init__(self, mu: MuonBatch, volume: Volume):
        self.mu, self.volume = mu, volume
        self.hits = self.mu.get_hits(self.volume.lw)
        self.compute_scatters()

    @staticmethod
    def get_muon_trajectory(hit_list: List[Tensor], unc_list: List[Tensor], lw: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        hits = [muons,(x,y,z)]
        uncs = [(unc,unc,0)]

        Assume no uncertainty for z

        In eval mode:
            Muons with <2 hits within panels have NaN trajectory.
            Muons with >=2 hits in panels have valid trajectories
        """

        hits, uncs = torch.stack(hit_list, dim=1), torch.stack(unc_list, dim=1)
        hits = torch.where(torch.isinf(hits), lw.mean() / 2, hits)

        stars, angles = [], []
        for i in range(2):  # seperate x and y resolutions
            inv_unc2 = uncs[:, :, i : i + 1] ** -2
            sum_inv_unc2 = inv_unc2.sum(dim=1)
            mean_xz = torch.sum(hits[:, :, [i, 2]] * inv_unc2, dim=1) / sum_inv_unc2
            mean_xz_z = torch.sum(hits[:, :, [i, 2]] * hits[:, :, 2:3] * inv_unc2, dim=1) / sum_inv_unc2
            mean_x = mean_xz[:, :1]
            mean_z = mean_xz[:, 1:]
            mean_x_z = mean_xz_z[:, :1]
            mean_z2 = mean_xz_z[:, 1:]

            stars.append((mean_x - ((mean_z * mean_x_z) / mean_z2)) / (1 - (mean_z.square() / mean_z2)))
            angles.append((mean_x_z - (stars[-1] * mean_z)) / mean_z2)

        xy_star = torch.cat(stars, dim=-1)
        angle = torch.cat(angles, dim=-1)

        def _calc_xyz(z: Tensor) -> Tensor:
            return torch.cat([xy_star + (angle * z), z], dim=-1)

        start = _calc_xyz(hits[:, 0, 2:3])  # Upper & lower hits. Only z coord is used therefore ok if xy were NaN/Inf
        end = _calc_xyz(hits[:, 1, 2:3])
        vec = end - start

        return vec, start

    def extract_hits(self) -> None:
        # reco x, reco y, gen z, must be a list to allow computation of uncertainty
        self.above_hits = [
            torch.cat([self.hits["above"]["reco_xy"][:, i], self.hits["above"]["z"][:, i]], dim=-1) for i in range(self.hits["above"]["reco_xy"].shape[1])
        ]
        self.below_hits = [
            torch.cat([self.hits["below"]["reco_xy"][:, i], self.hits["below"]["z"][:, i]], dim=-1) for i in range(self.hits["below"]["reco_xy"].shape[1])
        ]
        self.above_gen_hits = [
            torch.cat([self.hits["above"]["gen_xy"][:, i], self.hits["above"]["z"][:, i]], dim=-1) for i in range(self.hits["above"]["gen_xy"].shape[1])
        ]
        self.below_gen_hits = [
            torch.cat([self.hits["below"]["gen_xy"][:, i], self.hits["below"]["z"][:, i]], dim=-1) for i in range(self.hits["below"]["gen_xy"].shape[1])
        ]
        self.n_hits_above = len(self.above_hits)
        self.n_hits_below = len(self.below_hits)

    @abstractmethod
    def compute_tracks(self) -> None:
        pass

    @staticmethod
    def compute_coefs(v1: Tensor, v2: Tensor, v3: Tensor, p1: Tensor, p2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        print(v1[0], v2[0], v3[0], p1[0], p2[0])
        # solve point_1+t1*v1 + t3*v3 = p2+t2*v2 => p2-p1 = t1*v1 - t2*v2 + t3*v3
        dp = p2 - p1
        v1x = v1[:, 0:1]
        v1y = v1[:, 1:2]
        v1z = v1[:, 2:3]
        v2x = v2[:, 0:1]
        v2y = v2[:, 1:2]
        v2z = v2[:, 2:3]
        v3x = v3[:, 0:1]
        v3y = v3[:, 1:2]
        v3z = v3[:, 2:3]
        dpx = dp[:, 0:1]
        dpy = dp[:, 1:2]
        dpz = dp[:, 2:3]
        a = (v2x * v1y) - (v2y * v1x)

        t3 = ((dpx * (-(v1y * v1z * v2x) + (a * v1z) + (v1x * v1y * v2z))) + (dpy * ((v1x * v1z * v2x) - (v1x.square() * v2z))) + (a * dpz * v1x)) / (
            -(v1y * v1z * v2x * v3x) + (v1x * v1z * v2x * v3y) + (a * v1z * v3x) - (v1x * v1y * v2z * v3x) + (v1x.square() * v2z * v3y) - (a * v1x * v3z)
        )

        t2 = -((t3 * ((v1y * v3x) - (v1x * v3y))) - (dpx * v1y) + (dpy * v1x)) / a

        t1 = -((t2 * v2x) + (t3 * v3x) - dpx) / v1x

        return t1, t2, t3

    def compute_scatters(self) -> None:
        r"""
        Currently only handles detectors above and below passive volume

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

        self.extract_hits()
        self.compute_tracks()

        # scatter locations
        cross = torch.cross(self.track_in, self.track_out, dim=1)  # connecting vector perpendicular to both lines

        t1, t2, t3 = self.compute_coefs(self.track_in, self.track_out, cross, self.track_start_in, self.track_start_out)
        q1 = self.track_start_in + (t1 * self.track_in)  # closest point on v1
        self._loc = q1 + (t2 * cross / 2)  # Move halfway along v3 from q1
        self._loc_unc = None

        rhs = self.track_start_out - self.track_start_in
        lhs = torch.stack([self.track_in, -self.track_out, cross], dim=1).transpose(2, 1)
        coefs = torch.linalg.solve(
            lhs, rhs
        )  # solve point_1+t1*track_in + t3*cross = point_2+t2*track_out => point_2-point_1 = t1*track_in - t2*track_out + t3*cross
        c2 = torch.inverse(lhs) * rhs

        print(coefs[0], t1[0], t2[0], t3[0], c2[0], c2.shape)
        # q1 = self.above_hits[0] + (coefs[:, 0:1] * self.track_in)  # closest point on v1
        # self._loc = q1 + (coefs[:, 2:3] * cross / 2)  # Move halfway along v3 from q1
        # self._loc_unc = None

        # Theta deviations
        self._theta_in = torch.arctan(self.track_in[:, :2] / self.track_in[:, 2:3])
        self._theta_out = torch.arctan(self.track_out[:, :2] / self.track_out[:, 2:3])
        self._dtheta = torch.abs(self._theta_in - self._theta_out)
        self._theta_in_unc = None
        self._theta_out_unc = None
        self._dtheta_unc = None

        # xy deviations
        self._dxy = t3 * cross[:, :2]
        self._dxy_unc = None

    @abstractmethod
    def _compute_unc(self, var: Tensor, hits: List[Tensor], hit_uncs: List[Tensor]) -> Tensor:
        pass

    @property
    def location(self) -> Tensor:
        return self._loc

    @property
    def location_unc(self) -> Tensor:
        if self._loc_unc is None:
            self._loc_unc = self._compute_unc(
                var=self._loc,
                hits=self.above_hits + self.below_hits,
                hit_uncs=self.above_hit_uncs + self.below_hit_uncs,
            )
        return self._loc_unc

    @property
    def dtheta(self) -> Tensor:
        return self._dtheta

    @property
    def dtheta_unc(self) -> Tensor:
        if self._dtheta_unc is None:
            self._dtheta_unc = self._compute_unc(
                var=self._dtheta,
                hits=self.above_hits + self.below_hits,
                hit_uncs=self.above_hit_uncs + self.below_hit_uncs,
            )
        return self._dtheta_unc

    @property
    def dxy(self) -> Tensor:
        return self._dxy

    @property
    def dxy_unc(self) -> Tensor:
        if self._dxy_unc is None:
            self._dxy_unc = self._compute_unc(
                var=self._dxy,
                hits=self.above_hits + self.below_hits,
                hit_uncs=self.above_hit_uncs + self.below_hit_uncs,
            )
        return self._dxy_unc

    @property
    def theta_in(self) -> Tensor:
        return self._theta_in

    @property
    def theta_in_unc(self) -> Tensor:
        if self._theta_in_unc is None:
            self._theta_in_unc = self._compute_unc(var=self._theta_in, hits=self.above_hits, hit_uncs=self.above_hit_uncs)
        return self._theta_in_unc

    @property
    def theta_out(self) -> Tensor:
        return self._theta_out

    @property
    def theta_out_unc(self) -> Tensor:
        if self._theta_out_unc is None:
            self._theta_out_unc = self._compute_unc(var=self._theta_out, hits=self.below_hits, hit_uncs=self.below_hit_uncs)
        return self._theta_out_unc

    def plot_scatter(self, idx: int) -> None:
        x = np.hstack([self.hits["above"]["reco_xy"][idx, :, 0].detach().cpu().numpy(), self.hits["below"]["reco_xy"][idx, :, 0].detach().cpu().numpy()])
        y = np.hstack([self.hits["above"]["reco_xy"][idx, :, 1].detach().cpu().numpy(), self.hits["below"]["reco_xy"][idx, :, 1].detach().cpu().numpy()])
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


class VoxelScatterBatch(AbsScatterBatch):
    @staticmethod
    def _get_hit_uncs(dets: List[AbsDetectorLayer], hits: List[Tensor]) -> List[Tensor]:
        uncs = []
        for i, (l, h) in enumerate(zip(dets, hits)):
            if not isinstance(l, VoxelDetectorLayer):
                raise ValueError(f"Detector {l} is not a VoxelDetectorLayer")
            x = l.abs2idx(h)
            r = 1 / l.resolution[x[:, 0], x[:, 1]]
            uncs.append(torch.stack([r, r, torch.zeros_like(r)], dim=-1))
        return uncs

    def compute_tracks(self) -> None:
        self.above_hit_uncs = self._get_hit_uncs(self.volume.get_detectors()[: self.n_hits_above], self.above_hits)
        self.below_hit_uncs = self._get_hit_uncs(self.volume.get_detectors()[self.n_hits_above :], self.below_hits)
        self.track_in, self.track_start_in = self.get_muon_trajectory(self.above_hits, self.above_hit_uncs, self.volume.lw)
        self.track_out, self.track_start_out = self.get_muon_trajectory(self.below_hits, self.below_hit_uncs, self.volume.lw)

    @staticmethod
    def _compute_unc(var: Tensor, hits: List[Tensor], hit_uncs: List[Tensor]) -> Tensor:
        r"""
        Behaviour tested only
        """

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


class PanelScatterBatch(AbsScatterBatch):
    @staticmethod
    def _get_hit_uncs(zordered_panels: List[DetectorPanel], hits: List[Tensor]) -> List[Tensor]:
        uncs: List[Tensor] = []
        for l, h in zip(zordered_panels, hits):
            r = 1 / l.get_resolution(h[:, :2])
            uncs.append(torch.cat([r, torch.zeros((len(r), 1), device=r.device)], dim=-1))
        return uncs

    def compute_tracks(self) -> None:
        def _get_panels(i: int) -> List[DetectorPanel]:
            det = self.volume.get_detectors()[i]
            if not isinstance(det, PanelDetectorLayer):
                raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
            return [det.panels[i] for i in det.get_panel_zorder()]

        self.above_hit_uncs = self._get_hit_uncs(_get_panels(0), self.above_gen_hits)
        self.below_hit_uncs = self._get_hit_uncs(_get_panels(1), self.below_gen_hits)

        self.track_in, self.track_start_in = self.get_muon_trajectory(self.above_hits, self.above_hit_uncs, self.volume.lw)
        self.track_out, self.track_start_out = self.get_muon_trajectory(self.below_hits, self.below_hit_uncs, self.volume.lw)

    @staticmethod
    def _compute_unc(var: Tensor, hits: List[Tensor], hit_uncs: List[Tensor]) -> Tensor:
        r"""
        Behaviour tested only
        """

        unc2_sum = None
        for i, (xi, unci) in enumerate(zip(hits, hit_uncs)):
            unci = torch.where(torch.isinf(unci), torch.tensor([0], device=unci.device).type(unci.type()), unci)[:, None]
            for j, (xj, uncj) in enumerate(zip(hits, hit_uncs)):
                if j < i:
                    continue
                uncj = torch.where(torch.isinf(uncj), torch.tensor([0], device=uncj.device).type(uncj.type()), uncj)[:, None]
                dv_dx_2 = (
                    torch.nan_to_num(jacobian(var, xi)).sum(2) * torch.nan_to_num(jacobian(var, xj)).sum(2)
                    if i != j
                    else torch.nan_to_num(jacobian(var, xi)).sum(2) ** 2
                )  # Muons, var_xyz, hit_xyz
                unc_2 = (dv_dx_2 * unci * uncj).sum(2)  # Muons, (x,y,z)
                if unc2_sum is None:
                    unc2_sum = unc_2
                else:
                    unc2_sum = unc2_sum + unc_2
        return torch.sqrt(unc2_sum)
