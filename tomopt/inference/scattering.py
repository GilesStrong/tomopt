from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from ..muon import MuonBatch
from ..volume import Volume, DetectorPanel
from ..volume.layer import AbsDetectorLayer, VoxelDetectorLayer, PanelDetectorLayer
from ..utils import jacobian

__all__ = ["VoxelScatterBatch", "PanelScatterBatch", "GenScatterBatch"]


class AbsScatterBatch(metaclass=ABCMeta):
    # Hits
    _reco_hits: Optional[Tensor] = None
    _gen_hits: Optional[Tensor] = None
    _hit_uncs: Optional[Tensor] = None
    # Tracks
    _track_in: Optional[Tensor] = None
    _track_out: Optional[Tensor] = None
    _track_start_in: Optional[Tensor] = None
    _track_start_out: Optional[Tensor] = None
    _cross_track: Optional[Tensor] = None
    _track_coefs: Optional[Tensor] = None
    # Inferred variables
    _loc: Optional[Tensor] = None
    _loc_unc: Optional[Tensor] = None
    _theta_in: Optional[Tensor] = None
    _theta_in_unc: Optional[Tensor] = None
    _theta_out: Optional[Tensor] = None
    _theta_out_unc: Optional[Tensor] = None
    _dtheta: Optional[Tensor] = None
    _dtheta_unc: Optional[Tensor] = None
    _phi_in: Optional[Tensor] = None
    _phi_in_unc: Optional[Tensor] = None
    _phi_out: Optional[Tensor] = None
    _phi_out_unc: Optional[Tensor] = None
    _dphi: Optional[Tensor] = None
    _dphi_unc: Optional[Tensor] = None
    _theta_xy_in: Optional[Tensor] = None
    _theta_xy_in_unc: Optional[Tensor] = None
    _theta_xy_out: Optional[Tensor] = None
    _theta_xy_out_unc: Optional[Tensor] = None
    _dtheta_xy: Optional[Tensor] = None
    _dtheta_xy_unc: Optional[Tensor] = None
    _xyz_in: Optional[Tensor] = None
    _xyz_in_unc: Optional[Tensor] = None
    _xyz_out: Optional[Tensor] = None
    _xyz_out_unc: Optional[Tensor] = None
    _dxy: Optional[Tensor] = None
    _dxy_unc: Optional[Tensor] = None

    def __init__(self, mu: MuonBatch, volume: Volume):
        self.mu, self.volume = mu, volume
        self.device = self.mu.device
        self._hits = self.mu.get_hits()
        self.compute_scatters()

    @staticmethod
    def get_muon_trajectory(hits: Tensor, uncs: Tensor, lw: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        hits = (muons,panels,(x,y,z))
        uncs = (muons,panels,(unc,unc,0))

        Assume no uncertainty for z

        In eval mode:
            Muons with <2 hits within panels have NaN trajectory.
            Muons with >=2 hits in panels have valid trajectories
        """

        hits = torch.where(torch.isinf(hits), lw.mean().type(hits.type()) / 2, hits)
        uncs = torch.nan_to_num(uncs)  # Set Infs to large number

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

    @property
    def hits(self) -> Dict[str, Dict[str, Tensor]]:
        return self._hits

    @property
    def reco_hits(self) -> Optional[Tensor]:
        return self._reco_hits

    @property
    def gen_hits(self) -> Optional[Tensor]:
        return self._gen_hits

    @property
    def n_hits_above(self) -> Optional[int]:
        return self._n_hits_above

    @property
    def n_hits_below(self) -> Optional[int]:
        return self._n_hits_below

    @property
    def above_gen_hits(self) -> Optional[Tensor]:
        if self._gen_hits is None:
            return None
        else:
            return self._gen_hits[:, : self.n_hits_above]

    @property
    def below_gen_hits(self) -> Optional[Tensor]:
        if self._gen_hits is None:
            return None
        else:
            return self._gen_hits[:, self.n_hits_above :]

    @property
    def above_hits(self) -> Optional[Tensor]:
        if self._reco_hits is None:
            return None
        else:
            return self._reco_hits[:, : self.n_hits_above]

    @property
    def below_hits(self) -> Optional[Tensor]:
        if self._reco_hits is None:
            return None
        else:
            return self._reco_hits[:, self.n_hits_above :]

    @property
    def hit_uncs(self) -> Optional[Tensor]:
        return self._hit_uncs

    @property
    def above_hit_uncs(self) -> Optional[Tensor]:
        if self._hit_uncs is None:
            return None
        else:
            return self._hit_uncs[:, : self.n_hits_above]

    @property
    def below_hit_uncs(self) -> Optional[Tensor]:
        if self._hit_uncs is None:
            return None
        else:
            return self._hit_uncs[:, self.n_hits_above :]

    def extract_hits(self) -> None:
        # reco x, reco y, gen z, must be a list to allow computation of uncertainty
        above_hits = torch.stack(
            [torch.cat([self.hits["above"]["reco_xy"][:, i], self.hits["above"]["z"][:, i]], dim=-1) for i in range(self.hits["above"]["reco_xy"].shape[1])],
            dim=1,
        )  # muons, panels, xyz
        below_hits = torch.stack(
            [torch.cat([self.hits["below"]["reco_xy"][:, i], self.hits["below"]["z"][:, i]], dim=-1) for i in range(self.hits["below"]["reco_xy"].shape[1])],
            dim=1,
        )
        _above_gen_hits = torch.stack(
            [torch.cat([self.hits["above"]["gen_xy"][:, i], self.hits["above"]["z"][:, i]], dim=-1) for i in range(self.hits["above"]["gen_xy"].shape[1])],
            dim=1,
        )  # muons, panels, xyz
        _below_gen_hits = torch.stack(
            [torch.cat([self.hits["below"]["gen_xy"][:, i], self.hits["below"]["z"][:, i]], dim=-1) for i in range(self.hits["below"]["gen_xy"].shape[1])],
            dim=1,
        )
        self._n_hits_above = above_hits.shape[1]
        self._n_hits_below = below_hits.shape[1]

        # Combine all input vars into single tensor, NB ideally would stack to new dim but can't assume same number of panels above & below
        self._reco_hits = torch.cat((above_hits, below_hits), dim=1)  # muons, all panels, reco xyz
        self._gen_hits = torch.cat((_above_gen_hits, _below_gen_hits), dim=1)  # muons, all panels, true xyz

    @property
    def track_in(self) -> Optional[Tensor]:
        return self._track_in

    @property
    def track_start_in(self) -> Optional[Tensor]:
        return self._track_start_in

    @property
    def track_out(self) -> Optional[Tensor]:
        return self._track_out

    @property
    def track_start_out(self) -> Optional[Tensor]:
        return self._track_start_out

    @abstractmethod
    def compute_tracks(self) -> None:
        pass

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

        # Track computations
        self._cross_track = torch.cross(self.track_in, self.track_out, dim=1)  # connecting vector perpendicular to both lines

        rhs = self.track_start_out - self.track_start_in
        lhs = torch.stack([self.track_in, -self.track_out, self._cross_track], dim=1).transpose(2, 1)
        # coefs = torch.linalg.solve(lhs, rhs)  # solve p1+t1*v1 + t3*v3 = p2+t2*v2 => p2-p1 = t1*v1 - t2*v2 + t3*v3
        self._track_coefs = (lhs.inverse() @ rhs[:, :, None]).squeeze(-1)

    @staticmethod
    def _compute_phi(x: Tensor, y: Tensor) -> Tensor:
        phi = torch.arctan(y / x)  # (-pi/2, pi/2)
        m = x < 0
        phi[m] = phi[m] + torch.pi
        m = ((x > 0) * (y < 0)).bool()
        phi[m] = phi[m] + (2 * torch.pi)  # (0, 2pi)
        return phi

    def _compute_out_var_unc(self, var: Tensor) -> Tensor:
        r"""
        Behaviour tested only
        """

        # Compute dvar/dhits
        jac = torch.nan_to_num(jacobian(var, self.reco_hits)).sum(2)
        unc = torch.where(torch.isinf(self.hit_uncs), torch.tensor([0], device=self.device).type(self.hit_uncs.type()), self.hit_uncs)[:, None]
        jac, unc = jac.reshape(jac.shape[0], jac.shape[1], -1), unc.reshape(unc.shape[0], unc.shape[1], -1)

        # Compute unc^2 = unc_x*unc_y*dvar/dhit_x*dvar/dhit_y summing over all x,y inclusive combinations
        idxs = torch.combinations(torch.arange(0, unc.shape[-1]), with_replacement=True)
        unc_2 = (jac[:, :, idxs] * unc[:, :, idxs]).prod(-1)
        return unc_2.sum(-1).sqrt()

    @property
    def location(self) -> Tensor:
        if self._loc is None:
            q1 = self.track_start_in + (self._track_coefs[:, 0:1] * self.track_in)  # closest point on v1
            self._loc = q1 + (self._track_coefs[:, 2:3] * self._cross_track / 2)  # Move halfway along v3 from q1
            self._loc_unc = None
        return self._loc

    @property
    def location_unc(self) -> Tensor:
        if self._loc_unc is None:
            self._loc_unc = self._compute_out_var_unc(self.location)
        return self._loc_unc

    @property
    def dxy(self) -> Tensor:
        if self._dxy is None:
            self._dxy = self._track_coefs[:, 2:3] * self._cross_track[:, :2]
            self._dxy_unc = None
        return self._dxy

    @property
    def dxy_unc(self) -> Tensor:
        if self._dxy_unc is None:
            self._dxy_unc = self._compute_out_var_unc(self.dxy)
        return self._dxy_unc

    @property
    def theta_in(self) -> Tensor:
        if self._theta_in is None:
            self._theta_in = torch.arccos(-self.track_in[:, 2:3] / self.track_in.norm(dim=-1, keepdim=True))
            self._theta_in_unc = None
        return self._theta_in

    @property
    def theta_in_unc(self) -> Tensor:
        if self._theta_in_unc is None:
            self._theta_in_unc = self._compute_out_var_unc(self.theta_in)
        return self._theta_in_unc

    @property
    def theta_out(self) -> Tensor:
        if self._theta_out is None:
            self._theta_out = torch.arccos(-self.track_out[:, 2:3] / self.track_out.norm(dim=-1, keepdim=True))
            self._theta_out_unc = None
        return self._theta_out

    @property
    def theta_out_unc(self) -> Tensor:
        if self._theta_out_unc is None:
            self._theta_out_unc = self._compute_out_var_unc(self.theta_out)
        return self._theta_out_unc

    @property
    def phi_in(self) -> Tensor:
        if self._phi_in is None:
            self._phi_in = self._compute_phi(self.track_in[:, 1:2], self.track_in[:, 0:1])
            self._phi_in_unc = None
        return self._phi_in

    @property
    def phi_in_unc(self) -> Tensor:
        if self._phi_in_unc is None:
            self._phi_in_unc = self._compute_out_var_unc(self.phi_in)
        return self._phi_in_unc

    @property
    def phi_out(self) -> Tensor:
        if self._phi_out is None:
            self._phi_out = self._compute_phi(self.track_out[:, 1:2], self.track_out[:, 0:1])
            self._phi_out_unc = None
        return self._phi_out

    @property
    def phi_out_unc(self) -> Tensor:
        if self._phi_out_unc is None:
            self._phi_out_unc = self._compute_out_var_unc(self.phi_out)
        return self._phi_out_unc

    @property
    def dtheta(self) -> Tensor:
        r"""
        Volume ref frame
        """

        if self._dtheta is None:
            self._dtheta = torch.abs(self.theta_in - self.theta_out)
            self._dtheta_unc = None
        return self._dtheta

    @property
    def dtheta_unc(self) -> Tensor:
        if self._dtheta_unc is None:
            self._dtheta_unc = self._compute_out_var_unc(self.dtheta)
        return self._dtheta_unc

    @property
    def dphi(self) -> Tensor:
        r"""
        Volume ref frame
        """

        if self._dphi is None:
            # Is there a simpler formular?
            self._dphi = torch.min(
                torch.stack(
                    (
                        ((2 * torch.pi) - self.phi_in) + self.phi_out,
                        ((2 * torch.pi) - self.phi_out) + self.phi_in,
                        torch.abs(self.phi_in - self.phi_out),
                    ),
                    dim=0,
                ),
                dim=0,
            ).values
            self._dphi_unc = None
        return self._dphi

    @property
    def dphi_unc(self) -> Tensor:
        if self._dphi_unc is None:
            self._dphi_unc = self._compute_out_var_unc(self.dphi)
        return self._dphi_unc

    @property
    def theta_xy_in(self) -> Tensor:
        if self._theta_xy_in is None:
            self._theta_xy_in = torch.cat([(self.theta_in.tan() * self.phi_in.cos()).arctan(), (self.theta_in.tan() * self.phi_in.sin()).arctan()], dim=-1)
            self._theta_xy_in_unc = None
        return self._theta_xy_in

    @property
    def theta_xy_in_unc(self) -> Tensor:
        if self._theta_xy_in_unc is None:
            self._theta_xy_in_unc = self._compute_out_var_unc(self.theta_xy_in)
        return self._theta_xy_in_unc

    @property
    def theta_xy_out(self) -> Tensor:
        if self._theta_xy_out is None:
            self._theta_xy_out = torch.cat([(self.theta_out.tan() * self.phi_out.cos()).arctan(), (self.theta_out.tan() * self.phi_out.sin()).arctan()], dim=-1)
            self._theta_xy_out_unc = None
        return self._theta_xy_out

    @property
    def theta_xy_out_unc(self) -> Tensor:
        if self._theta_xy_out_unc is None:
            self._theta_xy_out_unc = self._compute_out_var_unc(self.theta_xy_out)
        return self._theta_xy_out_unc

    @property
    def dtheta_xy(self) -> Tensor:
        r"""
        Volume ref frame
        """

        if self._dtheta_xy is None:
            self._dtheta_xy = torch.abs(self.theta_xy_in - self.theta_xy_out)
            self._dtheta_xy_unc = None
        return self._dtheta_xy

    @property
    def dtheta_xy_unc(self) -> Tensor:
        if self._dtheta_xy_unc is None:
            self._dtheta_xy_unc = self._compute_out_var_unc(self.dtheta_xy)
        return self._dtheta_xy_unc

    @property
    def xyz_in(self) -> Tensor:
        if self._xyz_in is None:
            dz = self.volume.get_passive_z_range()[1] - self._track_start_in[:, 2:3]  # last panel to volume start
            self._xyz_in = self._track_start_in + ((dz / self._track_in[:, 2:3]) * self._track_in)
            self._xyz_in_unc = None
        return self._xyz_in

    @property
    def xyz_in_unc(self) -> Tensor:
        if self._xyz_in_unc is None:
            self._xyz_in_unc = self._compute_out_var_unc(self.xyz_in)
        return self._xyz_in_unc

    @property
    def xyz_out(self) -> Tensor:
        if self._xyz_out is None:
            dz = self._track_start_out[:, 2:3] - (self.volume.get_passive_z_range()[0])  # volume end to first panel
            self._xyz_out = self._track_start_out - ((dz / self._track_out[:, 2:3]) * self._track_out)
            self._xyz_out_unc = None
        return self._xyz_out

    @property
    def xyz_out_unc(self) -> Tensor:
        if self._xyz_out_unc is None:
            self._xyz_out_unc = self._compute_out_var_unc(self.xyz_out)
        return self._xyz_out_unc

    def plot_scatter(self, idx: int) -> None:
        xin, xout = self.hits["above"]["reco_xy"][idx, :, 0].detach().cpu().numpy(), self.hits["below"]["reco_xy"][idx, :, 0].detach().cpu().numpy()
        yin, yout = self.hits["above"]["reco_xy"][idx, :, 1].detach().cpu().numpy(), self.hits["below"]["reco_xy"][idx, :, 1].detach().cpu().numpy()
        zin, zout = self.hits["above"]["z"][idx, :, 0].detach().cpu().numpy(), self.hits["below"]["z"][idx, :, 0].detach().cpu().numpy()
        scatter = self.location[idx].detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        dtheta_xy = self.dtheta_xy[idx].detach().cpu().numpy()

        track_start_in, track_start_out = self.track_start_in[idx].detach().cpu().numpy(), self.track_start_out[idx].detach().cpu().numpy()
        track_in, track_out = self.track_in[idx].detach().cpu().numpy(), self.track_out[idx].detach().cpu().numpy()

        axs[0].plot(
            [
                track_start_in[0] + ((zin.max() - track_start_in[2]) * track_in[0] / track_in[2]),
                track_start_in[0] + ((zout.min() - track_start_in[2]) * track_in[0] / track_in[2]),
            ],
            [zin.max(), zout.min()],
        )
        axs[0].plot(
            [
                track_start_out[0] + ((zin.max() - track_start_out[2]) * track_out[0] / track_out[2]),
                track_start_out[0] + ((zout.min() - track_start_out[2]) * track_out[0] / track_out[2]),
            ],
            [zin.max(), zout.min()],
        )
        axs[0].scatter(xin, zin)
        axs[0].scatter(xout, zout)
        axs[0].scatter(scatter[0], scatter[2], label=r"$\Delta\theta_x=" + f"{dtheta_xy[0]:.1e}$")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("z")
        axs[0].legend()
        axs[1].plot(
            [
                track_start_in[1] + ((zin.max() - track_start_in[2]) * track_in[1] / track_in[2]),
                track_start_in[1] + ((zout.min() - track_start_in[2]) * track_in[1] / track_in[2]),
            ],
            [zin.max(), zout.min()],
        )
        axs[1].plot(
            [
                track_start_out[1] + ((zin.max() - track_start_out[2]) * track_out[1] / track_out[2]),
                track_start_out[1] + ((zout.min() - track_start_out[2]) * track_out[1] / track_out[2]),
            ],
            [zin.max(), zout.min()],
        )
        axs[1].scatter(yin, zin)
        axs[1].scatter(yout, zout)
        axs[1].scatter(scatter[1], scatter[2], label=r"$\Delta\theta_y=" + f"{dtheta_xy[1]:.1e}$")
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
    def __init__(self, mu: MuonBatch, volume: Volume):
        self.mu, self.volume = mu, volume
        self.device = self.mu.device
        self._hits = self.mu.get_hits((0, 0), self.volume.lw)
        self.compute_scatters()

    @staticmethod
    def _get_hit_uncs(dets: List[AbsDetectorLayer], hits: Tensor) -> Tensor:
        uncs = []
        for i, (l, h) in enumerate(zip(dets, hits.unbind(1))):
            if not isinstance(l, VoxelDetectorLayer):
                raise ValueError(f"Detector {l} is not a VoxelDetectorLayer")
            x = l.abs2idx(h)
            r = 1 / l.resolution[x[:, 0], x[:, 1]]
            uncs.append(torch.stack([r, r, torch.zeros_like(r, device=r.device)], dim=-1))
        return torch.stack(uncs, dim=1)  # muons, panels, unc xyz

    def compute_tracks(self) -> None:
        self._hit_uncs = self._get_hit_uncs(self.volume.get_detectors(), self.reco_hits)
        self._track_in, self._track_start_in = self.get_muon_trajectory(self.above_hits, self.above_hit_uncs, self.volume.lw)
        self._track_out, self._track_start_out = self.get_muon_trajectory(self.below_hits, self.below_hit_uncs, self.volume.lw)


class PanelScatterBatch(AbsScatterBatch):
    @staticmethod
    def _get_hit_uncs(zordered_panels: List[DetectorPanel], hits: Tensor) -> Tensor:
        uncs: List[Tensor] = []
        for l, h in zip(zordered_panels, hits.unbind(1)):
            xy = h[:, :2]
            r = 1 / (l.get_resolution(xy) * l.get_efficiency(xy, as_2d=True))
            uncs.append(torch.cat([r, torch.zeros((len(r), 1), device=r.device)], dim=-1))
        return torch.stack(uncs, dim=1)  # muons, panels, unc xyz

    def compute_tracks(self) -> None:
        def _get_panels() -> List[DetectorPanel]:
            panels = []
            for det in self.volume.get_detectors():
                if not isinstance(det, PanelDetectorLayer):
                    raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
                panels += [det.panels[j] for j in det.get_panel_zorder()]
            return panels

        self._hit_uncs = self._get_hit_uncs(_get_panels(), self.gen_hits)
        self._track_in, self._track_start_in = self.get_muon_trajectory(self.above_hits, self.above_hit_uncs, self.volume.lw)
        self._track_out, self._track_start_out = self.get_muon_trajectory(self.below_hits, self.below_hit_uncs, self.volume.lw)


class GenScatterBatch(AbsScatterBatch):
    def compute_tracks(self) -> None:
        self._hit_uncs = torch.ones_like(self._gen_hits)
        self._track_in, self._track_start_in = self.get_muon_trajectory(self.above_gen_hits, self.above_hit_uncs, self.volume.lw)
        self._track_out, self._track_start_out = self.get_muon_trajectory(self.below_gen_hits, self.below_hit_uncs, self.volume.lw)

    @staticmethod
    def _compute_unc(var: Tensor, hits: List[Tensor], hit_uncs: List[Tensor]) -> Tensor:
        return var.new_zeros(var.shape)
