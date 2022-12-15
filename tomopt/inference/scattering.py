from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor

from ..muon import MuonBatch
from ..volume import Volume
from ..utils import jacobian

r"""
Provides implementations of inference algorithms designed to extract variables related to muon scattering from the hits recorded by the detectors
"""

__all__ = ["ScatterBatch", "GenScatterBatch"]


class ScatterBatch:
    r"""
    Class for computing scattering information from the hits via incoming/outgoing trajectory fitting.

    Linear fits are performed separately to all hits associated with layer groups, as indicated by the `pos` attribute of the layers which recorded hits.
    Currently, the inference methods expect detectors above the passive layer to have `pos=='above'`,
    and those below the passive volume to have `pos=='below'`.
    Trajectory fitting is performed using an analytic likelihood minimisation, which considers uncertainties and efficiencies on the hits in x and y.

    .. important::
        The current separation of hits into above and below groups does not allow for e.g. a third set of detectors,
        since this split is based on the value of the `n_hits_above` attribute.

    One instance of this class should created for each :class:`~tomopt.muon.muon_batch.MuonBatch`.
    As part of the initialisation, muons will be filtered using :meth:`~tomopt.inference.ScatterBatch._filter_scatters`
    in order to avoid NaN/Inf gradients or values. This results in direct, in-place changes to the :class:`~tomopt.muon.muon_batch.MuonBatch`.

    Since many variables of the scattering can be inferred, but not all are required for further inference downstream,
    variables, and their uncertainties, are computed on a lazy basis, with memoisation: the values are only computed on the first request (if at all)
    and then stored in case of further requests.

    The dtheta, dphi, and total scattering variables are computed under the assumption of small angular scatterings.
    An assumption is necessary here, since there is a loss of information in the when the muons undergo scattering in theta and phi:
    since theta is [0,pi] a negative scattering in theta will always results in a positive theta, but phi can become phi+pi.
    When inferring the angular scattering, one cannot precisely tell whether instead a large scattering in phi occurred.
    The total scattering (`total_scatter`) is the quadrature sum of dtheta and dphi, and all three are computed under both hypotheses.
    The final values of these are chosen using the hypothesis which minimises the total amount of scattering.
    This assumption has been tested and found to be good.

    Arguments:
        mu: muons with hits to infer on
        volume: volume through which the muons travelled
    """

    # Hits
    _reco_hits: Optional[Tensor] = None  # (muons,hits,xyz) recorded hits with finite xy resolution
    _gen_hits: Optional[Tensor] = None  # (muons,hits,xyz) true positions of the muons
    _hit_uncs: Optional[Tensor] = None  # (muons,hits,xyz) uncertainty on the hits due to resolution
    _hit_effs: Optional[Tensor] = None  # (muons,hits,eff) efficiencies of the hits

    # Tracks
    _track_in: Optional[Tensor] = None  # (muons,xyz) incoming xyz vector
    _track_out: Optional[Tensor] = None  # (muons,xyz) outgoing xyz vector
    _track_start_in: Optional[Tensor] = None  # (muons,xyz) xyz location of initial point along incoming vector
    _track_start_out: Optional[Tensor] = None  # (muons,xyz) xyz location of initial point along outgoing vector
    _cross_track: Optional[Tensor] = None  # (muons,xyz) vector normal to both incoming and outgoing vectors
    _track_coefs: Optional[Tensor] = None  # (muons,xyz) distances along incoming, cross, and outgoing vectors to intersection points

    # Inferred variables
    _poca_xyz: Optional[Tensor] = None  # (muons,xyz) xyz location of PoCA
    _poca_xyz_unc: Optional[Tensor] = None
    _theta_in: Optional[Tensor] = None  # (muons,1) theta of incoming muons
    _theta_in_unc: Optional[Tensor] = None
    _theta_out: Optional[Tensor] = None  # (muons,1) theta of outgoing muons
    _theta_out_unc: Optional[Tensor] = None
    _dtheta: Optional[Tensor] = None  # (muons,1) delta theta between incoming & outgoing muons
    _dtheta_unc: Optional[Tensor] = None
    _phi_in: Optional[Tensor] = None  # (muons,1) phi of incoming muons
    _phi_in_unc: Optional[Tensor] = None
    _phi_out: Optional[Tensor] = None  # (muons,1) phi of outgoing muons
    _phi_out_unc: Optional[Tensor] = None
    _dphi: Optional[Tensor] = None  # (muons,1) delta phi between incoming & outgoing muons
    _dphi_unc: Optional[Tensor] = None
    _total_scatter: Optional[Tensor] = None  # (muons,1) quadrature sum of dtheta and dphi
    _total_scatter_unc: Optional[Tensor] = None
    _theta_xy_in: Optional[Tensor] = None  # (muons,xy) decomposed theta and phi of incoming muons in the zx and zy planes
    _theta_xy_in_unc: Optional[Tensor] = None
    _theta_xy_out: Optional[Tensor] = None  # (muons,xy) decomposed theta and phi of outgoing muons in the zx and zy planes
    _theta_xy_out_unc: Optional[Tensor] = None
    _dtheta_xy: Optional[Tensor] = None  # (muons,xy) delta theta_xy between incoming & outgoing muons in the zx and zy planes
    _dtheta_xy_unc: Optional[Tensor] = None
    _xyz_in: Optional[Tensor] = None  # (muons,xyz) inferred xy position of muon at the z-level of the top of the passive volume
    _xyz_in_unc: Optional[Tensor] = None
    _xyz_out: Optional[Tensor] = None  # (muons,xyz) inferred xy position of muon at the z-level of the bottom of the passive volume
    _xyz_out_unc: Optional[Tensor] = None
    _dxy: Optional[Tensor] = None  # (muons,xy) distances in x & y from PoCA to incoming|outgoing muons
    _dxy_unc: Optional[Tensor] = None
    _theta_msc: Optional[
        Tensor
    ] = None  # (muons,1) angle computed via the cosine dot-product rule between the incoming & outgoing muons; better to use _total_scatter
    _theta_msc_unc: Optional[Tensor] = None

    def __init__(self, mu: MuonBatch, volume: Volume):
        r"""
        Initialise scatter batch from a muon batch.
        During initialisation:
            The muons will be filtered in-place via :meth:`~tomopt.inference.ScatterBatch._filter_scatters`
            The trajectories for the incoming and outgoing muons will be fitted.
        """

        self.mu, self.volume = mu, volume
        self.device = self.mu.device
        self._hits = self.mu.get_hits()
        self._compute_scatters()

    def __len__(self) -> int:
        return len(self.mu)

    @staticmethod
    def get_muon_trajectory(hits: Tensor, uncs: Tensor, lw: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Fits a linear trajectory to a group of hits, whilst considering their uncertainties on their xy positions.
        No uncertainty is considered for z positions of hits.
        The fit is performed via an analytical likelihood-maximisation.


        .. important::
            Muons with <2 hits have NaN trajectory

        Arguments:
            hits: (muons,hits,xyz) tensor of hit positions
            uncs: (muons,hits,(unc x,unc y,0)) tensor of hit uncertainties
            lw: length and width of the passive layers of the volume

        Returns:
            vec: (muons,xyz) fitted-vector directions
            start: (muons,xyz) initial point of fitted-vector
        """

        hits = torch.where(torch.isinf(hits), lw.mean().type(hits.type()) / 2, hits)
        uncs = torch.nan_to_num(uncs)  # Set Infs to large number

        stars, angles = [], []
        for i in range(2):  # separate x and y resolutions
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

    def get_scatter_mask(self) -> Tensor:
        r"""
        Returns:
            (muons) Boolean tensor where True indicates that the PoCA of the muon is located within the passive volume
        """

        z = self.volume.get_passive_z_range()
        return (
            (self.poca_xyz[:, 0] >= 0)
            * (self.poca_xyz[:, 0] < self.volume.lw[0])
            * (self.poca_xyz[:, 1] >= 0)
            * (self.poca_xyz[:, 1] < self.volume.lw[1])
            * (self.poca_xyz[:, 2] >= z[0])
            * (self.poca_xyz[:, 2] < z[1])
        )

    def plot_scatter(self, idx: int) -> None:
        r"""
        Plots representation of hits and fitted trajectories for a single muon.

        Arguments:
            idx: index of muon to plot
        """

        xin, xout = self.hits["above"]["reco_xyz"][idx, :, 0].detach().cpu().numpy(), self.hits["below"]["reco_xyz"][idx, :, 0].detach().cpu().numpy()
        yin, yout = self.hits["above"]["reco_xyz"][idx, :, 1].detach().cpu().numpy(), self.hits["below"]["reco_xyz"][idx, :, 1].detach().cpu().numpy()
        zin, zout = self.hits["above"]["reco_xyz"][idx, :, 2].detach().cpu().numpy(), self.hits["below"]["reco_xyz"][idx, :, 2].detach().cpu().numpy()
        scatter = self.poca_xyz[idx].detach().cpu().numpy()
        dtheta_xy = self.dtheta_xy[idx].detach().cpu().numpy()
        dphi = self.dphi[idx].detach().cpu().numpy()
        phi_in = self.phi_in[idx].detach().cpu().numpy()
        phi_out = self.phi_out[idx].detach().cpu().numpy()
        theta_xy_in = self.theta_xy_in[idx].detach().cpu().numpy()
        theta_xy_out = self.theta_xy_out[idx].detach().cpu().numpy()
        track_start_in, track_start_out = self.track_start_in[idx].detach().cpu().numpy(), self.track_start_out[idx].detach().cpu().numpy()
        track_in, track_out = self.track_in[idx].detach().cpu().numpy(), self.track_out[idx].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].plot(
            [
                track_start_in[0] + ((zin.max() - track_start_in[2]) * track_in[0] / track_in[2]),
                track_start_in[0] + ((zout.min() - track_start_in[2]) * track_in[0] / track_in[2]),
            ],
            [zin.max(), zout.min()],
            label=r"$\theta_{x,in}=" + f"{theta_xy_in[0]:.2}$",
        )
        axs[0].plot(
            [
                track_start_out[0] + ((zin.max() - track_start_out[2]) * track_out[0] / track_out[2]),
                track_start_out[0] + ((zout.min() - track_start_out[2]) * track_out[0] / track_out[2]),
            ],
            [zin.max(), zout.min()],
            label=r"$\theta_{x,out}=" + f"{theta_xy_out[0]:.2}$",
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
            label=r"$\theta_{y,in}=" + f"{theta_xy_in[1]:.2}$",
        )
        axs[1].plot(
            [
                track_start_out[1] + ((zin.max() - track_start_out[2]) * track_out[1] / track_out[2]),
                track_start_out[1] + ((zout.min() - track_start_out[2]) * track_out[1] / track_out[2]),
            ],
            [zin.max(), zout.min()],
            label=r"$\theta_{y,out}=" + f"{theta_xy_out[1]:.2}$",
        )
        axs[1].scatter(yin, zin)
        axs[1].scatter(yout, zout)
        axs[1].scatter(scatter[1], scatter[2], label=r"$\Delta\theta_y=" + f"{dtheta_xy[1]:.1e}$")
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("z")
        axs[1].legend()

        axs[2].plot(
            [
                track_start_in[0],
                track_start_in[0] + np.cos(phi_in),
            ],
            [track_start_in[1], track_start_in[1] + np.sin(phi_in)],
            label=r"$\phi_{in}=" + f"{phi_in[0]:.3}$",
        )
        axs[2].plot(
            [
                track_start_out[0],
                track_start_out[0] - np.cos(phi_out),
            ],
            [track_start_out[1], track_start_out[1] - np.sin(phi_out)],
            label=r"$\phi_{out}=" + f"{phi_out[0]:.3}$",
        )
        axs[2].scatter(xin, yin)
        axs[2].scatter(xout, yout)
        axs[2].scatter(scatter[0], scatter[1], label=r"$\Delta\phi=" + f"{dphi[0]:.1e}$")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        axs[2].legend()
        plt.show()

    @staticmethod
    def _compute_theta_msc(p: Tensor, q: Tensor) -> Tensor:
        r"""
        Computes the angle between sets of vectors p and q via the cosine dot-product.

        .. warning::
            This angle is NOT the total amount of scattering; please use `total_scatter`.
            This code is kept in case the angle is still useful for inference.

        Arguments:
            p: (N,xyz) vectors 1
            q: (N,xyz) vectors 2

        Returns:
           (N,1) angles between vectors
        """

        return torch.arccos((p * q).sum(-1, keepdim=True) / (p.norm(dim=-1, keepdim=True) * q.norm(dim=-1, keepdim=True)))

    @staticmethod
    def _compute_theta(track: Tensor) -> Tensor:
        r"""
        Computes the theta angles of vectors

        Arguments:
            track: (N,xyz) components of the vectors

        Returns:
            (N,1) theta angles of the vectors
        """

        arg = -track[:, 2:3] / track.norm(dim=-1, keepdim=True)
        theta = arg.new_zeros(arg.shape)

        m = arg != 1
        theta[m] = torch.arccos(arg[m])

        return theta

    @staticmethod
    def _compute_phi(x: Tensor, y: Tensor) -> Tensor:
        r"""
        Computes the phi angles from the xy components of vectors

        Arguments:
            x: (N,1) x components of the vectors
            y: (N,1) y components of the vectors

        Returns:
            (N,1) phi angles of the vectors
        """

        phi = y.new_zeros(x.shape)

        # Case when x == 0
        phi[(x == 0) * (y > 0)] = torch.pi / 2
        phi[(x == 0) * (y < 0)] = 3 * torch.pi / 2

        m = x != 0
        phi[m] = torch.arctan(y[m] / x[m])
        # # Account for quadrants
        m = x < 0
        phi[m] = phi[m] + torch.pi
        m = (x > 0) * (y < 0)
        phi[m] = phi[m] + (2 * torch.pi)  # (0, 2pi)

        return phi

    @staticmethod
    def _compute_dtheta_dphi_scatter(theta_in: Tensor, phi_in: Tensor, theta_out: Tensor, phi_out: Tensor) -> Dict[str, Tensor]:
        r"""
        Computes dtheta, dphi, and total scattering variables under the assumption of small angular scatterings.
        An assumption is necessary here, since there is a loss of information in the when the muons undergo scattering in theta and phi:
        since theta is [0,pi] a negative scattering in theta will always results in a positive theta, but phi can become phi+pi.
        When inferring the angular scattering, one cannot precisely tell whether instead a large scattering in phi occurred.
        The total scattering (`total_scatter`) is the quadrature sum of dtheta and dphi, and all three are computed under both hypotheses.
        The final values of these are chosen using the hypothesis which minimises the total amount of scattering.
        This assumption has been tested and found to be good.

        Arguments:
            theta_in: (N,1) theta angle of incoming muons
            phi_in: (N,1) phi angle of incoming muons
            theta_out: (N,1) theta angle of outgoing muons
            phi_out: (N,1) phi angle of outgoing muons

        Returns:
            Dictionary of (N,1) tensors for "dtheta", "dphi", & "total_scatter"
        """

        theta_in = theta_in.squeeze(-1)
        phi_in = phi_in.squeeze(-1)
        theta_out = theta_out.squeeze(-1)
        phi_out = phi_out.squeeze(-1)

        dtheta = torch.stack([(theta_in - theta_out).abs(), theta_in + theta_out], dim=-1)
        dphi = torch.min(
            torch.stack(
                (
                    ((2 * torch.pi) - phi_in) + phi_out,
                    ((2 * torch.pi) - phi_out) + phi_in,
                    torch.abs(phi_in - phi_out),
                ),
                dim=0,
            ),
            dim=0,
        ).values
        dphi = torch.stack([dphi, torch.pi - dphi], dim=-1)
        total_scatter = (dtheta.square() + dphi.square()).sqrt()
        # Pick set with smallest scattering
        hypo = total_scatter.argmin(-1)
        i = np.arange(len(total_scatter))
        return {"dtheta": dtheta[i, hypo, None], "dphi": dphi[i, hypo, None], "total_scatter": total_scatter[i, hypo, None]}

    def _extract_hits(self) -> None:
        r"""
        Takes the dictionary of hits from the muons and combines them into single tensors of recorded and true hits.
        """

        self._n_hits_above = self.hits["above"]["reco_xyz"].shape[1]
        self._n_hits_below = self.hits["below"]["reco_xyz"].shape[1]

        # Combine all input vars into single tensor, NB ideally would stack to new dim but can't assume same number of panels above & below
        self._reco_hits = torch.cat((self.hits["above"]["reco_xyz"], self.hits["below"]["reco_xyz"]), dim=1)  # muons, all panels, reco xyz
        self._gen_hits = torch.cat((self.hits["above"]["gen_xyz"], self.hits["below"]["gen_xyz"]), dim=1)  # muons, all panels, true xyz
        self._hit_uncs = torch.cat((self.hits["above"]["unc_xyz"], self.hits["below"]["unc_xyz"]), dim=1)  # muons, all panels, xyz unc
        self._hit_effs = torch.cat((self.hits["above"]["eff"], self.hits["below"]["eff"]), dim=1)  # muons, all panels, eff

    def _compute_tracks(self) -> None:
        r"""
        Computes tracks from hits according to the uncertainty and efficiency of the hits, computed as 1/(resolution*efficiency).
        """

        self._track_in, self._track_start_in = self.get_muon_trajectory(self.above_hits, self.above_hit_uncs / self.above_hit_effs, self.volume.lw)
        self._track_out, self._track_start_out = self.get_muon_trajectory(self.below_hits, self.below_hit_uncs / self.below_hit_effs, self.volume.lw)

    def _filter_scatters(self) -> None:
        r"""
        Filters muons to avoid NaN/Inf gradients or values. This results in direct, in-place changes to the :class:`~tomopt.muon.muon_batch.MuonBatch`.
        This might seem heavy-handed, but tracks with invalid/extreme parameters can have NaN gradients, which can spoil the grads of all other muons.

        .. important::
            This method will remove any muon with at least one high-uncertainty hit, even if all the others are ok.
            This can mean that for some detector configurations with, e.g. one tiny detector, it might be best to manually remove the unneeded detector
            in order to gain an increase in the number of muons available for inference.

        The removal criteria are any of:
            total scattering is zero, NaN, or Inf
            incoming or outgoing tracks are parallel to the passive volume
            incoming or outgoing tracks are located far from the passive volume at the z-levels of its top or bottom
            Any hits associated with a muon are >= 1e10
        """

        # Only include muons that scatter
        theta_in = self._compute_theta(self.track_in)
        theta_out = self._compute_theta(self.track_out)
        phi_in = self._compute_phi(x=self.track_in[:, 0:1], y=self.track_in[:, 1:2])
        phi_out = self._compute_phi(x=self.track_out[:, 0:1], y=self.track_out[:, 1:2])

        total_scatter = self._compute_dtheta_dphi_scatter(theta_in=theta_in, phi_in=phi_in, theta_out=theta_out, phi_out=phi_out)["total_scatter"]
        keep_mask = (total_scatter != 0) * (~total_scatter.isnan()) * (~total_scatter.isinf())

        theta_msc = self._compute_theta_msc(self.track_in, self.track_out)
        keep_mask *= (theta_msc != 0) * (~theta_msc.isnan()) * (~theta_msc.isinf())

        # Remove muons with tracks parallel to volume
        keep_mask *= (theta_in - (torch.pi / 2) < -1e-5) * (theta_out - (torch.pi / 2) < -1e-5)

        # Remove muons with tracks entering or exiting far from the volume
        xy_in = self._compute_xyz_in()[:, :2]
        xy_out = self._compute_xyz_out()[:, :2]
        keep_mask *= (
            (xy_in > -10 * self.volume.lw).prod(-1)
            * (xy_in < 10 * self.volume.lw).prod(-1)
            * (xy_out > -10 * self.volume.lw).prod(-1)
            * (xy_out < 10 * self.volume.lw).prod(-1)
        )[:, None].bool()

        # Remove muons with high uncertainties; yes they get ignored during track fitting, but they can cause NaNs in the gradient
        keep_mask *= (self.hit_uncs[:, :, :2] < 1e10).all(-1).all(-1, keepdim=True)

        keep_mask.squeeze_()
        if not keep_mask.all():  # Recompute tracks and hits
            self.mu.filter_muons(keep_mask)
            self._hits = self.mu.get_hits()
            self._extract_hits()
            self._compute_tracks()

    def _compute_scatters(self) -> None:
        r"""
        Computes incoming and outgoing vectors, and the vector normal to them, from hits extracted from filtered muons.

        .. important::
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

        self._extract_hits()
        self._compute_tracks()
        self._filter_scatters()

        # Track computations
        self._cross_track = torch.cross(self.track_in, self.track_out, dim=1)  # connecting vector perpendicular to both lines

        rhs = self.track_start_out - self.track_start_in
        lhs = torch.stack([self.track_in, -self.track_out, self._cross_track], dim=1).transpose(2, 1)
        # coefs = torch.linalg.solve(lhs, rhs)  # solve p1+t1*v1 + t3*v3 = p2+t2*v2 => p2-p1 = t1*v1 - t2*v2 + t3*v3
        self._track_coefs = (lhs.inverse() @ rhs[:, :, None]).squeeze(-1)

    def _compute_xyz_in(self) -> Tensor:
        r"""
        Returns:
            (muon,xyz) tensor the positions of the muons at the z-level of the top of the passive volume
        """

        dz = self.volume.get_passive_z_range()[1] - self._track_start_in[:, 2:3]  # last panel to volume start
        return self._track_start_in + ((dz / self._track_in[:, 2:3]) * self._track_in)

    def _compute_xyz_out(self) -> Tensor:
        r"""
        Returns:
            (muon,xyz) tensor the positions of the muons at the z-level of the bottom of the passive volume
        """

        dz = self._track_start_out[:, 2:3] - (self.volume.get_passive_z_range()[0])  # volume end to first panel
        return self._track_start_out - ((dz / self._track_out[:, 2:3]) * self._track_out)

    def _compute_out_var_unc(self, var: Tensor) -> Tensor:
        r"""
        Computes the uncertainty on variable computed from the recorded hits due to the uncertainties on the hits, via gradient-based error propagation.
        This computation uses the triangle of the error matrix and does not assume zero-valued off-diagonal elements.

        .. warning::
            Behaviour tested only

        Arguments:
            var: (muons,*) tensor of variables computed from the recorded hits

        Returns:
            (muons,*) tensor of uncertainties on var
        """

        # Compute dvar/dhits
        jac = torch.nan_to_num(jacobian(var, self.reco_hits)).sum(2)
        unc = torch.where(torch.isinf(self.hit_uncs), torch.tensor([0], device=self.device).type(self.hit_uncs.type()), self.hit_uncs)[:, None]
        jac, unc = jac.reshape(jac.shape[0], jac.shape[1], -1), unc.reshape(unc.shape[0], unc.shape[1], -1)

        # Compute unc^2 = unc_x*unc_y*dvar/dhit_x*dvar/dhit_y summing over all x,y inclusive combinations
        idxs = torch.combinations(torch.arange(0, unc.shape[-1]), with_replacement=True)
        unc_2 = (jac[:, :, idxs] * unc[:, :, idxs]).prod(-1)
        return unc_2.sum(-1).sqrt()

    def _set_dtheta_dphi_scatter(self) -> None:
        r"""
        Simultaneously sets dtheta, dphi, and total scattering variables under the assumption of small angular scatterings.
        An assumption is necessary here, since there is a loss of information in the when the muons undergo scattering in theta and phi:
        since theta is [0,pi] a negative scattering in theta will always results in a positive theta, but phi can become phi+pi.
        When inferring the angular scattering, one cannot precisely tell whether instead a large scattering in phi occurred.
        The total scattering is computed as the (only) angle between incoming and outgoing tracks. Computation is done by the _compute_theta_msc() method.
        The final values of these are chosen using the hypothesis which minimises the total amount of scattering.
        This assumption has been tested and found to be good.
        """

        values = self._compute_dtheta_dphi_scatter(theta_in=self.theta_in, phi_in=self.phi_in, theta_out=self.theta_out, phi_out=self.phi_out)
        self._dtheta, self._dphi, self._total_scatter = values["dtheta"], values["dphi"], self._compute_theta_msc(self.track_in, self.track_out)
        self._dtheta_unc, self._dphi_unc, self._total_scatter_unc = None, None, None

    @property
    def hits(self) -> Dict[str, Dict[str, Tensor]]:
        r"""
        Returns:
            Dictionary of hits, as returned by :meth:`~tomopt.muon.muon_batch.MuonBatch.get_hits()`
        """

        return self._hits

    @property
    def reco_hits(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of recorded hits
        """

        return self._reco_hits

    @property
    def gen_hits(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of true hits
        """

        return self._gen_hits

    @property
    def n_hits_above(self) -> Optional[int]:
        r"""
        Returns:
            Number of hits per muon in the "above" detectors
        """

        return self._n_hits_above

    @property
    def n_hits_below(self) -> Optional[int]:
        r"""
        Returns:
            Number of hits per muon in the "below" detectors
        """

        return self._n_hits_below

    @property
    def above_gen_hits(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of true hits in the "above" detectors
        """

        if self._gen_hits is None:
            return None
        else:
            return self._gen_hits[:, : self.n_hits_above]

    @property
    def below_gen_hits(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of true hits in the "below" detectors
        """

        if self._gen_hits is None:
            return None
        else:
            return self._gen_hits[:, self.n_hits_above :]

    @property
    def above_hits(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of recorded hits in the "above" detectors
        """

        if self._reco_hits is None:
            return None
        else:
            return self._reco_hits[:, : self.n_hits_above]

    @property
    def below_hits(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of recorded hits in the "below" detectors
        """

        if self._reco_hits is None:
            return None
        else:
            return self._reco_hits[:, self.n_hits_above :]

    @property
    def hit_uncs(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of uncertainties on hits
        """

        return self._hit_uncs

    @property
    def hit_effs(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,eff) tensor of hit efficiencies
        """

        return self._hit_effs

    @property
    def above_hit_uncs(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of uncertainties on hits in the "above" detectors
        """

        if self._hit_uncs is None:
            return None
        else:
            return self._hit_uncs[:, : self.n_hits_above]

    @property
    def below_hit_uncs(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,xyz) tensor of uncertainties on hits in the "below" detectors
        """

        if self._hit_uncs is None:
            return None
        else:
            return self._hit_uncs[:, self.n_hits_above :]

    @property
    def above_hit_effs(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,effs) tensor of hit efficiencies in the "above" detectors
        """

        if self._hit_effs is None:
            return None
        else:
            return self._hit_effs[:, : self.n_hits_above]

    @property
    def below_hit_effs(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,hits,eff) tensor of hit efficiencies in the "below" detectors
        """

        if self._hit_effs is None:
            return None
        else:
            return self._hit_effs[:, self.n_hits_above :]

    @property
    def track_in(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,xyz) incoming xyz vector
        """

        return self._track_in

    @property
    def track_start_in(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,xyz) initial point of incoming xyz vector
        """

        return self._track_start_in

    @property
    def track_out(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,xyz) outgoing xyz vector
        """

        return self._track_out

    @property
    def track_start_out(self) -> Optional[Tensor]:
        r"""
        Returns:
            (muons,xyz) initial point of outgoing xyz vector
        """

        return self._track_start_out

    @property
    def poca_xyz(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) xyz location of PoCA
        """

        if self._poca_xyz is None:
            q1 = self.track_start_in + (self._track_coefs[:, 0:1] * self.track_in)  # closest point on v1
            self._poca_xyz = q1 + (self._track_coefs[:, 2:3] * self._cross_track / 2)  # Move halfway along v3 from q1
            self._poca_xyz_unc = None
        return self._poca_xyz

    @property
    def poca_xyz_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) uncertainty on poca_xyz
        """

        if self._poca_xyz_unc is None:
            self._poca_xyz_unc = self._compute_out_var_unc(self.poca_xyz)
        return self._poca_xyz_unc

    @property
    def dxy(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) distances in x & y from PoCA to incoming|outgoing muons
        """

        if self._dxy is None:
            self._dxy = self._track_coefs[:, 2:3] * self._cross_track[:, :2]
            self._dxy_unc = None
        return self._dxy

    @property
    def dxy_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) uncertainty on dxy
        """

        if self._dxy_unc is None:
            self._dxy_unc = self._compute_out_var_unc(self.dxy)
        return self._dxy_unc

    @property
    def theta_in(self) -> Tensor:
        r"""
        Returns:
            (muons,1) theta of incoming muons
        """

        if self._theta_in is None:
            self._theta_in = self._compute_theta(self.track_in)
            self._theta_in_unc = None
        return self._theta_in

    @property
    def theta_in_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on theta_in
        """

        if self._theta_in_unc is None:
            self._theta_in_unc = self._compute_out_var_unc(self.theta_in)
        return self._theta_in_unc

    @property
    def theta_out(self) -> Tensor:
        r"""
        Returns:
            (muons,1) theta of outgoing muons
        """

        if self._theta_out is None:
            self._theta_out = self._compute_theta(self.track_out)
            self._theta_out_unc = None
        return self._theta_out

    @property
    def theta_out_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on theta_out
        """

        if self._theta_out_unc is None:
            self._theta_out_unc = self._compute_out_var_unc(self.theta_out)
        return self._theta_out_unc

    @property
    def phi_in(self) -> Tensor:
        r"""
        Returns:
            (muons,1) phi of incoming muons
        """

        if self._phi_in is None:
            self._phi_in = self._compute_phi(x=self.track_in[:, 0:1], y=self.track_in[:, 1:2])
            self._phi_in_unc = None
        return self._phi_in

    @property
    def phi_in_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on phi_in
        """

        if self._phi_in_unc is None:
            self._phi_in_unc = self._compute_out_var_unc(self.phi_in)
        return self._phi_in_unc

    @property
    def phi_out(self) -> Tensor:
        r"""
        Returns:
            (muons,1) phi of outgoing muons
        """

        if self._phi_out is None:
            self._phi_out = self._compute_phi(x=self.track_out[:, 0:1], y=self.track_out[:, 1:2])
            self._phi_out_unc = None
        return self._phi_out

    @property
    def phi_out_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on phi_out
        """

        if self._phi_out_unc is None:
            self._phi_out_unc = self._compute_out_var_unc(self.phi_out)
        return self._phi_out_unc

    @property
    def dtheta(self) -> Tensor:
        r"""
        Returns:
            (muons,1) delta theta between incoming & outgoing muons
        """

        if self._dtheta is None:
            self._set_dtheta_dphi_scatter()
        return self._dtheta

    @property
    def dtheta_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on dtheta
        """

        if self._dtheta_unc is None:
            self._dtheta_unc = self._compute_out_var_unc(self.dtheta)
        return self._dtheta_unc

    @property
    def dphi(self) -> Tensor:
        r"""
        Returns:
            (muons,1) delta phi between incoming & outgoing muons
        """

        if self._dphi is None:
            self._set_dtheta_dphi_scatter()
        return self._dphi

    @property
    def dphi_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on dphi
        """

        if self._dphi_unc is None:
            self._dphi_unc = self._compute_out_var_unc(self.dphi)
        return self._dphi_unc

    @property
    def total_scatter(self) -> Tensor:
        r"""
        Returns:
            (muons,1) quadrature sum of dtheta and dphi; the total amount of angular scattering fot phi_in != phi_out != 0
        """

        if self._total_scatter is None:
            self._set_dtheta_dphi_scatter()
        return self._total_scatter

    @property
    def total_scatter_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on total_scatter
        """

        if self._total_scatter_unc is None:
            self._total_scatter_unc = self._compute_out_var_unc(self.total_scatter)
        return self._total_scatter_unc

    @property
    def theta_xy_in(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) decomposed theta and phi of incoming muons in the zx and zy planes
        """

        if self._theta_xy_in is None:
            self._theta_xy_in = torch.cat([(self.theta_in.tan() * self.phi_in.cos()).arctan(), (self.theta_in.tan() * self.phi_in.sin()).arctan()], dim=-1)
            self._theta_xy_in_unc = None
        return self._theta_xy_in

    @property
    def theta_xy_in_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) uncertainty on theta_xy_in
        """

        if self._theta_xy_in_unc is None:
            self._theta_xy_in_unc = self._compute_out_var_unc(self.theta_xy_in)
        return self._theta_xy_in_unc

    @property
    def theta_xy_out(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) decomposed theta and phi of outgoing muons in the zx and zy planes
        """

        if self._theta_xy_out is None:
            self._theta_xy_out = torch.cat([(self.theta_out.tan() * self.phi_out.cos()).arctan(), (self.theta_out.tan() * self.phi_out.sin()).arctan()], dim=-1)
            self._theta_xy_out_unc = None
        return self._theta_xy_out

    @property
    def theta_xy_out_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) uncertainty on theta_xy_out
        """

        if self._theta_xy_out_unc is None:
            self._theta_xy_out_unc = self._compute_out_var_unc(self.theta_xy_out)
        return self._theta_xy_out_unc

    @property
    def dtheta_xy(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) delta theta_xy between incoming & outgoing muons in the zx and zy planes
        """

        if self._dtheta_xy is None:
            self._dtheta_xy = torch.abs(self.theta_xy_in - self.theta_xy_out)
            self._dtheta_xy_unc = None
        return self._dtheta_xy

    @property
    def dtheta_xy_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xy) uncertainty on dtheta_xy
        """

        if self._dtheta_xy_unc is None:
            self._dtheta_xy_unc = self._compute_out_var_unc(self.dtheta_xy)
        return self._dtheta_xy_unc

    @property
    def xyz_in(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) inferred xy position of muon at the z-level of the top of the passive volume
        """

        if self._xyz_in is None:
            self._xyz_in = self._compute_xyz_in()
            self._xyz_in_unc = None
        return self._xyz_in

    @property
    def xyz_in_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) uncertainty on xyz_in
        """

        if self._xyz_in_unc is None:
            self._xyz_in_unc = self._compute_out_var_unc(self.xyz_in)
        return self._xyz_in_unc

    @property
    def xyz_out(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) inferred xy position of muon at the z-level of the bottom of the passive volume
        """

        if self._xyz_out is None:
            self._xyz_out = self._compute_xyz_out()
            self._xyz_out_unc = None
        return self._xyz_out

    @property
    def xyz_out_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) uncertainty on xyz_out
        """

        if self._xyz_out_unc is None:
            self._xyz_out_unc = self._compute_out_var_unc(self.xyz_out)
        return self._xyz_out_unc

    @property
    def theta_msc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) angle computed via the cosine dot-product rule between the incoming & outgoing muons; better to use _total_scatter
        """

        if self._theta_msc is None:
            self._theta_msc = self._compute_theta_msc(self.track_in, self.track_out)
            self._theta_msc_unc = None
        return self._theta_msc

    @property
    def theta_msc_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) uncertainty on theta_msc
        """

        if self._theta_msc_unc is None:
            self._theta_msc_unc = self._compute_out_var_unc(self.theta_msc)
        return self._theta_msc_unc


class GenScatterBatch(ScatterBatch):
    r"""
    Class for computing scattering information from the true hits via incoming/outgoing trajectory fitting.

    .. warning::
        This class is intended for diagnostic purposes only.
        The tracks and scatter variables carry no gradient w.r.t. detector parameters (except z position).

    Linear fits are performed separately to all hits associated with layer groups, as indicated by the `pos` attribute of the layers which recorded hits.
    Currently, the inference methods expect detectors above the passive layer to have `pos=='above'`,
    and those below the passive volume to have `pos=='below'`.
    Trajectory fitting is performed using an analytic likelihood minimisation, but no uncertainties on the hits are considered.

    .. important::
        The current separation of hits into above and below groups does not allow for e.g. a third set of detectors,
        since this split is based on the value of the `n_hits_above` attribute.

    One instance of this class should created for each :class:`~tomopt.muon.muon_batch.MuonBatch`.
    As part of the initialisation, muons will be filtered using :meth:`~tomopt.inference.ScatterBatch._filter_scatters`
    in order to avoid NaN/Inf values. This results in direct, in-place changes to the :class:`~tomopt.muon.muon_batch.MuonBatch`.

    Since many variables of the scattering can be inferred, but not all are required for further inference downstream,
    variables, and their uncertainties, are computed on a lazy basis, with memoisation: the values are only computed on the first request (if at all)
    and then stored in case of further requests.

    The dtheta, dphi, and total scattering variables are computed under the assumption of small angular scatterings.
    An assumption is necessary here, since there is a loss of information in the when the muons undergo scattering in theta and phi:
    since theta is [0,pi] a negative scattering in theta will always results in a positive theta, but phi can become phi+pi.
    When inferring the angular scattering, one cannot precisely tell whether instead a large scattering in phi occurred.
    The total scattering (`total_scatter`) is the quadrature sum of dtheta and dphi, and all three are computed under both hypotheses.
    The final values of these are chosen using the hypothesis which minimises the total amount of scattering.
    This assumption has been tested and found to be good.

    Arguments:
        mu: muons with hits to infer on
        volume: volume through which the muons travelled
    """

    def _compute_tracks(self) -> None:
        r"""
        Computes tracks from true muon positions.
        """

        self._hit_uncs = torch.ones_like(self._gen_hits)
        self._track_in, self._track_start_in = self.get_muon_trajectory(self.above_gen_hits, self.above_hit_uncs, self.volume.lw)
        self._track_out, self._track_start_out = self.get_muon_trajectory(self.below_gen_hits, self.below_hit_uncs, self.volume.lw)
