from __future__ import annotations
from typing import Dict, List, Union, Tuple, Optional
import math
from collections import defaultdict
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

from ..core import DEVICE

r"""
Provides container classes for a batch of many muons
"""

__all__ = ["MuonBatch"]


class MuonBatch:
    r"""
    Container class for a batch of many muons, defined by their position and kinematics.

    Each muon has its own:
        - x, y, and z position in metres, which are absolute coordinates in the volume frame.
        - theta, the angle in radians [0,pi) between the muon trajectory and the negative z-axis in the volume frame muons with a theta > pi/2 (i.e. travel upwards) may be removed automatically
        - phi, the anticlockwise angle in radians [0,2pi) between the muon trajectory and the positive x-axis, in the x-y plane of the volume frame.
        - momentum (mom), the absolute value of the muon momentum in GeV

    Muon properties should not be updated manually.  Instead, call:
        - `.propagate_dz_dz(dz)` to update the x,y,z positions of the muons for a given propagation dz in the z-axis.
        - `.propagate_dz_d(d)` to update the x,y,z positions of the muons for a given propagation d in the muons' trajectories.
        - `.scatter_dxy(dx_vol, dy_vol, mask)` to shift the x,y positions of the muons, for which the values of the optional Boolean mask is true, by the specified amount.
        - `.scatter_dtheta_dphi(dtheta_vol, dphi_vol, mask)` to alter the theta,phi angles of the muons, for which the values of the optional Boolean mask is true, by the specified amount.

    .. important::
        Muon momenta is currently constant

    .. important::
        Eventually the muon batch will be extended to store information about the inferred momentum of the muons `reco_mom`.
        However currently the `reco_mom` property will return the TRUE momentum of the muons, with no simulation of measurement precision.

    By default, the `MuonBatch` class only contains the current position of the muons,
    however the `.snapshot_xyz` method can be used to store the xy positions of the muons at any time, to a dictionary with float z-position keys, `xyz_hist`.

    In addition to storing the properties of the muons, the `MuonBatch` class is also used to store the detector hits associated with each muon.
    Hits may be added via the `.append_hits` method, and stored in the `_hits` attribute.
    Hits can then be retrieved by the `.get_hits` method.

    Arguments:
        xy_p_theta_phi: (N_muon, 5) tensor,
            with xy [m], p [GeV], theta [r] (0, pi/2) defined w.r.t z axis, phi [r] (0, 2pi) defined anticlockwise from x axis
        init_z: initial z position of all muons in the batch
        device: device on which to place the muon tensors
    """

    x_dim = 0
    y_dim = 1
    z_dim = 2
    p_dim = 3
    th_dim = 4
    ph_dim = 5
    _keep_mask: Optional[Tensor] = None  # After a scattering, this will be a Boolean mask of muons kept, to help with testing

    def __init__(self, xy_p_theta_phi: Tensor, init_z: Union[Tensor, float], device: torch.device = DEVICE):
        r"""
        Initialises the class from `xy_p_theta_phi`, a (N_muon, 5) tensor, and an initial z position for the batch.
        Muon trajectories (theta & phi) and positions (x,y,z) are in the reference frame of the volume.
        """

        self.device = device
        self._muons = xy_p_theta_phi.to(self.device)

        # Insert z position in tensor
        self._muons = F.pad(self._muons, (1, 0))
        self._muons[:, 0] = self._muons[:, 1]
        self._muons[:, 1] = self._muons[:, 2]
        self._muons[:, 2] = init_z

        self._removed_muons: Optional[Tensor] = None
        self._hits: Dict[str, Dict[str, List[Tensor]]] = defaultdict(lambda: defaultdict(list))
        self._xyz_hist: List[Tensor] = []

    def __repr__(self) -> str:
        return f"Batch of {len(self)} muons"

    def __len__(self) -> int:
        return len(self.muons)

    @staticmethod
    def phi_from_theta_xy(theta_x: Tensor, theta_y: Tensor) -> Tensor:
        r"""
        Computes the phi angle from theta_x and theta_y.

        .. important::
            This function does NOT work if theta is > pi/2

        Arguments:
            theta_x: angle from the negative z-axis in the xz plane
            theta_y: angle from the negative z-axis in the yz plane

        Returns:
            phi, the anti-clockwise angle from the positive x axis, in the xy plane
        """

        phi = torch.arctan(theta_y.tan() / theta_x.tan())  # (-pi/2, pi/2)
        m = theta_x < 0
        phi[m] = phi[m] + torch.pi
        m = ((theta_x >= 0) * (theta_y < 0)).bool()
        phi[m] = phi[m] + (2 * torch.pi)  # (0, 2pi)

        phi[(theta_x.abs() >= torch.pi / 2) + (theta_y.abs() >= torch.pi / 2)] = torch.nan
        return phi

    @staticmethod
    def theta_from_theta_xy(theta_x: Tensor, theta_y: Tensor) -> Tensor:
        r"""
        Computes the theta angle from theta_x and theta_y.

        .. important::
            This function does NOT work if theta is > pi/2

        Arguments:
            theta_x: angle from the negative z-axis in the xz plane
            theta_y: angle from the negative z-axis in the yz plane

        Returns:
            theta, the anti-clockwise angle from the negative z axis, in the xyz plane
        """

        theta = (theta_x.tan().square() + theta_y.tan().square()).sqrt().arctan()
        theta[(theta_x.abs() >= torch.pi / 2) + (theta_y.abs() >= torch.pi / 2)] = torch.nan
        return theta

    @staticmethod
    def theta_x_from_theta_phi(theta: Tensor, phi: Tensor) -> Tensor:
        r"""
        Computes the angle from the negative z-axis in the xz plane from theta and phi

        .. important::
            This function does NOT work if theta is > pi/2

        Arguments:
            theta: the anti-clockwise angle from the negative z axis, in the xyz plane
            phi: the anti-clockwise angle from the positive x axis, in the xy plane

        Returns:
            theta_x, the angle from the negative z-axis in the xz plane
        """

        tx = (theta.tan() * phi.cos()).arctan()
        tx[(theta >= torch.pi / 2)] = torch.nan
        return tx

    @staticmethod
    def theta_y_from_theta_phi(theta: Tensor, phi: Tensor) -> Tensor:
        r"""
        Computes the angle from the negative z-axis in the yz plane from theta and phi

        .. important::
            This function does NOT work if theta is > pi/2

        Arguments:
            theta: the anti-clockwise angle from the negative z axis, in the xyz plane
            phi: the anti-clockwise angle from the positive x axis, in the xy plane

        Returns:
            theta_y, the angle from the negative z-axis in the yz plane
        """

        ty = (theta.tan() * phi.sin()).arctan()
        ty[(theta >= torch.pi / 2)] = torch.nan
        return ty

    def scatter_dxyz(
        self, dx_vol: Optional[Tensor] = None, dy_vol: Optional[Tensor] = None, dz_vol: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> None:
        r"""
        Displaces the muons in xyz by the specified amounts.
        If a mask is supplied, then only muons with True mask elements are displaced.

        Arguments:
            dx_vol: (N,) tensor of displacements in x
            dy_vol: (N,) tensor of displacements in y
            dz_vol: (N,) tensor of displacements in z
            mask: (N,) Boolean tensor. If not None, only muons with True mask elements are displaced.
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()
        if dx_vol is not None:
            self._x[mask] = self._x[mask] + dx_vol
        if dy_vol is not None:
            self._y[mask] = self._y[mask] + dy_vol
        if dz_vol is not None:
            self._z[mask] = self._z[mask] + dz_vol

    def scatter_dtheta_dphi(self, dtheta_vol: Optional[Tensor] = None, dphi_vol: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> None:
        r"""
        Changes the trajectory of the muons in theta-phi by the specified amounts, with no change in their x,y,z positions.
        If a mask is supplied, then only muons with True mask elements are altered.

        Arguments:
            dtheta_vol: (N,) tensor of angular changes in theta
            dphi_vol: (N,) tensor of angular changes in phi
            mask: (N,) Boolean tensor. If not None, only muons with True mask elements are altered.
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()

        if dphi_vol is not None:
            self._phi = (self._phi[mask] + dphi_vol) % (2 * torch.pi)
        if dtheta_vol is not None:
            theta = (self._theta[mask] + dtheta_vol) % (2 * torch.pi)
            # Correct theta, must avoid double Bool mask
            phi = self._phi[mask]
            m = theta > torch.pi
            phi[m] = (phi[m] + torch.pi) % (2 * torch.pi)  # rotate in phi
            theta[m] = (2 * torch.pi) - theta[m]  # theta (0,pi)
            self._phi = phi
            self._theta = theta
        self.remove_upwards_muons()

    def scatter_dtheta_xy(self, dtheta_x_vol: Optional[Tensor] = None, dtheta_y_vol: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> None:
        r"""
        Changes the trajectory of the muons in theta-phi by the specified amounts in dtheta_xy, with no change in their x,y,z positions.
        If a mask is supplied, then only muons with True mask elements are altered.

        Arguments:
            dtheta_x_vol: (N,) tensor of angular changes in theta_x
            dtheta_y_vol: (N,) tensor of angular changes in theta_y
            mask: (N,) Boolean tensor. If not None, only muons with True mask elements are altered.
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()

        theta_x = self.theta_x_from_theta_phi(self.theta[mask], self.phi[mask])
        theta_y = self.theta_y_from_theta_phi(self.theta[mask], self.phi[mask])
        if dtheta_x_vol is not None:
            theta_x = theta_x + dtheta_x_vol
        if dtheta_y_vol is not None:
            theta_y = theta_y + dtheta_y_vol
        self.theta[mask] = self.theta_from_theta_xy(theta_x, theta_y).type(torch.float)
        self.phi[mask] = self.phi_from_theta_xy(theta_x, theta_y).type(torch.float)

        self.remove_upwards_muons()

    def remove_upwards_muons(self) -> None:
        r"""
        Removes muons, and their hits, if their theta >= pi/2, i.e. they are travelling upwards after a large scattering.
        Should be run after any changes to theta, but make sure that references (e.g. masks) to the complete set of muons are no longer required.
        """

        self._keep_mask = (self._theta < torch.pi / 2) & (~self._theta.isnan()) & (~self._phi.isnan())  # To keep
        self.filter_muons(self._keep_mask)

    def filter_muons(self, keep_mask: Tensor) -> None:
        r"""
        Removes all muons, and their associated hits, except for muons specified as True in `keep_mask`.

        Arguments:
            keep_mask: (N,) Boolean tensor. Muons with False elements will be removed, along with their hits.
        """

        if keep_mask.sum() < len(self):
            # Save muons, just in case they're useful for diagnostics
            if self._removed_muons is None:
                self._removed_muons = self._muons[~keep_mask].detach().cpu().numpy()
            else:
                self._removed_muons = np.concatenate((self._removed_muons, self._muons[~keep_mask].detach().cpu().numpy()), axis=0)

            # Remove muons and hits
            self._muons = self._muons[keep_mask]
            for pos in self._hits:  # TODO: Make a HitBatch class to make this easier?
                for var in self._hits[pos]:
                    for det, xy_pos in enumerate(self._hits[pos][var]):
                        self._hits[pos][var][det] = xy_pos[keep_mask]

    def propagate_dz(self, dz: Union[Tensor, float], mask: Optional[Tensor] = None) -> None:
        r"""
        Propagates all muons in their direction of flight such that afterwards they will all have moved a specified distance in the negative z direction.

        Arguments:
            dz: distance in metres to move in the negative z direction, i.e. a positive dz results in the muons travelling downwards.
            mask: (N,) Boolean tensor. If not None, only muons with True mask elements are altered.
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()
        theta = self._theta[mask]
        phi = self._phi[mask]

        r = dz / theta.cos()
        rst = r * theta.sin()
        self._x[mask] = self._x[mask] + (rst * phi.cos())
        self._y[mask] = self._y[mask] + (rst * phi.sin())
        self._z[mask] = self._z[mask] - dz

    def propagate_d(self, d: Union[Tensor, float], mask: Optional[Tensor] = None) -> None:
        r"""
        Propagates all muons in their direction of flight by the specified distances.

        Arguments:
            d: (1,) or (N,) distance(s) in metres to move.
            mask: (N,) Boolean tensor. If not None, only muons with True mask elements are altered.
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()
        theta = self._theta[mask]
        phi = self._phi[mask]

        rst = d * theta.sin()
        self._x[mask] = self._x[mask] + (rst * phi.cos())
        self._y[mask] = self._y[mask] + (rst * phi.sin())
        self._z[mask] = self._z[mask] - (d * theta.cos())

    def get_xy_mask(self, xy_low: Optional[Union[Tuple[float, float], Tensor]], xy_high: Optional[Union[Tuple[float, float], Tensor]]) -> Tensor:
        r"""
        Computes a (N,) Boolean tensor, with True values corresponding to muons which are within the supplied ranges in xy.

        Arguments:
            xy_low: (2,N) optional lower limit on xy positions
            xy_high: (2,N) optional upper limit on xy positions

        Returns:
            (N,) Boolean mask with True values corresponding to muons which are with xy positions >= xy_low and < xy_high
        """

        if xy_low is None:
            xy_low = (-math.inf, -math.inf)
        if xy_high is None:
            xy_high = (math.inf, math.inf)
        return (self.x >= xy_low[0]) * (self.x < xy_high[0]) * (self.y >= xy_low[1]) * (self.y < xy_high[1])

    def snapshot_xyz(self) -> None:
        r"""
        Store the current xy positions of the muons in `.xyz_hist`, indexed by the current z position.
        """

        self._xyz_hist.append(self.xyz.detach().cpu().clone().numpy())

    def append_hits(self, hits: Dict[str, Tensor], pos: str) -> None:
        r"""
        Record hits to `_hits`.

        Arguments:
            hits: dictionary of 'reco_xy', 'gen_xy', 'z' keys to (muons, *) tensors.
            pos: Position of detector array in which the hits were recorded, currently either 'above' or 'below'.
        """

        for k in hits:
            self._hits[pos][k].append(hits[k])

    def get_hits(
        self, xy_low: Optional[Union[Tuple[float, float], Tensor]] = None, xy_high: Optional[Union[Tuple[float, float], Tensor]] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        r"""
        Retrieve the recorded hits for the muons, optionally only for muons between the specified xy ranges.
        For ease of use, the list of hits are stacked into single tensors, resulting in
        a dictionary mapping detector-array position to a dictionary mapping hit variables to (N_muons, N_hits, *) tensors.

        Arguments:
            xy_low: (2,N) optional lower limit on xy positions
            xy_high: (2,N) optional upper limit on xy positions

        Returns:
            Hits, a dictionary mapping detector-array position to a dictionary mapping hit variables to (N_muons, N_hits, *) tensors.
        """

        if len(self._hits) == 0:
            raise ValueError("MuonBatch has no recorded hits")
        if xy_low is None and xy_high is None:
            return {p: {c: torch.stack(self._hits[p][c], dim=1) for c in self._hits[p]} for p in self._hits}
        else:
            m = self.get_xy_mask(xy_low, xy_high)
            return {p: {c: torch.stack(self._hits[p][c], dim=1)[m] for c in self._hits[p]} for p in self._hits}

    def dtheta_x(self, theta_ref_x: Tensor) -> Tensor:
        r"""
        Computes absolute difference in the theta_x between the muons and the supplied theta_x angles

        Arguments:
            theta_ref_x: (N,) tensor to compare with the muon theta_x values

        Returns:
            Absolute difference between muons' theta_x and the supplied reference theta_x
        """

        return torch.abs(self.theta_x - theta_ref_x)

    def dtheta_y(self, theta_ref_y: Tensor) -> Tensor:
        r"""
        Computes absolute difference in the theta_y between the muons and the supplied theta_y angles

        Arguments:
            theta_ref_y: (N,) tensor to compare with the muon theta_y values

        Returns:
            Absolute difference between muons' theta_y and the supplied reference theta_y
        """

        return torch.abs(self.theta_y - theta_ref_y)

    def dtheta(self, theta_ref: Tensor) -> Tensor:
        r"""
        Computes absolute difference in the theta between the muons and the supplied theta angles

        Arguments:
            theta_ref: (N,) tensor to compare with the muon theta values\

        Returns:
            Absolute difference between muons' theta and the supplied reference theta
        """

        return torch.abs(self.theta - theta_ref)

    def copy(self) -> MuonBatch:
        r"""
        Creates a copy of the muon batch at the current position and trajectories.
        Tensors are detached and cloned.

        .. important::
            This does NOT copy of hits

        Returns:
            New `MuonBatch` with xyz, and theta,phi equal to those of the current `MuonBatch`.
        """

        return MuonBatch(
            self._muons[:, sorted([self.x_dim, self.y_dim, self.p_dim, self.th_dim, self.ph_dim])].detach().clone(),
            init_z=self.z.detach().clone(),
            device=self.device,
        )

    @property
    def muons(self) -> Tensor:
        return self._muons

    @muons.setter
    def muons(self, muons: Tensor) -> None:
        self._muons = muons

    @property
    def upwards_muons(self) -> Tensor:
        return self._removed_muons

    @property
    def xyz_hist(self) -> List[Tensor]:
        return self._xyz_hist

    @property
    def x(self) -> Tensor:
        return self._x

    @x.setter
    def x(self, x: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dxy function to modify the x,y position of muons. Or modify the _muons attribute if you know what you're doing"
        )

    @property
    def _x(self) -> Tensor:
        return self._muons[:, self.x_dim]

    @_x.setter
    def _x(self, x: Tensor) -> None:
        self._muons[:, self.x_dim] = x

    @property
    def y(self) -> Tensor:
        return self._y

    @y.setter
    def y(self, y: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dxy or propagate methods to modify the x,y position of muons. Or modify the _muons attribute if you know what you're doing"
        )

    @property
    def _y(self) -> Tensor:
        return self._muons[:, self.y_dim]

    @_y.setter
    def _y(self, y: Tensor) -> None:
        self._muons[:, self.y_dim] = y

    @property
    def z(self) -> Tensor:
        return self._z

    @z.setter
    def z(self, z: Tensor) -> None:
        raise AttributeError(
            "Please use the propagate_dz or propagate_d function to modify the x,y position of muons. Or modify the _muons attribute if you know what you're doing"
        )

    @property
    def _z(self) -> Tensor:
        return self._muons[:, self.z_dim]

    @_z.setter
    def _z(self, z: Tensor) -> None:
        self._muons[:, self.z_dim] = z

    @property
    def xy(self) -> Tensor:
        return self._xy

    @xy.setter
    def xy(self, xy: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dxy or propagate methods to modify the x,y position of muons. Or modify the _muons attribute if you know what you're doing"
        )

    @property
    def _xy(self) -> Tensor:
        return self._muons[:, : self.y_dim + 1]

    @_xy.setter
    def _xy(self, xy: Tensor) -> None:
        self._muons[:, : self.y_dim + 1] = xy

    @property
    def xyz(self) -> Tensor:
        return self._xyz

    @xyz.setter
    def xyz(self, xyz: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dxy or propagate methods to modify the x,y position of muons. Or modify the _muons attribute if you know what you're doing"
        )

    @property
    def _xyz(self) -> Tensor:
        return self._muons[:, : self.z_dim + 1]

    @_xyz.setter
    def _xyz(self, xyz: Tensor) -> None:
        self._muons[:, : self.z_dim + 1] = xyz

    @property
    def mom(self) -> Tensor:
        return self._mom

    @mom.setter
    def mom(self, mom: Tensor) -> None:
        raise NotImplementedError()

    @property
    def _mom(self) -> Tensor:
        return self._muons[:, self.p_dim]

    @_mom.setter
    def _mom(self, mom: Tensor) -> None:
        self._muons[:, self.p_dim] = mom

    @property
    def reco_mom(self) -> Tensor:
        return self.mom

    @reco_mom.setter
    def reco_mom(self, mom: Tensor) -> None:
        raise NotImplementedError()

    @property
    def theta(self) -> Tensor:
        return self._theta

    @theta.setter
    def theta(self, theta: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dtheta_dphi method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )

    @property
    def _theta(self) -> Tensor:
        return self._muons[:, self.th_dim]

    @_theta.setter
    def _theta(self, theta: Tensor) -> None:
        self._muons[:, self.th_dim] = theta

    @property
    def phi(self) -> Tensor:
        return self._phi

    @phi.setter
    def phi(self, phi: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dtheta_dphi method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )

    @property
    def _phi(self) -> Tensor:
        return self._muons[:, self.ph_dim]

    @_phi.setter
    def _phi(self, phi: Tensor) -> None:
        self._muons[:, self.ph_dim] = phi

    @property
    def theta_x(self) -> Tensor:
        return self.theta_x_from_theta_phi(self.theta, self.phi)

    @theta_x.setter
    def theta_x(self, theta_x: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dtheta_dphi method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )

    @property
    def theta_y(self) -> Tensor:
        return self.theta_y_from_theta_phi(self.theta, self.phi)

    @theta_y.setter
    def theta_y(self, theta_y: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dtheta_dphi method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )

    @property
    def theta_xy(self) -> Tensor:
        return torch.stack((self.theta_x, self.theta_y), dim=-1)

    @theta_xy.setter
    def theta_xy(self, theta_xy: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dtheta_dphi method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )
