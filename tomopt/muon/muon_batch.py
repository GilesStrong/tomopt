from __future__ import annotations
from typing import Dict, List, Union, Tuple, Optional
import math
from collections import defaultdict, OrderedDict
import numpy as np

import torch
from torch import Tensor

from ..core import DEVICE

__all__ = ["MuonBatch"]


class MuonBatch:
    x_dim = 0
    y_dim = 1
    p_dim = 2
    th_dim = 3
    ph_dim = 4

    def __init__(self, xy_p_theta_phi: Tensor, init_z: Union[Tensor, float], device: torch.device = DEVICE):
        r"""
        N.B. xy [m], p [GeV], theta [r] (0, pi/2) defined w.r.t z axis, phi [r] (0, 2pi) defined anticlockwise from x axis
        Muon trajectories (theta & phi) and positions (x,y,z) are in the reference frame of the volume.
        """

        self.device = device
        self._muons = xy_p_theta_phi.to(self.device)
        self._upwards_muons: Optional[Tensor] = None
        if not isinstance(init_z, Tensor):
            init_z = Tensor([init_z])
        self._z = init_z.to(self.device)
        self._hits: Dict[str, Dict[str, List[Tensor]]] = defaultdict(lambda: defaultdict(list))
        self._xy_hist: Dict[Tensor, Tensor] = OrderedDict({})

    def __repr__(self) -> str:
        return f"Batch of {len(self)} muons"

    def __len__(self) -> int:
        return len(self.muons)

    @property
    def muons(self) -> Tensor:
        return self._muons

    @muons.setter
    def muons(self, muons: Tensor) -> None:
        self._muons = muons

    @property
    def upwards_muons(self) -> Tensor:
        return self._upwards_muons

    @property
    def xy_hist(self) -> Dict[Tensor, Tensor]:
        return self._xy_hist

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
    def z(self) -> Tensor:
        return self._z

    @z.setter
    def z(self, z: Tensor) -> None:
        raise AttributeError("Please use the propagate method to modify z. Or modify the _muons attribute if you know what you're doing")

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
            "Please use the scatter_dtheta_xy method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )

    @property
    def theta_y(self) -> Tensor:
        return self.theta_y_from_theta_phi(self.theta, self.phi)

    @theta_y.setter
    def theta_y(self, theta_y: Tensor) -> None:
        raise AttributeError(
            "Please use the scatter_dtheta_xy method to modify the direction of muons. Or modify the _muons attribute if you really know what you're doing"
        )

    def scatter_dxy(self, dx: Optional[Tensor] = None, dy: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> None:
        r"""
        dx & dy are expected to be the volume reference fram, not the muons'
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()
        if dx is not None:
            self._x[mask] = self._x[mask] + dx
        if dy is not None:
            self._y[mask] = self._y[mask] + dy

    def scatter_dtheta_dphi(self, dtheta: Optional[Tensor] = None, dphi: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> None:
        r"""
        dtheta & dphi are expected to be the volume reference fram, not the muons'
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()
        if dphi is not None:
            self._phi[mask] = (self._phi[mask] + dphi) % (2 * torch.pi)
        if dtheta is not None:
            theta = (self._theta[mask] + dtheta) % (2 * torch.pi)
            # Correct theta, must avoid double Bool mask
            phi = self._phi[mask]
            m = theta > torch.pi
            phi[m] = (phi[m] + torch.pi) % (2 * torch.pi)  # rotate in phi
            theta[m] = (2 * torch.pi) - theta[m]  # theta (0,pi)
            self._phi[mask] = phi
            self._theta[mask] = theta

        # self.remove_upwards_muons()

    def scatter_dtheta_xy(self, dtheta_x: Optional[Tensor] = None, dtheta_y: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> None:
        r"""
        dtheta_x & dtheta_y are expected to be the volume reference fram, not the muons'
        """

        if mask is None:
            mask = torch.ones(len(self._muons), device=self.device).bool()

        tx = self.theta_x[mask]
        ty = self.theta_y[mask]

        def add_theta(t: Tensor, dt: Tensor) -> Tensor:
            t = (t + dt) % (2 * torch.pi)  # Add in (0,2pi)
            # Convert to (-pi,pi)
            m = t > torch.pi
            t[m] = t[m] - (2 * torch.pi)
            return t

        if dtheta_x is not None:
            tx = add_theta(tx, dtheta_x)
        if dtheta_y is not None:
            ty = add_theta(ty, dtheta_y)

        print("tx", tx)
        print("ty", ty)

        self._theta[mask] = self.theta_from_theta_xy(tx, ty)
        self._phi[mask] = self.phi_from_theta_xy(tx, ty)

        # self.remove_upwards_muons()

    def remove_upwards_muons(self) -> None:
        r"""
        Remove muons, and their hits, if their theta >= pi/2, i.e. they are travelling upwards after a large scattering.
        Should be run after any changes to theta, but make sure that references (e.g. masks) to the complete set of muons are no longer required
        """

        mask = self._theta < torch.pi / 2  # To keep
        if mask.sum() < len(self):
            # Save muons, just in case they're useful for diagnostics
            if self._upwards_muons is None:
                self._upwards_muons = self._muons[~mask].detach().cpu().numpy()
            else:
                self._upwards_muons = np.concatenate((self._upwards_muons, self._muons[~mask].detach().cpu().numpy()), axis=0)

            # Remove muons and hits
            self._muons = self._muons[mask]
            for pos in self._hits:  # TODO: Make a HitBatch class to make this easier?
                for var in self._hits[pos]:
                    for det, xy_pos in enumerate(self._hits[pos][var]):
                        self._hits[pos][var][det] = xy_pos[mask]

    @staticmethod
    def phi_from_theta_xy(theta_x: Tensor, theta_y: Tensor) -> Tensor:
        phi = torch.arctan(theta_y.sin() / theta_x.sin())  # (-pi/2, pi/2)
        m = theta_x < 0
        phi[m] = phi[m] + torch.pi
        m = ((theta_x > 0) * (theta_y < 0)).bool()
        phi[m] = phi[m] + (2 * torch.pi)  # (0, 2pi)
        return phi

    @staticmethod
    def theta_from_theta_xy(theta_x: Tensor, theta_y: Tensor) -> Tensor:
        r"""
        This expects that 0<=theta<=pi, i.e. theta_x annd theta_y are never both greater than pi/4
        """

        stheta = (theta_x.sin().square() + theta_y.sin().square()).sqrt()
        theta = stheta.arcsin()

        pib2 = torch.pi / 2
        m = theta_x.abs() > pib2
        r = theta_x[m].sign() * pib2
        theta[m] = ((theta_x[m] - r).sin().square() + theta_y[m].sin().square()).sqrt().arcsin() + r

        m = theta_y.abs() > pib2
        r = theta_y[m].sign() * pib2
        theta[m] = (theta_x[m].sin().square() + (theta_y[m] - r).sin().square()).sqrt().arcsin() + r
        return theta

    @staticmethod
    def theta_x_from_theta_phi(theta: Tensor, phi: Tensor) -> Tensor:
        tx = (theta.sin() * phi.cos()).arcsin()
        pib2 = torch.pi / 2
        m = theta.abs() > pib2
        tx[m] = (torch.sin(theta[m] - pib2) * phi[m].cos()).arcsin() + (phi[m].cos()).arcsin()
        return tx

    @staticmethod
    def theta_y_from_theta_phi(theta: Tensor, phi: Tensor) -> Tensor:
        ty = (theta.sin() * phi.sin()).arcsin()
        pib2 = torch.pi / 2
        m = theta.abs() > pib2
        ty[m] = (torch.sin(theta[m] - pib2) * phi[m].sin()).arcsin() + (phi[m].sin()).arcsin()
        return ty

    def propagate(self, dz: Union[Tensor, float]) -> None:
        r = dz / self._theta.cos()
        rst = r * self._theta.sin()
        self._x = self._x + (rst * self._phi.cos())
        self._y = self._y + (rst * self._phi.sin())
        self._z = self._z - dz

    def get_xy_mask(self, xy_low: Optional[Union[Tuple[float, float], Tensor]], xy_high: Optional[Union[Tuple[float, float], Tensor]]) -> Tensor:
        if xy_low is None:
            xy_low = (-math.inf, -math.inf)
        if xy_high is None:
            xy_high = (math.inf, math.inf)
        return (self.x >= xy_low[0]) * (self.x < xy_high[0]) * (self.y >= xy_low[1]) * (self.y < xy_high[1])

    def snapshot_xyz(self) -> None:
        self.xy_hist[self.z.detach().cpu().clone().numpy()[0]] = self.xy.detach().cpu().clone().numpy()

    def append_hits(self, hits: Dict[str, Tensor], pos: str) -> None:
        for k in hits:
            self._hits[pos][k].append(hits[k])

    def get_hits(
        self, xy_low: Optional[Union[Tuple[float, float], Tensor]] = None, xy_high: Optional[Union[Tuple[float, float], Tensor]] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        if len(self._hits) == 0:
            raise ValueError("MuonBatch has no recorded hits")
        if xy_low is None and xy_high is None:
            return {p: {c: torch.stack(self._hits[p][c], dim=1) for c in self._hits[p]} for p in self._hits}
        else:
            m = self.get_xy_mask(xy_low, xy_high)
            return {p: {c: torch.stack(self._hits[p][c], dim=1)[m] for c in self._hits[p]} for p in self._hits}

    def dtheta_x(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta_x - mu.theta_x)

    def dtheta_y(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta_y - mu.theta_y)

    def dtheta(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta - mu.theta)

    def copy(self) -> MuonBatch:
        return MuonBatch(self._muons.detach().clone(), init_z=self.z.detach().clone(), device=self.device)
