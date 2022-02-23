from __future__ import annotations
from typing import Dict, List, Union, Tuple, Optional
import math
from collections import defaultdict, OrderedDict

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
        self.device = device
        self.muons = xy_p_theta_phi.to(self.device)
        if not isinstance(init_z, Tensor):
            init_z = Tensor([init_z])
        self.z = init_z.to(self.device)
        self.hits: Dict[str, Dict[str, List[Tensor]]] = defaultdict(lambda: defaultdict(list))
        self.xy_hist: Dict[Tensor, Tensor] = OrderedDict({})

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
    def x(self) -> Tensor:
        return self._muons[:, self.x_dim]

    @x.setter
    def x(self, x: Tensor) -> None:
        self._muons[:, self.x_dim] = x

    @property
    def y(self) -> Tensor:
        return self._muons[:, self.y_dim]

    @y.setter
    def y(self, y: Tensor) -> None:
        self._muons[:, self.y_dim] = y

    @property
    def xy(self) -> Tensor:
        return self._muons[:, : self.y_dim + 1]

    @xy.setter
    def xy(self, xy: Tensor) -> None:
        self._muons[:, : self.y_dim + 1] = xy

    @property
    def mom(self) -> Tensor:
        return self._muons[:, self.p_dim]

    @mom.setter
    def mom(self, mom: Tensor) -> None:
        self._muons[:, self.p_dim] = mom

    @property
    def reco_mom(self) -> Tensor:
        return self.mom

    @reco_mom.setter
    def reco_mom(self, mom: Tensor) -> None:
        raise NotImplementedError()

    @property
    def theta_x(self) -> Tensor:
        return (self.theta.sin() * self.phi.cos()).arcsin()

    @theta_x.setter
    def theta_x(self, theta_x: Tensor) -> None:
        raise NotImplementedError()

    @property
    def theta_y(self) -> Tensor:
        return (self.theta.sin() * self.phi.sin()).arcsin()

    @theta_y.setter
    def theta_y(self, theta_y: Tensor) -> None:
        raise NotImplementedError()

    @property
    def theta(self) -> Tensor:
        return self._muons[:, self.th_dim]

    @theta.setter
    def theta(self, theta: Tensor) -> None:
        self._muons[:, self.th_dim] = theta

    @property
    def phi(self) -> Tensor:
        return self._muons[:, self.ph_dim]

    @phi.setter
    def phi(self, phi: Tensor) -> None:
        self._muons[:, self.ph_dim] = phi

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
        return (theta_x.sin().square() + theta_y.sin().square()).sqrt().arcsin()

    def propagate(self, dz: Union[Tensor, float]) -> None:
        with torch.no_grad():
            r = dz / self.theta.cos()
            rst = r * self.theta.sin()
            self.x = self.x + (rst * self.phi.cos())
            self.y = self.y + (rst * self.phi.sin())
            self.z = self.z - dz

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
            self.hits[pos][k].append(hits[k])

    def get_hits(
        self, xy_low: Optional[Union[Tuple[float, float], Tensor]] = None, xy_high: Optional[Union[Tuple[float, float], Tensor]] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        if len(self.hits) == 0:
            raise ValueError("MuonBatch has no recorded hits")
        if xy_low is None and xy_high is None:
            return {p: {c: torch.stack(self.hits[p][c], dim=1) for c in self.hits[p]} for p in self.hits}
        else:
            m = self.get_xy_mask(xy_low, xy_high)
            return {p: {c: torch.stack(self.hits[p][c], dim=1)[m] for c in self.hits[p]} for p in self.hits}

    def dtheta_x(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta_x - mu.theta_x)

    def dtheta_y(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta_y - mu.theta_y)

    def dtheta(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta - mu.theta)

    def copy(self) -> MuonBatch:
        return MuonBatch(self._muons.detach().clone(), init_z=self.z.detach().clone(), device=self.device)
