from __future__ import annotations
from typing import Dict, List, Union
from collections import defaultdict, OrderedDict

import torch
from torch import Tensor

from ..core import DEVICE

__all__ = ["MuonBatch"]


class MuonBatch:
    def __init__(self, muons: Tensor, init_z: Union[Tensor, float], device: torch.device = DEVICE):
        r"""
        coords = (0:x~Uniform[0,1], 1:y~Uniform[0,1], 2:p=100GeV, 3:theta_x~cos2(a) a~Uniform[0,0.5pi], 4:theta_y~cos2(a) a~Uniform[0,0.5pi]
        """

        self.device = device
        self.muons = muons.to(self.device)
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
        return self._muons[:, 0]

    @x.setter
    def x(self, x: Tensor) -> None:
        self._muons[:, 0] = x

    @property
    def y(self) -> Tensor:
        return self._muons[:, 1]

    @y.setter
    def y(self, y: Tensor) -> None:
        self._muons[:, 1] = y

    @property
    def xy(self) -> Tensor:
        return self._muons[:, :2]

    @xy.setter
    def xy(self, xy: Tensor) -> None:
        self._muons[:, :2] = xy

    @property
    def p(self) -> Tensor:
        return self._muons[:, 2]

    @p.setter
    def p(self, p: Tensor) -> None:
        self._muons[:, 2] = p

    @property
    def reco_p(self) -> Tensor:
        return self.p

    @reco_p.setter
    def reco_p(self, p: Tensor) -> None:
        raise NotImplementedError()

    @property
    def theta_x(self) -> Tensor:
        return self._muons[:, 3]

    @theta_x.setter
    def theta_x(self, theta_x: Tensor) -> None:
        self._muons[:, 3] = theta_x

    @property
    def theta_y(self) -> Tensor:
        return self._muons[:, 4]

    @theta_y.setter
    def theta_y(self, theta_y: Tensor) -> None:
        self._muons[:, 4] = theta_y

    @property
    def theta(self) -> Tensor:
        return torch.sqrt(((self.theta_x) ** 2) + ((self.theta_y) ** 2))

    @theta.setter
    def theta(self, theta: Tensor) -> None:
        raise NotImplementedError()

    def propagate(self, dz: float) -> None:
        self.x = self.x + (dz * torch.tan(self.theta_x))
        self.y = self.y + (dz * torch.tan(self.theta_y))
        self.z -= dz

    def get_xy_mask(self, lw: Tensor) -> Tensor:
        return (self.x >= 0) * (self.x < lw[0]) * (self.y >= 0) * (self.y < lw[1])

    def snapshot_xyz(self) -> None:
        self.xy_hist[self.z.detach().cpu().clone().numpy()[0]] = self.xy.detach().cpu().clone().numpy()

    def append_hits(self, hits: Dict[str, Tensor], pos: str) -> None:
        for k in hits:
            self.hits[pos][k].append(hits[k])

    def get_hits(self, lw: Tensor) -> Dict[str, Dict[str, Tensor]]:
        if len(self.hits) == 0:
            raise ValueError("MuonBatch has no recorded hits")
        m = self.get_xy_mask(lw)
        return {p: {c: torch.stack(self.hits[p][c], dim=1)[m] for c in self.hits[p]} for p in self.hits}

    def dtheta_x(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta_x - mu.theta_x)

    def dtheta_y(self, mu: MuonBatch) -> Tensor:
        return torch.abs(self.theta_y - mu.theta_y)

    def dtheta(self, mu: MuonBatch) -> Tensor:
        return torch.sqrt((self.dtheta_x(mu) ** 2) + (self.dtheta_y(mu) ** 2))

    def copy(self) -> MuonBatch:
        return MuonBatch(self._muons.detach().clone(), init_z=self.z.detach().clone(), device=self.device)
