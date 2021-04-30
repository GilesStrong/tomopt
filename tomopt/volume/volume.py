from typing import Tuple, List

import torch
from torch import nn, Tensor

from . import DetectorLayer, PassiveLayer
from ..muon import MuonBatch

__all__ = ["Volume"]


class Volume(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def get_detectors(self) -> List[DetectorLayer]:
        return [l for l in self.layers if isinstance(l, DetectorLayer)]

    def get_passives(self) -> List[DetectorLayer]:
        return [l for l in self.layers if isinstance(l, PassiveLayer)]

    def get_rad_cube(self) -> Tensor:
        vols = reversed(self.get_passives())
        return torch.stack([v.rad_length for v in vols], dim=0)

    def lookup_coords(self, xyz: Tensor, passive_only: bool) -> Tensor:
        r"""Assume same size for all layers for now and no intermedeate detector layers"""
        if len(xyz.shape) == 1:
            xyz = xyz[None, :]
        sz = self.layers[0].size
        if passive_only:
            xyz[:, 2] = xyz[:, 2] - self.get_passives()[-1].z + sz
        return torch.floor(xyz / sz).long()

    def forward(self, mu: MuonBatch) -> None:  # Expand to take volume as input, too
        for l in self.layers:
            l(mu)
            mu.snapshot_xyz()

    def get_cost(self) -> Tensor:
        cost = None
        for l in self.layers:
            if hasattr(l, "get_cost"):
                if cost is None:
                    cost = l.get_cost()
                else:
                    cost = cost + l.get_cost()
                return cost

    @property
    def lw(self) -> Tensor:
        return self.layers[-1].lw

    @property
    def size(self) -> float:
        return self.layers[-1].size  # Same size for each layer

    @property
    def h(self) -> float:
        return len(self.layers) * self.layers[-1].size  # Same size for each layer

    def get_passive_z_range(self) -> Tuple[Tensor, Tensor]:
        ps = self.get_passives()
        return ps[-1].z - self.size, ps[0].z
