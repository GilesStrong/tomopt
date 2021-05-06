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

    def get_passives(self) -> List[PassiveLayer]:
        return [l for l in self.layers if isinstance(l, PassiveLayer)]

    def get_rad_cube(self) -> Tensor:
        vols = list(reversed(self.get_passives()))  # reversed to match lookup_xyz_coords: layer zero = bottom layer
        if len(vols) == 0:
            raise ValueError("self.layers contains no passive layers")
        return torch.stack([v.rad_length for v in vols if v.rad_length is not None], dim=0)

    def lookup_xyz_coords(self, xyz: Tensor, passive_only: bool) -> Tensor:
        r"""Assume same size for all layers for now and no intermedeate detector layers"""
        if len(xyz.shape) == 1:
            xyz = xyz[None, :]

        if passive_only:
            if n := (
                ((xyz[:, :2] > self.lw) + (xyz[:, :2] < 0)).sum(1) + (xyz[:, 2] < self.get_passive_z_range()[0]) + ((xyz[:, 2] > self.get_passive_z_range()[1]))
            ).sum():
                raise ValueError(f"{n} Coordinates outside passive volume")
            xyz[:, 2] = xyz[:, 2] - self.get_passive_z_range()[0]
        else:
            if n := (((xyz[:, :2] > self.lw) + (xyz[:, :2] < 0)).sum(1) + (xyz[:, 2] < 0) + ((xyz[:, 2] > self.h))).sum():
                raise ValueError(f"{n} Coordinates outside volume")
        return torch.floor(xyz / self.size).long()

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
        if cost is None:
            cost = torch.zeros((1))
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
