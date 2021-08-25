from tomopt.volume.layer import Layer
from typing import Tuple, List, Callable

import torch
from torch import nn, Tensor

from .layer import AbsDetectorLayer, PassiveLayer
from ..muon import MuonBatch

__all__ = ["Volume"]


class Volume(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self._device = self._get_device()
        
    @property
    def device(self) -> torch.device:
        return self._device
        
    def _get_device(self):
        device = self.layers[0].device
        for l in self.layers[1:]:
            if l.device != device:
                raise ValueError('All layers must use the same device, but found multiple devices')
        return device

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]

    def get_detectors(self) -> List[AbsDetectorLayer]:
        return [l for l in self.layers if isinstance(l, AbsDetectorLayer)]

    def get_passives(self) -> List[PassiveLayer]:
        return [l for l in self.layers if isinstance(l, PassiveLayer)]

    def get_rad_cube(self) -> Tensor:
        vols = list(reversed(self.get_passives()))  # reversed to match lookup_xyz_coords: layer zero = bottom layer
        if len(vols) == 0:
            raise ValueError("self.layers contains no passive layers")
        rads = [v.rad_length for v in vols if v.rad_length is not None]
        if len(rads) > 0:
            return torch.stack([v.rad_length for v in vols if v.rad_length is not None], dim=0)
        else:
            raise AttributeError('None of volume layers have a non-None rad_length attribute')

    def lookup_passive_xyz_coords(self, xyz: Tensor) -> Tensor:
        r"""Assume same size for all layers for now and no intermedeate detector layers"""
        if len(xyz.shape) == 1:
            xyz = xyz[None, :]

        if n := (
            ((xyz[:, :2] > self.lw) + (xyz[:, :2] < 0)).sum(1) + (xyz[:, 2] < self.get_passive_z_range()[0]) + ((xyz[:, 2] > self.get_passive_z_range()[1]))
        ).sum():
            raise ValueError(f"{n} Coordinates outside passive volume")
        xyz[:, 2] = xyz[:, 2] - self.get_passive_z_range()[0]
        return torch.floor(xyz / self.passive_size).long()

    def load_rad_length(self, rad_length_func: Callable[..., Tensor]) -> None:
        for p in self.get_passives():
            p.load_rad_length(rad_length_func)

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
            cost = torch.zeros((1), device=self.device)
        return cost

    @property
    def lw(self) -> Tensor:
        return self.get_passives()[-1].lw

    @property
    def passive_size(self) -> float:
        return self.get_passives()[-1].size  # Same size for each passive layer

    @property
    def h(self) -> float:
        return self.layers[0].z

    def get_passive_z_range(self) -> Tuple[Tensor, Tensor]:
        ps = self.get_passives()
        return ps[-1].z - self.passive_size, ps[0].z
