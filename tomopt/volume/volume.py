from tomopt.volume.layer import Layer
from typing import Tuple, List, Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layer import AbsDetectorLayer, PassiveLayer
from ..muon import MuonBatch

__all__ = ["Volume"]


class Volume(nn.Module):
    def __init__(self, layers: nn.ModuleList, budget: Optional[Tensor] = None):
        super().__init__()
        self.layers, self.budget = layers, budget
        self._device = self._get_device()
        self._check_passives()
        self._target: Optional[Tensor] = None
        self._edges: Optional[Tensor] = None

        self.budget_weights = nn.Parameter(torch.zeros(len([l for l in self.layers if hasattr(l, "get_cost")]), device=self._device))

    @property
    def edges(self) -> Tensor:
        if self._edges is None:
            self._edges = self.build_edges()
        return self._edges

    @property
    def centres(self) -> Tensor:
        if self._edges is None:
            self._edges = self.build_edges()
        return self._edges + (self.passive_size / 2)

    def build_edges(self) -> Tensor:
        bounds = (
            self.passive_size
            * np.mgrid[
                0 : round(self.lw.detach().cpu().numpy()[0] / self.passive_size) : 1,
                0 : round(self.lw.detach().cpu().numpy()[1] / self.passive_size) : 1,
                round(self.get_passive_z_range()[0].detach().cpu().numpy()[0] / self.passive_size) : round(
                    self.get_passive_z_range()[1].detach().cpu().numpy()[0] / self.passive_size
                ) : 1,
            ]
        )
        # bounds[2] = np.flip(bounds[2])  # z is reversed
        return torch.tensor(bounds.reshape(3, -1).transpose(-1, -2), dtype=torch.float32, device=self.device)

    def _check_passives(self) -> None:
        lw, sz = None, None
        for l in self.get_passives():
            if lw is None:
                lw = l.lw
            elif (lw != l.lw).any():
                raise ValueError("All passive layers must have the same length and width (LW)")
            if sz is None:
                sz = l.size
            elif sz != l.size:
                raise ValueError("All passive layers must have the same size")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def target(self) -> Optional[Tensor]:
        return self._target

    def _get_device(self) -> torch.device:
        device = self.layers[0].device
        if len(self.layers) > 1:
            for l in self.layers[1:]:
                if l.device != device:
                    raise ValueError("All layers must use the same device, but found multiple devices")
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
            raise AttributeError("None of volume layers have a non-None rad_length attribute")

    def lookup_passive_xyz_coords(self, xyz: Tensor) -> Tensor:
        r"""Assume same size for all layers for now and no intermedeate detector layers"""
        if len(xyz.shape) == 1:
            xyz = xyz[None, :]

        if n := (
            ((xyz[:, :2] > self.lw) + (xyz[:, :2] < 0)).sum(1) + (xyz[:, 2] < self.get_passive_z_range()[0]) + ((xyz[:, 2] > self.get_passive_z_range()[1]))
        ).sum():
            raise ValueError(f"{n} coordinate(s) outside passive volume")
        xyz[:, 2] = xyz[:, 2] - self.get_passive_z_range()[0]
        return torch.floor(xyz / self.passive_size).long()

    def load_rad_length(self, rad_length_func: Callable[..., Tensor], target: Optional[Tensor] = None) -> None:
        self._target = target
        for p in self.get_passives():
            p.load_rad_length(rad_length_func)

    def forward(self, mu: MuonBatch) -> None:  # Expand to take volume as input, too
        if self.budget is not None:
            budget_idx = 0
            layer_budgets = self.budget * F.softmax(self.budget_weights, dim=-1)
        for l in self.layers:
            if self.budget is not None and hasattr(l, "get_cost"):
                l(mu, budget=layer_budgets[budget_idx])
                budget_idx += 1
            else:
                l(mu)
            mu.snapshot_xyz()

    def get_cost(self) -> Tensor:
        cost = None
        if self.budget is not None:
            budget_idx = 0
            layer_budgets = self.budget * F.softmax(self.budget_weights, dim=-1)
        for l in self.layers:
            if hasattr(l, "get_cost"):
                c = l.get_cost()
                if self.budget is not None:
                    c = c * layer_budgets[budget_idx]
                    budget_idx += 1
                if cost is None:
                    cost = c
                else:
                    cost = cost + c
        if cost is None:
            cost = torch.zeros((1), device=self.device)
        if self.budget is not None and (cost - self.budget).abs() > 1e-3:
            raise RuntimeWarning(f"Total layer cost, {cost} does not match the specified budget, {self.budget}")
        return cost

    @property
    def lw(self) -> Tensor:
        return self.get_passives()[-1].lw  # Same LW for each passive layer

    @property
    def passive_size(self) -> float:
        return self.get_passives()[-1].size  # Same size for each passive layer

    @property
    def h(self) -> Tensor:
        return self.layers[0].z

    def get_passive_z_range(self) -> Tuple[Tensor, Tensor]:
        ps = self.get_passives()
        return ps[-1].z - self.passive_size, ps[0].z
