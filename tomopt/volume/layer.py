from tomopt.volume.panel import DetectorPanel
from typing import Iterator, Optional, Callable, Dict, Tuple, List, Union
import numpy as np
from abc import ABCMeta, abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .scatter_model import SCATTER_MODEL
from ..core import DEVICE
from ..muon import MuonBatch

__all__ = ["PassiveLayer", "VoxelDetectorLayer", "PanelDetectorLayer"]


class Layer(nn.Module):
    def __init__(self, lw: Tensor, z: float, size: float, device: torch.device = DEVICE):
        super().__init__()
        self.lw, self.z, self.size, self.device = lw.to(device), torch.tensor([z], device=device), size, device
        self.rad_length: Optional[Tensor] = None

    def _compute_displacements(self, *, x0: Tensor, deltaz: Union[Tensor, float], theta_xy: Tensor, mom: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if not SCATTER_MODEL.initialised:
            SCATTER_MODEL.load_data()  # Delay loading until requrired

        dtheta, dxy = SCATTER_MODEL.compute_scattering(x0=x0, deltaz=deltaz, theta_xy=theta_xy, mom=mom)
        return dtheta[:, 0], dtheta[:, 1], dxy[:, 0], dxy[:, 1]

    def scatter_and_propagate(self, mu: MuonBatch, deltaz: Union[Tensor, float]) -> None:
        """
        This function produces a model of multiple scattering through a layer of material
        of depth deltaz

        TODO: Expand to sum over traversed voxels
        """

        if self.rad_length is not None:
            mask = mu.get_xy_mask((0, 0), self.lw)  # Only scatter muons inside volume
            xy_idx = self.mu_abs2idx(mu, mask)

            x0 = self.rad_length[xy_idx[:, 0], xy_idx[:, 1]]  # Already masked
            dx, dy, dtheta_x, dtheta_y = self._compute_displacements(x0=x0, deltaz=deltaz, theta_xy=mu.theta_xy[mask], mom=mu.mom[mask])

            # Update to position at scattering.
            mu.x[mask] = mu.x[mask] + dx
            mu.y[mask] = mu.y[mask] + dy
            mu.propagate(deltaz)
            mu.theta_x[mask] = mu.theta_x[mask] + dtheta_x
            mu.theta_y[mask] = mu.theta_y[mask] + dtheta_y
        else:
            mu.propagate(deltaz)

    def mu_abs2idx(self, mu: MuonBatch, mask: Optional[Tensor] = None) -> Tensor:
        xy = mu.xy
        if mask is not None:
            xy = xy[mask]
        return self.abs2idx(xy)

    def abs2idx(self, xy: Tensor) -> Tensor:
        return torch.floor(xy / self.size).long()

    @abstractmethod
    def forward(self, mu: MuonBatch) -> None:
        pass


class PassiveLayer(Layer):
    def __init__(self, lw: Tensor, z: float, size: float, rad_length_func: Optional[Callable[..., Tensor]] = None, device: torch.device = DEVICE):
        super().__init__(lw=lw, z=z, size=size, device=device)
        if rad_length_func is not None:
            self.load_rad_length(rad_length_func)

    def load_rad_length(self, rad_length_func: Callable[..., Tensor]) -> None:
        self.rad_length = rad_length_func(z=self.z, lw=self.lw, size=self.size).to(self.device)

    def forward(self, mu: MuonBatch, n: int = 2) -> None:
        for _ in range(n):
            self.scatter_and_propagate(mu, deltaz=self.size)


class AbsDetectorLayer(Layer, metaclass=ABCMeta):
    def __init__(
        self,
        pos: str,
        *,
        lw: Tensor,
        z: float,
        size: float,
        device: torch.device = DEVICE,
    ):
        super().__init__(lw=lw, z=z, size=size, device=device)
        self.pos = pos

    @abstractmethod
    def forward(self, mu: MuonBatch) -> None:
        pass

    @abstractmethod
    def get_cost(self) -> Tensor:
        pass

    def conform_detector(self) -> None:
        pass


class VoxelDetectorLayer(AbsDetectorLayer):
    def __init__(
        self,
        pos: str,
        *,
        init_res: float,
        init_eff: float,
        lw: Tensor,
        z: float,
        size: float,
        eff_cost_func: Callable[[Tensor], Tensor],
        res_cost_func: Callable[[Tensor], Tensor],
        device: torch.device = DEVICE,
    ):
        super().__init__(pos=pos, lw=lw, z=z, size=size, device=device)
        self.resolution = nn.Parameter(torch.zeros(list((self.lw / size).long()), device=self.device) + init_res)
        self.efficiency = nn.Parameter(torch.zeros(list((self.lw / size).long()), device=self.device) + init_eff)
        self.eff_cost_func, self.res_cost_func = eff_cost_func, res_cost_func

    def conform_detector(self) -> None:
        with torch.no_grad():
            self.resolution.clamp_(min=1, max=1e7)
            self.efficiency.clamp_(min=1e-7, max=1)

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:  # to dense and add precision
        mask = mu.get_xy_mask((0, 0), self.lw)
        res = torch.zeros(len(mu), device=self.device)  # Zero detection outside detector
        xy_idxs = self.mu_abs2idx(mu, mask)
        res[mask] = self.resolution[xy_idxs[:, 0], xy_idxs[:, 1]]
        res = F.relu(res[:, None]) + 1e-17  # Negative resolution --> zero+eps due to cost function def

        xy0 = torch.zeros((len(mu), 2), device=self.device)
        xy0[mask] = xy_idxs.float() * self.size  # Low-left of voxel
        rel_xy = mu.xy - xy0
        reco_rel_xy = rel_xy + (torch.randn((len(mu), 2), device=self.device), res)
        reco_rel_xy = torch.clamp(reco_rel_xy, 0, self.size - 1e-7)  # Prevent reco hit from exiting triggering voxel
        reco_xy = xy0 + reco_rel_xy

        hits = {
            "reco_xy": reco_xy,
            "gen_xy": mu.xy.detach().clone(),
            "z": self.z.expand_as(mu.x)[:, None] - (self.size, 2),
        }
        return hits

    def forward(self, mu: MuonBatch) -> None:
        self.scatter_and_propagate(mu, self.size / 2)
        mu.append_hits(self.get_hits(mu), self.pos)
        self.scatter_and_propagate(mu, self.size / 2)

    def get_cost(self) -> Tensor:
        return self.eff_cost_func(self.efficiency).sum() + self.res_cost_func(self.resolution).sum()


class PanelDetectorLayer(AbsDetectorLayer):
    def __init__(
        self,
        pos: str,
        lw: Tensor,
        z: float,
        size: float,
        panels: nn.ModuleList,  # nn.ModuleList[DetectorPanel]
    ):
        super().__init__(pos=pos, lw=lw, z=z, size=size, device=self.get_device(panels))
        if isinstance(panels, list):
            panels = nn.ModuleList(panels)
        self.panels = panels

    @staticmethod
    def get_device(panels: nn.ModuleList) -> torch.device:
        device = panels[0].device
        if len(panels) > 1:
            for p in panels[1:]:
                if p.device != device:
                    raise ValueError("All panels must use the same device, but found multiple devices")
        return device

    def get_panel_zorder(self) -> List[int]:
        return list(np.argsort([p.z.detach().cpu().item() for p in self.panels])[::-1])

    def yield_zordered_panels(self) -> Iterator[DetectorPanel]:
        for i in self.get_panel_zorder():
            yield self.panels[i]

    def conform_detector(self) -> None:
        lw = self.lw.detach().cpu().numpy()
        z = self.z.detach().cpu()[0]
        for p in self.panels:
            p.clamp_params(xyz_low=(0, 0, z - self.size), xyz_high=(lw[0], lw[1], z))

    def forward(self, mu: MuonBatch) -> None:
        for p in self.yield_zordered_panels():
            self.scatter_and_propagate(mu, mu.z - p.z)  # Move to panel
            mu.append_hits(p.get_hits(mu), self.pos)
        self.scatter_and_propagate(mu, mu.z - (self.z - self.size))  # Move to bottom of layer

    def get_cost(self) -> Tensor:
        cost = None
        for p in self.panels:
            cost = p.get_cost() if cost is None else cost + p.get_cost()
        return cost
