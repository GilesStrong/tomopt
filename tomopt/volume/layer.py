from tomopt.volume.panel import DetectorPanel
from tomopt.volume.heatmap import DetectorHeatMap
from typing import Iterator, Optional, Callable, Dict, Tuple, List, Union
import math
import numpy as np
from abc import ABCMeta, abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..core import DEVICE, SCATTER_COEF_A, SCATTER_COEF_B
from ..muon import MuonBatch

__all__ = ["PassiveLayer", "VoxelDetectorLayer", "PanelDetectorLayer"]


class Layer(nn.Module):
    def __init__(self, lw: Tensor, z: float, size: float, device: torch.device = DEVICE):
        super().__init__()
        self.lw, self.z, self.size, self.device = lw.to(device), torch.tensor([z], dtype=torch.float32, device=device), size, device
        self.rad_length: Optional[Tensor] = None

    @staticmethod
    def _compute_n_x0(*, x0: Tensor, deltaz: Union[Tensor, float], theta: Tensor) -> Tensor:
        return deltaz / (x0 * torch.cos(theta))

    def _compute_displacements(
        self, *, n_x0: Tensor, deltaz: Union[Tensor, float], theta_x: Tensor, theta_y: Tensor, mom: Tensor, log_term: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Returns dx, dy, dtheta, dphi of the muons in the refernce frame of the volume
        """

        n = len(n_x0)
        z1 = torch.randn(n, device=self.device)
        z2 = torch.randn(n, device=self.device)

        theta0 = (SCATTER_COEF_A / mom) * torch.sqrt(n_x0)
        if log_term:
            theta0 = theta0 * (1 + (SCATTER_COEF_B * torch.log(n_x0)))
        # These are in the muons' reference frames NOT the volume's!!!
        theta_msc = math.sqrt(2) * z2 * theta0
        phi_msc = torch.rand(n, device=self.device) * 2 * math.pi
        dh_msc = math.sqrt(2) * deltaz * torch.sin(theta0) * ((z1 / math.sqrt(12)) + (z2 / 2))

        # Compute dtheta_xy in muon ref frame, but we're free to rotate the muon,
        # since dtheta_xy doesn't depend on muon position
        # Therefore assign theta_y axis (muon ref) to be in the theta direction (vol ref),
        # and theta_x axis (muon ref) to be in the phi direction (vol ref)
        dphi = theta_msc * torch.cos(phi_msc)  # dtheta_y in muon ref
        dtheta = theta_msc * torch.sin(phi_msc)  # dtheta_x in muon ref

        # Note that if a track incides on a layer
        # with angle theta_mu, the dx and dy displacements are relative to zero angle
        # (generation of MSC formulas are oblivious of angle of incidence) so we need
        # to rescale them by cos of thetax and thetay
        dx = dh_msc * torch.cos(phi_msc) * torch.cos(theta_x)
        dy = dh_msc * torch.sin(phi_msc) * torch.cos(theta_y)
        return dx, dy, dtheta, dphi

    def scatter_and_propagate(self, mu: MuonBatch, deltaz: Union[Tensor, float]) -> None:
        """
        This function produces a model of multiple scattering through a layer of material
        of depth deltaz

        TODO: Expand to sum over traversed voxels
        """

        if self.rad_length is not None:
            mask = mu.get_xy_mask((0, 0), self.lw)  # Only scatter muons inside volume
            xy_idx = self.mu_abs2idx(mu, mask)

            x0 = self.rad_length[xy_idx[:, 0], xy_idx[:, 1]]
            n_x0 = self._compute_n_x0(x0=x0, deltaz=deltaz, theta=mu.theta[mask])
            dx, dy, dtheta, dphi = self._compute_displacements(n_x0=n_x0, deltaz=deltaz, theta_x=mu.theta_x[mask], theta_y=mu.theta_y[mask], mom=mu.mom[mask])

            # Update to position at scattering.
            mu.scatter_dxy(dx=dx, dy=dy, mask=mask)
            mu.propagate(deltaz)
            mu.scatter_dtheta_dphi(dtheta=dtheta, dphi=dphi, mask=mask)
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

    def __repr__(self) -> str:
        return f"""PassiveLayer located at z={self.z}"""

    def load_rad_length(self, rad_length_func: Callable[..., Tensor]) -> None:
        self.rad_length = rad_length_func(z=self.z, lw=self.lw, size=self.size).to(self.device)

    def forward(self, mu: MuonBatch, n: int = 2) -> None:
        for _ in range(n):
            self.scatter_and_propagate(mu, deltaz=self.size / n)


class AbsDetectorLayer(Layer, metaclass=ABCMeta):
    _n_costs = 0

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
        self.type_label = ""

    @abstractmethod
    def forward(self, mu: MuonBatch) -> None:
        pass

    @abstractmethod
    def get_cost(self) -> Tensor:
        pass

    def conform_detector(self) -> None:
        pass

    def assign_budget(self, budget: Optional[Tensor]) -> None:
        r"""
        For assigning budget withput calling forward
        """

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
        self.type_label = "voxel"

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
        reco_rel_xy = rel_xy + (torch.randn((len(mu), 2), device=self.device) / res)
        reco_rel_xy = torch.clamp(reco_rel_xy, 0, self.size - 1e-7)  # Prevent reco hit from exiting triggering voxel
        reco_xy = xy0 + reco_rel_xy

        hits = {
            "reco_xy": reco_xy,
            "gen_xy": mu.xy.detach().clone(),
            "z": self.z.expand_as(mu.x)[:, None] - (self.size / 2),
        }
        return hits

    def forward(self, mu: MuonBatch) -> None:
        self.scatter_and_propagate(mu, self.size / 2)
        mu.append_hits(self.get_hits(mu), self.pos)
        self.scatter_and_propagate(mu, self.size / 2)

    def get_cost(self) -> Tensor:
        return self.eff_cost_func(self.efficiency).sum() + self.res_cost_func(self.resolution).sum()


class PanelDetectorLayer(AbsDetectorLayer):
    def __init__(self, pos: str, lw: Tensor, z: float, size: float, panels: Union[List[DetectorPanel], List[DetectorHeatMap], nn.ModuleList]):
        if isinstance(panels, list):
            panels = nn.ModuleList(panels)
        super().__init__(pos=pos, lw=lw, z=z, size=size, device=self.get_device(panels))
        self.panels = panels
        self._n_costs = len(self.panels)
        if isinstance(panels[0], DetectorHeatMap):
            self.type_label = "heatmap"
        elif isinstance(panels[0], DetectorPanel):
            self.type_label = "panel"

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

    def yield_zordered_panels(self) -> Union[Iterator[Tuple[int, DetectorPanel]], Iterator[Tuple[int, DetectorHeatMap]]]:
        for i in self.get_panel_zorder():
            yield i, self.panels[i]

    def conform_detector(self) -> None:
        lw = self.lw.detach().cpu().numpy()
        z = self.z.detach().cpu()[0]
        for p in self.panels:
            if self.type_label == "heatmap":
                xy_low = p.xy_fix[0] - p.range_mult * p.delta_xy
                xy_high = p.xy_fix[1] + p.range_mult * p.delta_xy
                xy_low = torch.max(torch.tensor(0.0), xy_low)
                xy_high = torch.min(torch.tensor(lw[0]), xy_high)

                p.clamp_params(
                    musigz_low=(xy_low, 0.0, z - self.size),
                    musigz_high=(xy_high, lw[1], z),
                )
            else:
                p.clamp_params(
                    xyz_low=(0, 0, z - self.size),
                    xyz_high=(lw[0], lw[1], z),
                )

    def forward(self, mu: MuonBatch) -> None:
        for i, p in self.yield_zordered_panels():
            self.scatter_and_propagate(mu, mu.z - p.z.detach())  # Move to panel
            hits = p.get_hits(mu)
            mu.append_hits(hits, self.pos)
        self.scatter_and_propagate(mu, mu.z - (self.z - self.size))  # Move to bottom of layer

    def get_cost(self) -> Tensor:
        cost = None
        for p in self.panels:
            cost = p.get_cost() if cost is None else cost + p.get_cost()
        return cost

    def assign_budget(self, budget: Optional[Tensor]) -> None:
        if budget is not None:
            for i, p in self.yield_zordered_panels():
                p.assign_budget(budget[i])
