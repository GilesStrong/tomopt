from typing import Iterator, Optional, Callable, Dict, List, Union, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
import math

import torch
from torch import nn, Tensor

from .scatter_model import SCATTER_MODEL
from ..core import DEVICE, SCATTER_COEF_A, SCATTER_COEF_B
from ..muon import MuonBatch
from tomopt.volume.panel import DetectorPanel
from tomopt.volume.heatmap import DetectorHeatMap

__all__ = ["PassiveLayer", "PanelDetectorLayer"]


class Layer(nn.Module):
    def __init__(self, lw: Tensor, z: float, size: float, scatter_model: str = "pdg", device: torch.device = DEVICE):
        super().__init__()
        self.lw, self.z, self.size, self.scatter_model, self.device = (
            lw.to(device),
            torch.tensor([z], dtype=torch.float32, device=device),
            size,
            scatter_model,
            device,
        )
        self.rad_length: Optional[Tensor] = None

    def _geant_scatter(self, *, x0: Tensor, deltaz: Union[Tensor, float], theta_xy: Tensor, mom: Tensor) -> Dict[str, Tensor]:
        return SCATTER_MODEL.compute_scattering(x0=x0, deltaz=deltaz, theta_xy=theta_xy, mom=mom)

    def _pdg_scatter(self, *, x0: Tensor, deltaz: Union[Tensor, float], theta_xy: Tensor, mom: Tensor, log_term: bool = True) -> Dict[str, Tensor]:
        r"""
        Returns dx, dy, dtheta_x, dtheta_y of the muons in the refernce frame of the volume
        """

        flight = deltaz / torch.cos(theta_xy)
        n_x0 = flight / x0

        n = len(n_x0)
        z1 = torch.randn((n, 2), device=self.device)
        z2 = torch.randn((n, 2), device=self.device)
        theta0 = (SCATTER_COEF_A / mom) * torch.sqrt(n_x0)
        if log_term:
            theta0 = theta0 * (1 + (SCATTER_COEF_B * torch.log(n_x0)))
        # These are in the muons' reference frames NOT the volume's!!!
        dtheta_xy_mu = z1 * theta0
        dxy_mu = flight * torch.sin(theta0) * ((z1 / math.sqrt(12)) + (z2 / 2))

        # We compute dtheta_xy in muon ref frame, but we're free to rotate the muon,
        # since dtheta_xy doesn't depend on muon position
        # Therefore assign theta_x axis (muon ref) to be in the theta direction (vol ref),
        # and theta_y axis (muon ref) to be in the phi direction (vol ref)
        dtheta_vol = dtheta_xy_mu[:, 0]  # dtheta_x in muon ref
        dphi_vol = dtheta_xy_mu[:, 1]  # dtheta_y in muon ref

        # Note that if a track incides on a layer
        # with angle theta_mu, the dx and dy displacements are relative to zero angle
        # (generation of MSC formulas are oblivious of angle of incidence) so we need
        # to rescale them by cos of thetax and thetay
        dxy_vol = dxy_mu * torch.cos(theta_xy)
        return {"dtheta_vol": dtheta_vol, "dphi_vol": dphi_vol, "dx_vol": dxy_vol[:, 0], "dy_vol": dxy_vol[:, 1]}

    def _compute_scattering(self, *, x0: Tensor, deltaz: Union[Tensor, float], theta_xy: Tensor, mom: Tensor) -> Dict[str, Tensor]:
        if self.scatter_model == "pdg":
            return self._pdg_scatter(x0=x0, deltaz=deltaz, theta_xy=theta_xy, mom=mom)
        elif self.scatter_model == "geant4":
            return self._geant_scatter(x0=x0, deltaz=deltaz, theta_xy=theta_xy, mom=mom)
        else:
            raise ValueError(f"Scatter model {self.scatter_model} is not currently supported.")

    def scatter_and_propagate(self, mu: MuonBatch, deltaz: Union[Tensor, float]) -> None:
        """
        This function produces a model of multiple scattering through a layer of material
        of depth deltaz

        TODO: Expand to sum over traversed voxels
        """

        if self.rad_length is not None:
            mask = mu.get_xy_mask((0, 0), self.lw)  # Only scatter muons inside volume
            xy_idx = self.mu_abs2idx(mu, mask)

            x0 = self.rad_length[xy_idx[:, 0], xy_idx[:, 1]][:, None]
            scatterings = self._compute_scattering(x0=x0, deltaz=deltaz, theta_xy=mu.theta_xy[mask], mom=mu.mom[mask][:, None])

            # Update to position at scattering.
            mu.scatter_dxy(dx_vol=scatterings["dx_vol"], dy_vol=scatterings["dy_vol"], mask=mask)
            mu.propagate(deltaz)
            mu.scatter_dtheta_dphi(dtheta_vol=scatterings["dtheta_vol"], dphi_vol=scatterings["dphi_vol"], mask=mask)
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
    def __init__(
        self,
        lw: Tensor,
        z: float,
        size: float,
        rad_length_func: Optional[Callable[..., Tensor]] = None,
        dz_step: float = 0.05,
        scatter_model: str = "pdg",
        device: torch.device = DEVICE,
    ):
        super().__init__(lw=lw, z=z, size=size, device=device)
        self.dz_step = dz_step
        self.n_steps = int(np.round(self.size / self.dz_step))
        self.scatter_model = scatter_model
        if rad_length_func is not None:
            self.load_rad_length(rad_length_func)

    def __repr__(self) -> str:
        return f"""PassiveLayer located at z={self.z}"""

    def load_rad_length(self, rad_length_func: Callable[..., Tensor]) -> None:
        self.rad_length = rad_length_func(z=self.z, lw=self.lw, size=self.size).to(self.device)

    def forward(self, mu: MuonBatch) -> None:
        if self.scatter_model == "geant4":
            if not SCATTER_MODEL.initialised:
                SCATTER_MODEL.load_data()  # Delay loading until requrired
            n = int(self.size / SCATTER_MODEL.deltaz)
            dz = SCATTER_MODEL.deltaz
        elif self.scatter_model == "pdg":
            dz, n = self.dz_step, self.n_steps
        else:
            raise ValueError(f"Scatter model {self.scatter_model} is not currently supported.")

        for _ in range(n):
            self.scatter_and_propagate(mu, deltaz=dz)
        mu.propagate(mu.z - (self.z - self.size))  # In case of floating point-precision, ensure muons are at the bottom of the layer


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
            i = 0
            for _, p in self.yield_zordered_panels():  # This really should be an enumerate, but MyPy then thinks assign_budget is a Tensor...
                p.assign_budget(budget[i])
                i += 1
