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

r"""
Provides implementations of the layers in z, which are used to construct volumes, both the passive scattering layers, and the active detection layers.
"""

__all__ = ["PassiveLayer", "PanelDetectorLayer"]


class Layer(nn.Module):
    r"""
    Abstract base class forr volume layers.
    The length and width (lw) is the spans of the layer in metres in x and y, and the layer begins at x=0, y=0.
    z indicates the position of the top of the layer, in meters, and size is the distance from the top of the layer to the bottom.
    size is also used to set the lenght, width, and height of the voxels that make up the layer.

    ..important::
        Users must ensure that both the length and width of the layer are divisible by size

    If the layer is set to scatter muons (`rad_length` is not None), then two scattering models are available:
        'pdg': The default and currently recommended model based on the Gaussian scattering model described in
            https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf
        'pgeant': An under-development model based on a parameterised fit to data sampled from GEANT 4
    """

    def __init__(self, lw: Tensor, z: float, size: float, scatter_model: str = "pdg", device: torch.device = DEVICE):
        r"""
        Arguments:
            lw: the length and width of the layer in the x and y axes in metres, starting from (x,y)=(0,0).
            z: the z position of the top of layer in metres. The bottom of the layer will be located at z-size
            size: the voxel size in metres. Must be such that lw is divisible by the specified size.
            scatter_model: String selection for the scattering model to use. Currently either 'pdg' or 'pgeant'.
            device: device on which to place tensors
        """

        super().__init__()
        self.lw, self.z, self.size, self.scatter_model, self.device = (
            lw.to(device),
            torch.tensor([z], dtype=torch.float32, device=device),
            size,
            scatter_model,
            device,
        )
        self.rad_length: Optional[Tensor] = None

    def _pgeant_scatter(self, *, x0: Tensor, deltaz: Union[Tensor, float], theta: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the parametersised GEANT 4 model.

        Arguments:
            x0: (N) tensor of the X0 of the voxel each muon is traversing
            deltaz: The amound of distance the muons will travel in the z direction in metres
            theta: (N) tensor of the theta angles of the muons. This is used to compute the total flight path of the muons
            theta_x: (N) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frame to the volume's
            theta_y: (N) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frame to the volume's
            mom: (N) tensor of the absolute value of the momentum of each muon

        Retruns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_vol, dphi_vol, dx_vol, & dy_vol
        """

        return SCATTER_MODEL.compute_scattering(x0=x0, deltaz=deltaz, theta=theta, theta_x=theta_x, theta_y=theta_y, mom=mom)

    def _pdg_scatter(
        self, *, x0: Tensor, deltaz: Union[Tensor, float], theta: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor, log_term: bool = True
    ) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the PDG model https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf

        Arguments:
            x0: (N) tensor of the X0 of the voxel each muon is traversing
            deltaz: The amound of distance the muons will travel in the z direction in metres
            theta: (N) tensor of the theta angles of the muons. This is used to compute teh total flight path of the muons
            theta_x: (N) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frames to the volume's
            theta_y: (N) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frames to the volume's
            mom: (N) tensor of the absolute value of the momentum of each muon

        Retruns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_vol, dphi_vol, dx_vol, & dy_vol
        """

        flight = deltaz / torch.cos(theta)
        n_x0 = flight / x0

        n = len(n_x0)
        z1 = torch.randn((2, n), device=self.device)
        z2 = torch.randn((2, n), device=self.device)
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
        dtheta_vol = dtheta_xy_mu[0]  # dtheta_x in muon ref
        dphi_vol = dtheta_xy_mu[1]  # dtheta_y in muon ref

        # Note that if a track incides on a layer
        # with angle theta_mu, the dx and dy displacements are relative to zero angle
        # (generation of MSC formulas are oblivious of angle of incidence) so we need
        # to rescale them by cos of thetax and thetay
        dx_vol = dxy_mu[0] * torch.cos(theta_x)
        dy_vol = dxy_mu[1] * torch.cos(theta_y)
        return {"dtheta_vol": dtheta_vol, "dphi_vol": dphi_vol, "dx_vol": dx_vol, "dy_vol": dy_vol}

    def _compute_scattering(
        self, *, x0: Tensor, deltaz: Union[Tensor, float], theta: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor
    ) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the chosen model

        Arguments:
            x0: (N) tensor of the X0 of the voxel each muon is traversing
            deltaz: The amound of distance the muons will travel in the z direction in metres
            theta: (N) tensor of the theta angles of the muons. This is used to compute teh total flight path of the muons
            theta_x: (N) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frames to the volume's
            theta_y: (N) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frames to the volume's
            mom: (N) tensor of the absolute value of the momentum of each muon

        Retruns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_vol, dphi_vol, dx_vol, & dy_vol
        """
        if self.scatter_model == "pdg":
            return self._pdg_scatter(x0=x0, deltaz=deltaz, theta=theta, theta_x=theta_x, theta_y=theta_y, mom=mom)
        elif self.scatter_model == "pgeant":
            return self._pgeant_scatter(x0=x0, deltaz=deltaz, theta=theta, theta_x=theta_x, theta_y=theta_y, mom=mom)
        else:
            raise ValueError(f"Scatter model {self.scatter_model} is not currently supported.")

    def scatter_and_propagate(self, mu: MuonBatch, deltaz: Union[Tensor, float]) -> None:
        r"""
        Propagates the muons through (part of) the layer, such that afterwards all the muons are deltaz lower than their starting position.
        If the layer is set to scatter muons (`rad_length` is not None),
        then the muons will also undergo scattering (changes in their trajectories and positions) according to the scatter model of the layer.

        .. warning::
            When computing scatterings, the X0 used for eaach muon is that of the starting voxel:
            If a muon moves into a neighbouring voxel of differing X0, then this will only be accounted for in the next deltaz step.

        Arguments:
            mu: muons to propagate
            deltaz: amount of distance in metres in the negative z direction that the muons shoudl travel (positive number lowers the muon position)
        """

        if self.rad_length is not None:
            mask = mu.get_xy_mask((0, 0), self.lw)  # Only scatter muons inside volume
            xy_idx = self.mu_abs2idx(mu, mask)

            x0 = self.rad_length[xy_idx[:, 0], xy_idx[:, 1]]
            scatterings = self._compute_scattering(
                x0=x0, deltaz=deltaz, theta=mu.theta[mask], theta_x=mu.theta_x[mask], theta_y=mu.theta_y[mask], mom=mu.mom[mask]
            )

            # Update to position at scattering.
            mu.scatter_dxy(dx_vol=scatterings["dx_vol"], dy_vol=scatterings["dy_vol"], mask=mask)
            mu.propagate(deltaz)
            mu.scatter_dtheta_dphi(dtheta_vol=scatterings["dtheta_vol"], dphi_vol=scatterings["dphi_vol"], mask=mask)
        else:
            mu.propagate(deltaz)

    def mu_abs2idx(self, mu: MuonBatch, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Helper method to return the voxel indices in the layer that muons currently occupy.

        .. warning::
            This method does NOT account for the possibility of muons being outside the layer.
            Please also supply a mask, to only select muons inside the layer.

        Arguments:
            mu: muons to look up
            mask: Optional (N) Boolean tensor where True indicates that the muon position should be checked

        Returns:
            (N,2) tensor of voxel indices in x,y
        """

        xy = mu.xy
        if mask is not None:
            xy = xy[mask]
        return self.abs2idx(xy)

    def abs2idx(self, xy: Tensor) -> Tensor:
        r"""
        Helper method to return the voxel indices in the layer of the supplied tensor of xy positions.

        .. warning::
            This method does NOT account for the possibility of positions may be outside the layer.
            Please ensure that positions are inside the layer.

        Arguments:
            xy: (N,2) tensor of absolute xy postions in metres in the volume frame

        Returns:
            (N,2) tensor of voxel indices in x,y
        """

        return torch.floor(xy / self.size).long()

    @abstractmethod
    def forward(self, mu: MuonBatch) -> None:
        r"""
        Inheriting classes should override this method to implement the passage of the muons through the layer.

        Arguments:
            mu: the incoming batch of muons
        """

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
        if self.scatter_model == "pgeant":
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
