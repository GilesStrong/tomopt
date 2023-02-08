from typing import Iterator, Optional, Dict, List, Union, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
import math

import torch
from torch import nn, Tensor

from .scatter_model import PGEANT_SCATTER_MODEL
from ..core import DEVICE, SCATTER_COEF_A, SCATTER_COEF_B, RadLengthFunc
from ..muon import MuonBatch
from tomopt.volume.panel import DetectorPanel
from tomopt.volume.heatmap import DetectorHeatMap

import matplotlib.pyplot as plt

def check_muon_batch(mu:MuonBatch,figtitle:str,mask=None)->None:
    if mask is None:
        mask = torch.ones_like(mu.theta,dtype=bool)
        isnone=True

    fig,ax = plt.subplots(ncols=2,nrows=3,figsize=(8,10))
    fig.suptitle(figtitle)
    ax = ax.ravel()
    ax[0].hist(mu.x[mask],bins=100,label='in propagation',alpha=.3)
    ax[0].hist(mu.x[mask==False],bins=100,label='at rest',alpha=.3)
    ax[0].set_xlabel('x')
    ax[0].legend()

    ax[1].hist(mu.y[mask],bins=100,label='in propagation',alpha=.3)
    ax[1].hist(mu.y[mask==False],bins=100,label='at rest',alpha=.3)
    ax[1].set_xlabel('y')
    ax[1].legend()

    ax[2].hist(mu.theta_x[mask],bins=100,label='in propagation',alpha=.3)
    ax[2].hist(mu.theta_x[mask==False],bins=100,label='at rest',alpha=.3)
    ax[2].set_xlabel('tethat_x')
    ax[2].legend()

    ax[3].hist(mu.theta_y[mask],bins=100,label='in propagation',alpha=.3)
    ax[3].hist(mu.theta_y[mask==False],bins=100,label='at rest',alpha=.3)
    ax[3].set_xlabel('tethat_y')
    ax[3].legend()

    ax[4].hist(mu.z[mask],bins=100,label='in propagation',alpha=.3)
    ax[4].hist(mu.z[mask==False],bins=100,label='at rest',alpha=.3)
    ax[4].set_xlabel('z')
    ax[4].legend()

    ax[5].hist(mu.phi[mask],bins=100,label='in propagation',alpha=.3)
    ax[5].hist(mu.phi[mask==False],bins=100,label='at rest',alpha=.3)
    ax[5].set_xlabel('phi')
    ax[5].legend()
    plt.tight_layout()
    plt.show()

r"""
Provides implementations of the layers in z, which are used to construct volumes, both the passive scattering layers, and the active detection layers.
"""

__all__ = ["AbsLayer", "PassiveLayer", "AbsDetectorLayer", "PanelDetectorLayer"]


class AbsLayer(nn.Module, metaclass=ABCMeta):
    r"""
    Abstract base class for volume layers.
    The length and width (`lw`) is the spans of the layer in metres in x and y, and the layer begins at x=0, y=0.
    z indicates the position of the top of the layer, in meters, and size is the distance from the top of the layer to the bottom.
    size is also used to set the length, width, and height of the voxels that make up the layer.

    .. important::
        Users must ensure that both the length and width of the layer are divisible by size

    Arguments:
        lw: the length and width of the layer in the x and y axes in metres, starting from (x,y)=(0,0).
        z: the z position of the top of layer in metres. The bottom of the layer will be located at z-size
        size: the voxel size in metres. Must be such that lw is divisible by the specified size.
        device: device on which to place tensors
    """

    def __init__(self, lw: Tensor, z: float, size: float, device: torch.device = DEVICE):
        super().__init__()
        self.lw, self.z, self.size, self.device = (
            lw.to(device),
            torch.tensor([z], dtype=torch.float32, device=device),
            size,
            device,
        )
        self.rad_length: Optional[Tensor] = None

    @abstractmethod
    def forward(self, mu: MuonBatch) -> None:
        r"""
        Inheriting classes should override this method to implement the passage of the muons through the layer.

        Arguments:
            mu: the incoming batch of muons
        """

        pass

    def get_lw_z_size(self) -> Tuple[Tensor, Tensor, float]:
        r"""
        Returns:
            The length and width of the layer in the x and y axes in metres, starting from (x,y)=(0,0), the z position of the top of layer in metres, and the voxel size in metres.
        """
        return self.lw, self.z, self.size


class PassiveLayer(AbsLayer):
    r"""
    Default layer of containing passive material that scatters the muons.
    The length and width (`lw`) is the spans of the layer in metres in x and y, and the layer begins at x=0, y=0.
    z indicates the position of the top of the layer, in meters, and size is the distance from the top of the layer to the bottom.
    size is also used to set the length, width, and height of the voxels that make up the layer.

    .. important::
        Users must ensure that both the length and width of the layer are divisible by size

    If the layer is set to scatter muons (`rad_length` is not None), then two scattering models are available:
        - 'pdg': The default and currently recommended model based on the Gaussian scattering model described in https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf
        - 'pgeant': An under-development model based on a parameterised fit to data sampled from GEANT 4

    The X0 values of each voxel is defined via a "radiation-length function", which should return an (n_x,n_y) tensor of voxel X0 values,
    when called with the `z`, `lw`, and `size` of the layer. For example:

    .. code-block:: python

        def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
            rad_length = torch.ones(list((lw / size).long())) * X0["lead"]
            if z < 0.5:
                rad_length[...] = X0["beryllium"]
            return rad_length

    This function can either be supplied during initialisation, or later via the `load_rad_length` method.

    Arguments:
        lw: the length and width of the layer in the x and y axes in metres, starting from (x,y)=(0,0).
        z: the z position of the top of layer in metres. The bottom of the layer will be located at z-size
        size: the voxel size in metres. Must be such that lw is divisible by the specified size.
        rad_length_func: lookup function that returns an (n_x,n_y) tensor of voxel X0 values for the layer.
            After initialisation, the `load_rad_length` method may be used to load X0 layouts.
        step_sz: The step size in metres over which to compute muon propagation and scattering.
        scatter_model: String selection for the scattering model to use. Currently either 'pdg' or 'pgeant'.
        device: device on which to place tensors
    """

    def __init__(
        self,
        lw: Tensor,
        z: float,
        size: float,
        rad_length_func: Optional[RadLengthFunc] = None,
        step_sz: float = 0.01,
        scatter_model: str = "pdg",
        device: torch.device = DEVICE,
    ):
        super().__init__(lw=lw, z=z, size=size, device=device)
        self.step_sz = step_sz
        self.scatter_model = scatter_model
        if rad_length_func is not None:
            self.load_rad_length(rad_length_func)

    def __repr__(self) -> str:
        return f"""PassiveLayer located at z={self.z}"""

    def load_rad_length(self, rad_length_func: RadLengthFunc) -> None:
        r"""
        Loads a new X0 layout into the layer voxels.

        Arguments:
            rad_length_func: lookup function that returns an (n_x,n_y) tensor of voxel X0 values for the layer.
        """

        self.rad_length = rad_length_func(z=self.z, lw=self.lw, size=self.size).to(self.device)

    def forward(self, mu: MuonBatch) -> None:
        r"""
        Propagates the muons through the layer to the bottom in a series of scattering steps.
        If the 'pdg' model is used, then the step size is the `step_sz` of the layer, as supplied during initialisation.
        If the 'pgeant' model is used, the the step size specified as part of the fitting of the scattering model.

        Arguments:
            mu: the incoming batch of muons
        """

        mu.propagate_dz(mu.z - self.z)  # Move muons to the top of the layer
        mask = torch.ones(len(mu), device=self.device).bool()
        while mask.any():
            self.scatter_and_propagate(mu, mask=mask)
            mask = (mu.z > (self.z - self.size)) & (mu.z <= self.z)  # Only scatter/propagate muons inside the layer
        mu.propagate_dz(mu.z - (self.z - self.size))  # Ensure muons are at the bottom of the layer

    def scatter_and_propagate(self, mu: MuonBatch, mask: Optional[Tensor] = None) -> None:
        r"""
        Propagates the muons through (part of) the layer by the prespecified `step_sz`.
        If the layer is set to scatter muons (`rad_length` is not None),
        then the muons will also undergo scattering (changes in their trajectories and positions) according to the scatter model of the layer.

        .. warning::
            When computing scatterings, the X0 used for each muon is that of the starting voxel:
            If a muon moves into a neighbouring voxel of differing X0, then this will only be accounted for in the next step.

        Arguments:
            mu: muons to propagate
            mask: Optional (N,) Boolean mask. Only muons with True values will be scattered and propagated
        """

        if self.rad_length is not None:
            scatter_mask = mu.get_xy_mask((0, 0), self.lw) & mask  # Only scatter muons inside volume

            xy_idx = self.mu_abs2idx(mu, scatter_mask)
            x0 = self.rad_length[xy_idx[:, 0], xy_idx[:, 1]]

            # Ensure that scattering steps don't extend outside the layer
            # TODO extend this to consider transverse voxel boundaries
            step_sz = torch.ones_like(x0) * self.step_sz
            dz = mu.z - (self.z - self.size)
            r_out = dz[scatter_mask] / mu.theta[scatter_mask].cos()
            m = r_out < step_sz
            step_sz[m] = r_out[m]

            scatterings = self._compute_scattering(
                x0=x0,
                theta=mu.theta[scatter_mask],
                phi=mu.phi[scatter_mask],
                theta_x=mu.theta_x[scatter_mask],
                theta_y=mu.theta_y[scatter_mask],
                mom=mu.mom[scatter_mask],
                step_sz=step_sz,
            )

            # Update to position at scattering.
            mu.scatter_dxyz(dx_vol=scatterings["dx_vol"], dy_vol=scatterings["dy_vol"], dz_vol=scatterings["dz_vol"], mask=scatter_mask)
            mu.propagate_d(self.step_sz, mask)  # Still propagate muons that weren't scattered
            # Muons exiting the layer will be moved back to the bottom of the layer. Perform this here BEFORE their trajectories are updated.
            dz = mu.z - (self.z - self.size)
            exit_mask = (dz < 0) & mask
            mu.propagate_dz(dz[exit_mask], mask=exit_mask)
            mu.scatter_dtheta_xy(dtheta_x_vol=scatterings["dtheta_x_vol"], dtheta_y_vol=scatterings["dtheta_y_vol"], mask=scatter_mask)
        else:
            mu.propagate_d(self.step_sz, mask)

    def mu_abs2idx(self, mu: MuonBatch, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Helper method to return the voxel indices in the layer that muons currently occupy.

        .. warning::
            This method does NOT account for the possibility of muons being outside the layer.
            Please also supply a mask, to only select muons inside the layer.

        Arguments:
            mu: muons to look up
            mask: Optional (muons) Boolean tensor where True indicates that the muon position should be checked

        Returns:
            (muons,2) tensor of voxel indices in x,y
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
            xy: (N,xy) tensor of absolute xy positions in metres in the volume frame

        Returns:
            (N,xy) tensor of voxel indices in x,y
        """

        return torch.floor(xy / self.size).long()

    def _pgeant_scatter(self, *, x0: Tensor, theta: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the parameterised GEANT 4 model.

        Arguments:
            x0: (N,) tensor of the X0 of the voxel each muon is traversing
            theta: (N,) tensor of the theta angles of the muons. This is used to compute the total flight path of the muons
            theta_x: (N,) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frame to the volume's
            theta_y: (N,) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frame to the volume's
            mom: (N,) tensor of the absolute value of the momentum of each muon

        Returns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_vol, dphi_vol, dx_vol, & dy_vol
        """

        return PGEANT_SCATTER_MODEL.compute_scattering(x0=x0, step_sz=self.step_sz, theta=theta, theta_x=theta_x, theta_y=theta_y, mom=mom)

    def _pdg_scatter(
        self, *, x0: Tensor, theta: Tensor, phi: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor, step_sz: Tensor, log_term: bool = True
    ) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the PDG model https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf
        Scattering and displacements are generated in the muon reference frame.
        They are then converted in the volume reference frame using Euler rotation matrices.

        Arguments:
            x0: (N,) tensor of the X0 of the voxel each muon is traversing
            theta: (N,) tensor of the theta angles of the muons. This is used to proceed to the muon -> volume reference frame conversion
            phi: (N,) tensor of the phi angles of the muons. This is used to proceed to the muon -> volume reference frame conversion
            theta_x: (N,) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frames to the volume's
            theta_y: (N,) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frames to the volume's
            mom: (N,) tensor of the absolute value of the momentum of each muon
            step_sz: (N,) tensor of the scattering step length (x distance in the PDG model).

        Returns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_x_vol, dtheta_y_vol, dx_vol, dy_vol and dz_vol
        """
        n_x0 = step_sz / x0

        n = len(n_x0)
        z1 = torch.randn((2, n), device=self.device)
        z2 = torch.randn((2, n), device=self.device)
        theta0 = (SCATTER_COEF_A / mom) * torch.sqrt(n_x0)
        if log_term:
            theta0 = theta0 * (1 + (SCATTER_COEF_B * torch.log(n_x0)))
        # These are in the muons' reference frames NOT the volume's!!!
        # Make sure that scattering angle in the muon reference frame < pi/2
        # to ensure conversion into the volume reference frame
        dtheta_xy_mu = torch.clamp(z1 * theta0, max=math.pi/2.2)
        dxy_mu = step_sz * theta0 * ((z1 / math.sqrt(12)) + (z2 / 2))

        # Note that if a track incides on a layer
        # with angle theta_mu, the dx and dy displacements are relative to to the muon
        # (generation of MSC formulas are oblivious of angle of incidence) so we need
        # to decompose them into displacements in x,y,z in the volume frame
        phi_defined = theta != 0  # If theta is a zero, there is no phi defined
        dx_vol = torch.where(phi_defined, -dxy_mu[0] * torch.sin(phi) - dxy_mu[1] * torch.cos(-theta) * torch.cos(phi), dxy_mu[0])
        dy_vol = torch.where(phi_defined, dxy_mu[0] * torch.cos(phi) - dxy_mu[1] * torch.cos(-theta) * torch.sin(phi), dxy_mu[1])
        dz_vol = torch.where(phi_defined, dxy_mu[1] * torch.sin(-theta), theta.new_zeros(dxy_mu[1].shape))

        # We need to convert deflection in the muon reference frame (dtheta_x_m and dtheta_y_m)
        # into deflection in the volume reference frame (dtheta_x_vol and dtheta_y_vol)
        # In order to do so, we will use equations obtained from Euler rotation matrices where the
        # Muon reference frame (R') is rotated by angle theta (angle between z and z')
        # and phi (x and x' axis) w.r.t the Volume reference frame (R)
        # 1 - Convert the deflection dtheta_xy_muon in the muon reference frame to a x and y (xy_muon) distance in the muon reference fr
        # 2 - We convert that xy_muon distance into the volume reference frame xy_volume
        # 3 - We convert that xy_volume distance into an angular deflection in the volume reference frame theta_x_vol and theta_y_vol
        # Currently, we do not compute a deflection but we compute the final angle the muon will have once it exits the volume
        # define point M used as a reference to compute the original muon direction and the updated one

        dtheta_x_m = dtheta_xy_mu[0]
        dtheta_y_m = dtheta_xy_mu[1]

        ref_point = theta.new_ones([3, len(theta)])
        ref_point[0] = torch.tan(theta_x)
        ref_point[1] = torch.tan(theta_y)

        r = torch.sqrt(ref_point[0]**2+ref_point[1]**2+ref_point[2]**2)
        # 1 -
        dx_m = r * torch.tan(dtheta_x_m)
        dy_m = r * torch.tan(dtheta_y_m)
        # 2 -
        dx_vol_angle = torch.where(phi_defined, -dx_m * torch.sin(phi) - dy_m * torch.cos(theta) * torch.cos(phi), dx_m)
        dy_vol_angle = torch.where(phi_defined, dx_m * torch.cos(phi) - dy_m * torch.cos(theta) * torch.sin(phi), dy_m)
        dz_vol_angle = torch.where(phi_defined, dy_m * torch.sin(theta), torch.zeros_like(dxy_mu[1]))
        # 3 -
        d_out = -ref_point
        d_out[0] = d_out[0] + dx_vol_angle
        d_out[1] = d_out[1] + dy_vol_angle
        d_out[2] = d_out[2] + dz_vol_angle
        dtheta_x_vol = torch.arctan(d_out[0] / d_out[2]) - theta_x
        dtheta_y_vol = torch.arctan(d_out[1] / d_out[2]) - theta_y

        return {
            "dtheta_x_vol": dtheta_x_vol,
            "dtheta_y_vol": dtheta_y_vol,
            "dx_vol": dx_vol,
            "dy_vol": dy_vol,
            "dz_vol": dz_vol,
            "dtheta_x_m": dtheta_x_m,
            "dtheta_y_m": dtheta_y_m,
        }

    def _compute_scattering(
        self, *, x0: Tensor, theta: Tensor, phi: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor, step_sz: Tensor
    ) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the chosen model

        Arguments:
            x0: (N,) tensor of the X0 of the voxel each muon is traversing
            theta: (N,) tensor of the theta angles of the muons. This is used to compute the total flight path of the muons
            theta_x: (N,) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frames to the volume's
            theta_y: (N,) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frames to the volume's
            mom: (N,) tensor of the absolute value of the momentum of each muon

        Returns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_vol, dphi_vol, dx_vol, & dy_vol
        """
        if self.scatter_model == "pdg":
            return self._pdg_scatter(x0=x0, theta=theta, phi=phi, theta_x=theta_x, theta_y=theta_y, mom=mom, step_sz=step_sz)
        elif self.scatter_model == "pgeant":
            return self._pgeant_scatter(x0=x0, theta=theta, theta_x=theta_x, theta_y=theta_y, mom=mom)
        else:
            raise ValueError(f"Scatter model {self.scatter_model} is not currently supported.")


class AbsDetectorLayer(AbsLayer, metaclass=ABCMeta):
    r"""
    Abstract base class for layers designed to record muon positions (hits) using detectors.
    Inheriting classes should override a number methods to do with costs/budgets, and hit recording.

    When optimisation of operating in 'fixed budget' mode, the :class:`~tomopt.volume.volume.Volume` will check the `_n_costs` class attribute of the layer
    and will add this to the total number of learnable budget assignments, and pass that number of budgets as an (_n_costs) tensor.
    By default this is zero, and inheriting classes should set the correct number during initialisation, or via a new default value.

    Some parts of TomOpt act differently on detector layers, according to how the detectors are modelled.
    A `type_label` attribute is used to encode extra information, rather than relying purely on the object-instance type.

    Multiple detection layers can be grouped together, via their `pos` attribute (position); a string-encoded value.
    By default, the inference methods expect detectors above the passive layer to have `pos=='above'`,
    and those below the passive volume to have `pos=='below'`.
    When retrieving hits from the muon batch, hits will be stacked together with other hits from the same `pos`.

    The length and width (`lw`) is the spans of the layer in metres in x and y, and the layer begins at x=0, y=0.
    z indicates the position of the top of the layer, in meters, and size is the distance from the top of the layer to the bottom.

    .. important::
        By default, the detectors will not scatter muons.

    Arguments:
        pos: string-encoding of the detector-layer group
        lw: the length and width of the layer in the x and y axes in metres, starting from (x,y)=(0,0).
        z: the z position of the top of layer in metres. The bottom of the layer will be located at z-size
        size: the voxel size in metres. Must be such that lw is divisible by the specified size.
        device: device on which to place tensors
    """

    _n_costs = 0  # number of budgets that the detector layer requests

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
        r"""
        Inheriting classes should override this method to implement the passage of the muons through the layer,
        and record muon positions (hits) according to the detector model.

        Arguments:
            mu: the incoming batch of muons
        """
        pass

    @abstractmethod
    def get_cost(self) -> Tensor:
        r"""
        Inheriting classes should override this method to return the total, current cost of the detector(s) in the layer.

        Returns:
            Single-element tensor with the current total cost of the detector in the layer.
        """

        pass

    def conform_detector(self) -> None:
        r"""
        Optional method designed to ensure that the detector parameters lie within any require boundaries, etc.
        It will be called via the :class:`~tomopt.optimisation.wrapper.AbsVolumeWrapper` after any update to the detector layers, but by default does nothing.
        """

        pass

    def assign_budget(self, budget: Optional[Tensor]) -> None:
        r"""
        Inheriting classes should override this method to correctly assign elements of an (_n_costs) tensor to the parts of the detector to which they relate.
        All ordering of the tensor is defined using the function,
        but proper optimisation of the budgets may require that the same ordering is used, or that it is deterministic.

        Arguments:
            budget: (_n_costs) tensor of budget assignments in unit currency
        """

        pass


class PanelDetectorLayer(AbsDetectorLayer):
    r"""
    A detector layer class that uses multiple "panels" to record muon positions (hits).
    Currently, two "panel" types are available: :class:`~tomopt.volume.panel.DetectorPanel` and :class:`~tomopt.volume.heatmap.DetectorHeatMap`
    Each detector layer, however, should contain the same type of panel, as this is used to set the `type_label` of the layer.

    When optimisation of operating in 'fixed budget' mode, the :class:`~tomopt.volume.volume.Volume` will check the `_n_costs` class attribute of the layer
    and will add this to the total number of learnable budget assignments, and pass that number of budgets as an (_n_costs) tensor.
    During initialisation, this is set to the number of panels in the layer, at time of initialisation.

    Multiple detection layers can be grouped together, via their `pos` attribute (position); a string-encoded value.
    By default, the inference methods expect detectors above the passive layer to have `pos=='above'`,
    and those below the passive volume to have `pos=='below'`.
    When retrieving hits from the muon batch, hits will be stacked together with other hits from the same `pos`.

    The length and width (`lw`) is the spans of the layer in metres in x and y, and the layer begins at x=0, y=0.
    z indicates the position of the top of the layer, in meters, and size is the distance from the top of the layer to the bottom.

    .. important::
        The detector panels do not scatter muons.

    Arguments:
        pos: string-encoding of the detector-layer group
        lw: the length and width of the layer in the x and y axes in metres, starting from (x,y)=(0,0).
        z: the z position of the top of layer in metres. The bottom of the layer will be located at z-size
        size: the voxel size in metres. Must be such that lw is divisible by the specified size.
        panels: The set of initialised panels to contain in the detector layer
    """

    def __init__(self, pos: str, *, lw: Tensor, z: float, size: float, panels: Union[List[DetectorPanel], List[DetectorHeatMap], nn.ModuleList]):
        if isinstance(panels, list):
            panels = nn.ModuleList(panels)
        super().__init__(pos=pos, lw=lw, z=z, size=size, device=self.get_device(panels))

        self.panels = panels
        if isinstance(panels[0], DetectorHeatMap):
            self.type_label = "heatmap"
            self._n_costs = len(self.panels)

        elif isinstance(panels[0], DetectorPanel):
            self.type_label = "panel"
            self._n_costs = len(self.panels)

    @staticmethod
    def get_device(panels: nn.ModuleList) -> torch.device:
        r"""
        Helper method to ensure that all panels are on the same device, and return that device.
        If not all the panels are on the same device, then an exception will be raised.

        Arguments:
            panels: ModuleLists of either :class:`~tomopt.volume.panel.DetectorPanel` or :class:`~tomopt.volume.heatmap.DetectorHeatMap` objects on device

        Returns:
            Device on which all the panels are.
        """

        device = panels[0].device
        if len(panels) > 1:
            for p in panels[1:]:
                if p.device != device:
                    raise ValueError("All panels must use the same device, but found multiple devices")
        return device

    def get_panel_zorder(self) -> List[int]:
        r"""
        Returns:
            The indices of the panels in order of decreasing z-position.
        """

        return list(np.argsort([p.z.detach().cpu().item() for p in self.panels])[::-1])

    def yield_zordered_panels(self) -> Union[Iterator[Tuple[int, DetectorPanel]], Iterator[Tuple[int, DetectorHeatMap]]]:
        r"""
        Yields the index of the panel, and the panel, in order of decreasing z-position.

        Returns:
            Iterator yielding panel indices and panels in order of decreasing z-position.
        """

        for i in self.get_panel_zorder():
            yield i, self.panels[i]

    def conform_detector(self) -> None:
        r"""
        Loops through panels and calls their `clamp_params` method, to ensure that panels are located within the bounds of the detector layer.
        It will be called via the :class:`~tomopt.optimisation.wrapper.AbsVolumeWrapper` after any update to the detector layers.
        """

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
        r"""
        Propagates muons to each detector panel, in order of decreasing z-position, and calls their `get_hits` method to record hits to the muon batch.
        After this, the muons will be propagated to the bottom of the detector layer.

        Arguments:
            mu: the incoming batch of muons
        """

        for i, p in self.yield_zordered_panels():
            mu.propagate_dz(mu.z - p.z.detach())  # Move to panel
            hits = p.get_hits(mu)
            mu.append_hits(hits, self.pos)
        mu.propagate_dz(mu.z - (self.z - self.size))  # Move to bottom of layer

    def get_cost(self) -> Tensor:
        r"""
        Returns the total, current cost of the detector(s) in the layer, as computed by looping over the panels and summing the returned values of calls to
        their `get_cost` methods.

        Returns:
            Single-element tensor with the current total cost of the detector in the layer.
        """

        cost = None
        for p in self.panels:
            cost = p.get_cost() if cost is None else cost + p.get_cost()
        return cost

    def assign_budget(self, budget: Optional[Tensor]) -> None:
        r"""
        Passes elements of an (_n_costs) tensor to each of the panels' `assign_budget` method.
        Panels are ordered by decreasing z-position, i.e. the zeroth budget element will relate always to the highest panel,
        rather than necessarily to the same panel through the optimisation process

        # TODO investigate whether it would be better to instead assign budgets based on a fixed ordering, rather than the z-order of the panels.

        Arguments:
            budget: (_n_costs) tensor of budget assignments in unit currency
        """
        if budget is not None:
            i = 0
            for _, p in self.yield_zordered_panels():  # This really should be an enumerate, but MyPy then thinks assign_budget is a Tensor...
                p.assign_budget(budget[i])
                i += 1
