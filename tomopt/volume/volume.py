from typing import Tuple, List, Optional, Any
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layer import AbsDetectorLayer, AbsLayer, PassiveLayer, PanelDetectorLayer
from .panel import DetectorPanel
from .heatmap import DetectorHeatMap
from ..muon import MuonBatch
from ..core import RadLengthFunc

r"""
Provides implementation of wrapper classes for containing multiple passive layers and detector layers, which act as interfaces to them.
"""

__all__ = ["Volume"]


class Volume(nn.Module):
    r"""
    The `Volume` class is used to contain both passive layers and detector layers.
    It is designed to act as an interface to them for the convenience of e.g. :class:`~tomopt.optimisation.wrapper.volume_wrapper.VolumeWrapper`,
    and to allow new passive-volume layouts to be loaded.

    When optimisation is acting in `fixed-budget` mode, the volume is also responsible for learning the optimal assignments of the budget to detector parts.

    Volumes can also have a "target" value. This could be e.g. the class ID of the passive-volume configuration which is currently loaded.
    See e.g. :class:`~tomopt.optimisation.loss.VolumeClassLoss`.
    The target can be set as part of the call to :meth:`~tomopt.volume.volume.Volume.load_rad_length`

    The volume is expected to have its low-left-front (zxy) corner located at (0,0,0) metres.

    .. important::
        Currently this class expects that all :class:`~tomopt.volume.layer.PassiveLayer`s form a single contiguous block,
        i.e. it does not currently support sparse, or multiple, passive volumes.

    Arguments:
        layers: `torch.nn.ModuleList` of instantiated :class:`~tomopt.volume.layer.AbsLayer`s, ordered in decreasing z position.
        budget: optional budget of the detector in currency units.
            Supplying a value for the optional budget, here, will prepare the volume to learn budget assignments to the detectors,
            and configure the detectors for the budget.
    """

    def __init__(self, layers: nn.ModuleList, budget: Optional[float] = None):
        r"""
        Initialises the volume with the set of layers (both detector and passive),
        which should be supplied as a `torch.nn.ModuleList` ordered in decreasing z position.
        Supplying a value for the optional budget, here, will prepare the volume to learn budget assignments to the detectors,
        and configure the detectors for the budget.
        """

        super().__init__()
        self.layers = layers
        self._device = self._get_device()
        self.budget = None if budget is None else torch.tensor(budget, device=self._device)
        self._check_passives()
        self._target: Optional[Tensor] = None
        self._edges: Optional[Tensor] = None

        if self.budget is not None:
            self._configure_budget()

    def __getitem__(self, idx: int) -> AbsLayer:
        return self.layers[idx]

    def get_passive_z_range(self) -> Tuple[Tensor, Tensor]:
        r"""
        Returns:
            The z position of the bottom of the lowest passive layer, and the z position of the top of the highest passive layer.
        """

        ps = self.get_passives()
        return ps[-1].z - self.passive_size, ps[0].z

    def build_edges(self) -> Tensor:
        r"""
        Computes the zxy locations of low-left-front edges of voxels in the passive layers of the volume.
        """

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
        return torch.tensor(
            bounds.reshape(3, -1).transpose(-1, -2), dtype=torch.float32, device=self.device
        )  # TODO: Check that xyz shape is expected, and not zxy

    def get_detectors(self) -> List[AbsDetectorLayer]:
        r"""
        Returns:
            A list of all :class:`~tomopt.volume.layer.AbsDetectorLayer`s in the volume, in the order of `layers` (normally decreasing z position)
        """

        return [l for l in self.layers if isinstance(l, AbsDetectorLayer)]

    def get_passives(self) -> List[PassiveLayer]:
        r"""
        Returns:
            A list of all :class:`~tomopt.volume.layer.PassiveLayer`s in the volume, in the order of `layers` (normally decreasing z position)
        """

        return [l for l in self.layers if isinstance(l, PassiveLayer)]

    def get_rad_cube(self) -> Tensor:
        r"""
        Returns:
            zxy tensor of the values stored in the voxels of the passive volume, with the lowest layer being found in the zeroth z index position.
        """

        vols = list(reversed(self.get_passives()))  # reversed to match lookup_xyz_coords: layer zero = bottom layer
        if len(vols) == 0:
            raise ValueError("self.layers contains no passive layers")
        rads = [v.rad_length for v in vols if v.rad_length is not None]
        if len(rads) > 0:
            return torch.stack([v.rad_length for v in vols if v.rad_length is not None], dim=0)
        else:
            raise AttributeError("None of volume layers have a non-None rad_length attribute")

    def lookup_passive_xyz_coords(self, xyz: Tensor) -> Tensor:
        r"""
        Looks up the voxel indices of the supplied list of absolute positions in the volume frame

        .. warning::
            Assumes the same size for all passive layers, and that they form a single contiguous block

        Arguments:
            xyz: an (N,xyz) tensor of absolute positions in the volume frame

        Returns:
            an (N,xyz) tensor of zero-ordered voxel indices, which correspond to the supplied positions
        """

        if len(xyz.shape) == 1:
            xyz = xyz[None, :]

        if n := (
            ((xyz[:, :2] > self.lw) + (xyz[:, :2] < 0)).sum(1) + (xyz[:, 2] < self.get_passive_z_range()[0]) + ((xyz[:, 2] > self.get_passive_z_range()[1]))
        ).sum():
            raise ValueError(f"{n} coordinate(s) outside passive volume")
        xyz[:, 2] = xyz[:, 2] - self.get_passive_z_range()[0]
        return torch.floor(xyz / self.passive_size).long()

    def load_rad_length(self, rad_length_func: RadLengthFunc, target: Optional[Tensor] = None) -> None:
        r"""
        Loads a new passive-volume configuration.
        Optionally, a "target" for the configuration may also be supplied.
        This could be e.g. the class ID of the passive-volume configuration which is currently loaded.
        See e.g. :class:`~tomopt.optimisation.loss.VolumeClassLoss`.

        Arguments:
            rad_length_func: lookup function that returns an (n_x,n_y) tensor of voxel X0 values for the layer.
            target: optional target for the new layout
        """

        self._target = target
        for p in self.get_passives():
            p.load_rad_length(rad_length_func)

    def assign_budget(self) -> None:
        r"""
        Distributed the total budget for the detector system amongst the various sub-detectors.
        When assigning budgets to layers, the budget weights are softmax-normalised to one, and multiplied by the total budget.
        Slices of these budgets are then passed to the layers, with the length of the slices being taken from `_n_layer_costs`.
        """

        if self.budget is not None:
            budget_idx, layer_idx = 0, 0
            layer_budgets = self.budget * F.softmax(self.budget_weights, dim=-1)
            for l in self.layers:
                if self.budget is not None and hasattr(l, "get_cost"):
                    n = self._n_layer_costs[layer_idx]
                    l.assign_budget(layer_budgets[budget_idx : budget_idx + n])
                    budget_idx += n
                    layer_idx += 1

    def forward(self, mu: MuonBatch) -> None:
        r"""
        Propagates muons through each layer in turn.
        Prior to propagating muons, the :meth:`~tomopt.volume.volume.Volume.assign_budget` method is called.

        Arguments:
            mu: the incoming batch of muons
        """

        self.assign_budget()

        for l in self.layers:
            l(mu)
            mu.snapshot_xyz()

    def get_cost(self) -> Tensor:
        r"""
        Returns:
            The total, current cost of the layers in the volume,
                or the assigned budget for the volume (these two values should be the same but, the actual cost won't be evaluated explicitly)
        """

        cost = None
        if self.budget is not None:
            return self.budget
        for i, l in enumerate(self.layers):
            if hasattr(l, "get_cost"):
                c = l.get_cost()
                if cost is None:
                    cost = c
                else:
                    cost = cost + c
        if cost is None:
            cost = torch.zeros((1), device=self.device)
        return cost

    def draw(self, xlim: Tuple[float, float] = (-1, 2), ylim: Tuple[float, float] = (-1, 2), zlim: Tuple[float, float] = (0, 1.2)) -> None:
        r"""
        Draws the layers/panels pertaining to the volume.
        When using this in a jupyter notebook, use "%matplotlib notebook" to have an interactive plot that you can rotate.

        Arguments:
            The axis range in x, y, and z for the three-dimensional plot.
                Defaults are based on examples/panel_detectors/00_Hello_World.ipynb, user needs to tweak them as needed
        """
        ax = plt.figure(figsize=(9, 9)).add_subplot(projection="3d")
        ax.computed_zorder = False
        # TODO: find a way to fix transparency overlap in order to have passive layers in front of bottom active layers.
        passivearrays: List[Any] = []
        activearrays: List[Any] = []

        for layer in self.layers:
            # fmt: off
            if isinstance(layer, PassiveLayer):
                lw, thez, size = layer.get_lw_z_size()
                roundedz = np.round(thez.item(), 2)
                # TODO: split these to allow for different alpha values (want: more transparent in front, more opaque in the back)
                rect = [
                    [
                        (0, 0, roundedz - size),
                        (0 + lw[0].item(), 0, roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                        (0, 0 + lw[1].item(), roundedz - size)
                    ],
                    [
                        (0, 0, roundedz - size),
                        (0 + lw[0].item(), 0, roundedz - size),
                        (0 + lw[0].item(), 0, roundedz),
                        (0, 0, roundedz)
                    ],
                    [
                        (0, 0 + lw[1].item(), roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                        (0, 0 + lw[1].item(), roundedz)
                    ],
                    [
                        (0, 0, roundedz - size),
                        (0, 0 + lw[1].item(), roundedz - size),
                        (0, 0 + lw[1].item(), roundedz),
                        (0, 0, roundedz)
                    ],
                    [
                        (0 + lw[0].item(), 0, roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                        (0 + lw[0].item(), 0, roundedz)
                    ],
                ]

                col = "red" if isinstance(layer, DetectorPanel) else ("blue" if isinstance(layer, PassiveLayer) else "black")

                passivearrays.append([rect, col, roundedz, 1])
                continue
            # if not passive layer...
            if isinstance(layer, PanelDetectorLayer):
                for i, p in layer.yield_zordered_panels():
                    if isinstance(p, DetectorHeatMap):
                        raise TypeError("Drawing not supported yet for DetectorHeatMap panels")
                    col = "red" if isinstance(p, DetectorPanel) else ("blue" if isinstance(p, PassiveLayer) else "black")
                    if not isinstance(p.xy, Tensor):
                        raise ValueError("Panel xy is not a tensor, for some reason")
                    if not isinstance(p.z, Tensor):
                        raise ValueError("Panel z is not a tensor, for some reason")
                    rect = [[
                        (p.xy.data[0].item() - p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() - p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item()),
                        (p.xy.data[0].item() + p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() - p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item()),
                        (p.xy.data[0].item() + p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() + p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item()),
                        (p.xy.data[0].item() - p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() + p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item())
                    ]]

                    activearrays.append([rect, col, p.z.data[0].item(), 0.2])
            else:
                raise TypeError("Volume.draw does not yet support layers of type", type(layer))
            # fmt: on

        allarrays = activearrays + passivearrays
        allarrays.sort(key=lambda x: x[2])

        # fmt: off
        for voxelandcolour in allarrays:
            ax.add_collection3d(Poly3DCollection(voxelandcolour[0], facecolors=voxelandcolour[1], linewidths=1, edgecolors=voxelandcolour[1], alpha=voxelandcolour[3],
                                                 zorder=voxelandcolour[2], sort_zpos=voxelandcolour[2]))
        # fmt: on
        plt.ylim(xlim)
        plt.xlim(ylim)
        ax.set_zlim(zlim)
        plt.title("Volume layers")
        red_patch = mpatches.Patch(color="red", label="Active Detector Layers")
        pink_patch = mpatches.Patch(color="pink", label="Passive Layers")
        ax.legend(handles=[red_patch, pink_patch])
        plt.show()

    def _configure_budget(self) -> None:
        r"""
        Creates a list of learnable parameters, which acts as the fractional assignment of the total budget to various parts of the detectors.
        The `budget_weights` contains all these assignments with no explicit hierarchy.

        Ordering of the elements is thus:
            Each layer, as ordered in `layers` is checked for a `get_cost` attribute.
                If the layer has this attribute, then the number of costs that that layer has `layer._n_costs` is appended to a list, `_n_layer_costs`
            A tensor, `budget_weights`, is then instantiated with a number of zero-valued elements equal to the total number of individual detector costs

        When assigning budgets to layers, the budget weights are softmax-normalised to one, and multiplied by the total budget.
        Slices of these budgets are then passed to the layers, with the length of the slices being taken from `_n_layer_costs`.

        After the `budget_weights` are initialised, the :meth:`~tomopt.volume.volume.Volume.assign_budget` is called automatically.
        """

        self._n_layer_costs = [l._n_costs for l in self.layers if hasattr(l, "get_cost")]  # Number of different costs in the detector layer
        self.budget_weights = nn.Parameter(torch.zeros(np.sum(self._n_layer_costs), device=self._device))  # Assignment of budget amongst all costs
        self.assign_budget()

    def _get_device(self) -> torch.device:
        device = self.layers[0].device
        if len(self.layers) > 1:
            for l in self.layers[1:]:
                if l.device != device:
                    raise ValueError("All layers must use the same device, but found multiple devices")
        return device

    def _check_passives(self) -> None:
        r"""
        Ensures that all :class:`~tomopt.volume.layer.PassiveLayer`s have the same sizes
        """

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
    def lw(self) -> Tensor:
        r"""
        Returns:
            The length and width of the passive volume
        """

        return self.get_passives()[-1].lw  # Same LW for each passive layer

    @property
    def passive_size(self) -> float:
        r"""
        Returns:
            The size of voxels in the passive volume
        """

        return self.get_passives()[-1].size  # Same size for each passive layer

    @property
    def h(self) -> Tensor:
        r"""
        Returns:
            The height of the volume (including both passive and detector layers), as computed from the z position of the zeroth layer.
        """

        return self.layers[0].z

    @property
    def edges(self) -> Tensor:
        r"""
        zxy locations of low-left-front edges of voxels in the passive layers of the volume.
        """

        if self._edges is None:
            self._edges = self.build_edges()
        return self._edges

    @property
    def centres(self) -> Tensor:
        r"""
        zxy locations of the centres of voxels in the passive layers of the volume.
        """

        if self._edges is None:
            self._edges = self.build_edges()
        return self._edges + (self.passive_size / 2)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def target(self) -> Optional[Tensor]:
        r"""
        Returns:
            The "target" value of the volume. This could be e.g. the class ID of the passive-volume configuration which is currently loaded.
            See e.g. :class:`~tomopt.optimisation.loss.VolumeClassLoss`.
            The target can be set as part of the call to :meth:`~tomopt.volume.volume.Volume.load_rad_length`
        """

        return self._target
