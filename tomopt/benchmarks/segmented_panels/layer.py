from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from tomopt.core import DEVICE
from tomopt.muon import MuonBatch
from tomopt.volume.layer import AbsDetectorLayer
from tomopt.volume.panel import DetectorPanel

from .panel import SegmentedSigmoidDetectorPanel

_all__ = ["SegmentedPanelDetectorLayer"]


class SegmentedPanelDetectorLayer(AbsDetectorLayer):
    r"""
    Provides implementation of a detector layer class featuring a segmented
    sigmoid model for resolution and efficiency.
    """

    def __init__(
        self,
        pos: str,
        *,
        lw: Tensor,
        xy: Tensor,
        z: float,
        zlow: float,
        zhigh: float,
        size: float,
        panel_z_spacing: float,
        gap_size: float,
        n_panels: int,
        n_panels_seg: int,
        res: float,
        eff: float,
        smooth: float,
        realistic_validation: bool = True,
        realistic_training: bool = False,
    ):
        super().__init__(pos=pos, lw=lw, z=z, size=size)

        self.gap_size = nn.Parameter(torch.tensor(gap_size, device=self.device))
        self.zlow = z - size
        self.zhigh = z
        self.xy = xy

        # Instantiate panels and ensure they use the shared gap_size
        self.panels = nn.ModuleList(
            [
                SegmentedSigmoidDetectorPanel(
                    init_xyz=(xy[0].cpu().detach().item(), xy[1].cpu().detach().item(), z - (i * panel_z_spacing) / (n_panels - 1)),
                    init_xy_span=(lw[0].cpu().detach().item(), lw[1].cpu().detach().item()),
                    device=DEVICE,
                    res=res,
                    eff=eff,
                    smooth=smooth,
                    n_panels=n_panels_seg,
                    init_gap=self.gap_size,
                    realistic_validation=realistic_validation,
                    realistic_training=realistic_training,
                )
                for i in range(n_panels)
            ]
        )

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

    def yield_zordered_panels(self) -> Union[Iterator[Tuple[int, DetectorPanel]], Iterator[Tuple[int, SegmentedSigmoidDetectorPanel]]]:
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

        self.lw.detach().cpu().numpy()
        self.z.detach().cpu()[0]

        with torch.no_grad():
            # eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            # self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            # self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            # self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            # self.xy_span[0].clamp_(min=xyz_high[0] / 20, max=10 * xyz_high[0])
            # self.xy_span[1].clamp_(min=xyz_high[1] / 20, max=10 * xyz_high[1])
            self.gap_size.clamp_(min=0.01)

            for p in self.panels:
                #     if self.type_label == "heatmap":
                #         xy_low = p.xy_fix[0] - p.range_mult * p.delta_xy
                #         xy_high = p.xy_fix[1] + p.range_mult * p.delta_xy
                #         xy_low = torch.max(torch.tensor(0.0), xy_low)
                #         xy_high = torch.min(torch.tensor(lw[0]), xy_high)

                #         p.clamp_params(
                #             musigz_low=(xy_low, 0.0, z - self.size),
                #             musigz_high=(xy_high, lw[1], z),
                #         )
                #     else:
                p.clamp_params(
                    xyz_low=(-100, -100, self.zlow),
                    xyz_high=(100, 100, self.zhigh),
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
