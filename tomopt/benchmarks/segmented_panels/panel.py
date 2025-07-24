from typing import Optional, Tuple, Union

import numpy as np
import torch  # type: ignore
from torch import Tensor, nn

from tomopt.core import DEVICE
from tomopt.volume.panel import DetectorPanel

r"""
Provides implementation of a detector panel class featuring a segmented
sigmoid model for resolution and efficiency.

This model accounts for resolution drop-off at gaps between panels, and is
useful for simulating panels at the same z position as a two-dimensional array
of panels.
"""

__all__ = ["SegmentedSigmoidDetectorPanel"]


class SegmentedSigmoidDetectorPanel(DetectorPanel):

    def __init__(
        self,
        *,
        n_panels: int,
        smooth: Union[float, Tensor],
        res: float,
        eff: float,
        init_xyz: Tuple[float, float, float],
        init_xy_span: Tuple[float, float],
        init_gap: nn.Parameter,
        m2_cost: float = 1,
        budget: Optional[Tensor] = None,
        realistic_validation: bool = True,
        realistic_training: bool = False,
        device: torch.device = DEVICE,
    ):
        super().__init__(
            res=res,
            eff=eff,
            init_xyz=init_xyz,
            init_xy_span=init_xy_span,
            m2_cost=m2_cost,
            budget=budget,
            realistic_validation=realistic_validation,
            device=device,
        )
        self.n_panels = n_panels  # Number of panels along one axis (n x n)
        self.gap_size = init_gap  # Learnable gap size
        self.realistic_training = realistic_training

        # Smooth will be massaged to Tensor, but MyPy doesn't spot this
        self.smooth = smooth  # type: ignore

    def sig_model(self, xy: Tensor) -> Tensor:
        r"""
        Models fractional resolution and efficiency for an n x n segmented
        detector panel array.
        This accounts for resolution drop-off at gaps between panels.

        Arguments:
            xy: (N, 2) tensor of positions.

        Returns:
            Multiplicative coefficients for the resolution or efficiency of
            the panel array based on the xy position relative to the segmented
            panel positions and their sizes.
        """
        n_panels = self.n_panels  # TODO: Support different numbers of panels in x and y
        self.panel_size = self.get_scaled_xy_span() / n_panels  # Size of each panel
        gap_size = self.gap_size  # Gap between panels

        # Compute effective panel centers
        panel_spacing = self.panel_size + gap_size  # Distance between adjacent panel centers
        panel_indices = torch.arange(n_panels, device=xy.device, dtype=torch.float32) - (n_panels - 1) / 2
        panel_centers_x = self.xy[0] + panel_indices * panel_spacing[0]
        panel_centers_y = self.xy[1] + panel_indices * panel_spacing[1]
        self.panel_centers = torch.stack((panel_centers_x, panel_centers_y), dim=1)

        # Compute resolution coefficient for each panel
        delta = (xy[:, None, :] - self.panel_centers[None, :, :]) / (self.panel_size / 2)
        panel_coef = torch.sigmoid((1 - (torch.sign(delta) * delta)) / self.smooth)
        coef = panel_coef.sum(dim=1)  # Aggregate over all panels

        # Normalize output
        coef = coef / coef.max()

        return coef

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the xy resolutions of panel at the supplied list of xy points.
        If running in evaluation mode with `realistic_validation`,
        then these will be the full resolution of the panel for points inside
        the panel (indicated by the mask), and zero outside.
        Otherwise, the Sigmoid model will be used.

        Arguments:
            xy: (N,xy) tensor of positions
            mask: optional pre-computed (N,) Boolean mask, where True
                  indicates that the xy point is inside the panel.
                  Only used in evaluation mode and if `realistic_validation`
                  is True. If required, but not supplied, than will be computed
                  automatically.

        Returns:
            res, a (N,xy) tensor of the resolution at the xy points
        """

        if not isinstance(self.resolution, Tensor):
            raise ValueError(f"{self.resolution} is not a Tensor for some reason.")  # To appease MyPy
        if self.training or not self.realistic_validation:
            res = self.resolution * self.sig_model(xy)
            res = torch.clamp_min(res, 1e-10)  # To avoid NaN gradients
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            res = torch.zeros((len(xy), 2), device=self.device)  # Zero detection outside detector
            res[mask] = self.resolution
        return res

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the efficiency of panel at the supplied list of xy points.
        If running in evaluation mode with `realistic_validation`,
        then these will be the full efficiency of the panel for points inside the panel (indicated by the mask), and zero outside.
        Otherwise, the Sigmoid model will be used.

        Arguments:
            xy: (N,) or (N,xy) tensor of positions
            mask: optional pre-computed (N,) Boolean mask, where True indicates that the xy point is inside the panel.
                Only used in evaluation mode and if `realistic_validation` is True.
                If required, but not supplied, than will be computed automatically.

        Returns:
            eff, a (N,)tensor of the efficiency at the xy points
        """

        if not isinstance(self.efficiency, Tensor):
            raise ValueError(f"{self.efficiency} is not a Tensor for some reason.")  # To appease MyPy
        if not self.realistic_training and (self.training or not self.realistic_validation):
            eff = self.efficiency * self.sig_model(xy).prod(dim=-1)
            eff = torch.clamp_min(eff, 1e-10)  # To avoid NaN gradients
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            eff = torch.zeros(len(xy), device=self.device)  # Zero detection outside detector
            eff[mask] = self.efficiency
        return eff

    @property
    def smooth(self) -> Tensor:
        return self._smooth

    @smooth.setter
    def smooth(self, smooth: Union[float, Tensor]) -> None:
        if not smooth > 0:
            raise ValueError("smooth argument must be positive and non-zero")
        if not isinstance(smooth, Tensor):
            smooth = Tensor([smooth], device=self.device)
        if hasattr(self, "_smooth"):
            self._smooth = smooth
        else:
            self.register_buffer("_smooth", smooth)

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        r"""
        Ensures that the panel is centred within the supplied xyz range,
        and that the span of the panel is between xyz_high/20 and xyz_high*10.
        A small random number < 1e-3 is added/subtracted to the min/max z position of the panel, to ensure it doesn't overlap with other panels.

        Arguments:
            xyz_low: minimum x,y,z values for the panel centre in metres
            xyz_high: maximum x,y,z values for the panel centre in metres
        """

        with torch.no_grad():
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            # self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            # self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            # self.xy_span[0].clamp_(min=xyz_high[0] / 20, max=10 * xyz_high[0])
            # self.xy_span[1].clamp_(min=xyz_high[1] / 20, max=10 * xyz_high[1])

    def get_xy_mask(self, xy: Tensor) -> Tensor:
        r"""
        Computes which of the xy points lie inside the physical panel.

        Arguments:
            xy: xy2) tensor of points

        Returns:
            (N,) Boolean mask, where True indicates the point lies inside the panel
        """

        mask = torch.ones_like(xy, dtype=torch.bool)

        for cx in self.panel_centers[:, 0]:
            start = cx - self.panel_size[0] / 2
            end = cx + self.panel_size[0] / 2
            mask *= (xy[:, 0] >= start) * (xy[:, 0] < end)

        for cy in self.panel_centers[:, 1]:
            start = cy - self.panel_size[1] / 2
            end = cy + self.panel_size[1] / 2
            mask *= (xy[:, 1] >= start) * (xy[:, 1] < end)

        return mask
