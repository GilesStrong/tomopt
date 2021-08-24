from typing import Tuple, Callable, Optional, Dict
import numpy as np

import torch
from torch import nn, Tensor

from ..muon import MuonBatch
from ..core import DEVICE


__all__ = ["DetectorPanel"]


class DetectorPanel(nn.Module):
    def __init__(
        self,
        *,
        res: float,
        eff: float,
        init_xyz: Tuple[float, float, float],
        init_xy_span: Tuple[float, float],
        area_cost_func: Callable[[Tensor], Tensor],
        realistic_validation: bool = False,
        device: torch.device = DEVICE,
    ):
        if res <= 0:
            raise ValueError("Resolution must be positive")
        if eff <= 0:
            raise ValueError("Efficiency must be positive")

        super().__init__()
        self.area_cost_func, self.realistic_validation, self.device = area_cost_func, realistic_validation, device
        self.register_buffer("resolution", torch.tensor(float(res), requires_grad=True, device=self.device))
        self.register_buffer("efficiency", torch.tensor(float(eff), requires_grad=True, device=self.device))
        self.xy = nn.Parameter(torch.tensor(init_xyz[:2], device=self.device))
        self.z = nn.Parameter(torch.tensor(init_xyz[2:3], device=self.device))
        self.xy_span = nn.Parameter(torch.tensor(init_xy_span, device=self.device))

    def __repr__(self) -> str:
        return f"""{self.__class__} located at xy={self.xy.data}, z={self.z.data}, and xy span {self.xy_span.data}"""

    def get_xy_mask(self, xy: Tensor) -> Tensor:
        xy_low = self.xy - (self.xy_span / 2)
        xy_high = self.xy + (self.xy_span / 2)
        return (xy[:, 0] >= xy_low[0]) * (xy[:, 0] < xy_high[0]) * (xy[:, 1] >= xy_low[1]) * (xy[:, 1] < xy_high[1])

    def get_gauss(self) -> torch.distributions.Normal:
        try:
            return torch.distributions.Normal(self.xy, self.xy_span)  # maybe upscale span?
        except ValueError:
            raise ValueError(f"Invalid parameters for Gaussian: loc={self.xy}, scale={self.xy_span}")

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not isinstance(self.resolution, Tensor):
            raise ValueError(f"{self.resolution} is not a Tensor for some reason.")  # To appease MyPy
        if self.training or not self.realistic_validation:
            g = self.get_gauss()
            res = self.resolution * torch.exp(g.log_prob(xy)) / torch.exp(g.log_prob(self.xy))  # Maybe detach the normalisation?
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            res = torch.zeros((len(xy), 2), device=self.device)  # Zero detection outside detector
            res[mask] = self.resolution
        return res

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not isinstance(self.efficiency, Tensor):
            raise ValueError(f"{self.efficiency} is not a Tensor for some reason.")  # To appease MyPy
        if self.training or not self.realistic_validation:
            g = self.get_gauss()
            scale = torch.exp(g.log_prob(xy)) / torch.exp(g.log_prob(self.xy))  # Maybe detach the normalisation?
            eff = self.efficiency * torch.prod(scale, dim=-1)  # Maybe weight product by xy distance?
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            eff = torch.zeros(len(xy), device=self.device)  # Zero detection outside detector
            eff[mask] = self.efficiency
        return eff

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:
        mask = mu.get_xy_mask(self.xy - (self.xy_span / 2), self.xy + (self.xy_span / 2))  # Muons in panel

        xy0 = self.xy - (self.xy_span / 2)  # Low-left of voxel
        rel_xy = mu.xy - xy0
        res = self.get_resolution(mu.xy, mask)
        rel_xy = rel_xy + (torch.randn((len(mu), 2), device=self.device) / res)

        if not self.training and self.realistic_validation:  # Prevent reco hit from exiting panel
            span = self.xy_span.detach().cpu().numpy()
            rel_xy[mask] = torch.stack([torch.clamp(rel_xy[mask][:, 0], 0, span[0]), torch.clamp(rel_xy[mask][:, 1], 0, span[1])], dim=-1)
        reco_xy = xy0 + rel_xy

        hits = {
            "reco_xy": reco_xy,
            "gen_xy": mu.xy.detach().clone(),
            "z": self.z.expand_as(mu.x)[:, None],
        }
        return hits

    def get_cost(self) -> Tensor:
        return self.area_cost_func(self.xy_span.prod())

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        with torch.no_grad():
            eps = np.random.uniform(0, 5e-5)  # prevent hits at same z due to clamping
            self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            self.xy_span[0].clamp_(min=1e-7, max=xyz_high[0])
            self.xy_span[1].clamp_(min=1e-7, max=xyz_high[1])

    @property
    def x(self) -> Tensor:
        return self.xy[0]

    @property
    def y(self) -> Tensor:
        return self.xy[1]
