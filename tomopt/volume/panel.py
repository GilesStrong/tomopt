from typing import Tuple, Optional, Dict
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
        m2_cost: float,
        realistic_validation: bool = False,
        device: torch.device = DEVICE,
    ):
        if res <= 0:
            raise ValueError("Resolution must be positive")
        if eff <= 0:
            raise ValueError("Efficiency must be positive")

        super().__init__()
        self.realistic_validation, self.device = realistic_validation, device
        self.register_buffer("m2_cost", torch.tensor(float(m2_cost), device=self.device))
        self.register_buffer("resolution", torch.tensor(float(res), device=self.device))
        self.register_buffer("efficiency", torch.tensor(float(eff), device=self.device))
        self.xy = nn.Parameter(torch.tensor(init_xyz[:2], device=self.device))
        self.z = nn.Parameter(torch.tensor(init_xyz[2:3], device=self.device))
        self.xy_span = nn.Parameter(torch.tensor(init_xy_span, device=self.device))
        self.budget_scale = torch.ones(1, device=device)

    def get_scaled_xy_span(self) -> Tensor:
        return self.xy_span * self.budget_scale

    def __repr__(self) -> str:
        return f"""{self.__class__} located at xy={self.xy.data}, z={self.z.data}, and xy span {self.xy_span.data}"""

    def get_xy_mask(self, xy: Tensor) -> Tensor:
        span = self.get_scaled_xy_span()
        xy_low = self.xy - (span / 2)
        xy_high = self.xy + (span / 2)
        return (xy[:, 0] >= xy_low[0]) * (xy[:, 0] < xy_high[0]) * (xy[:, 1] >= xy_low[1]) * (xy[:, 1] < xy_high[1])

    def get_gauss(self) -> torch.distributions.Normal:
        try:
            return torch.distributions.Normal(self.xy, self.get_scaled_xy_span() / 4)  # We say that the panel widths corresponds to 2-sigma of the Gaussian
        except ValueError:
            raise ValueError(f"Invalid parameters for Gaussian: loc={self.xy}, scale={self.get_scaled_xy_span() / 4}")

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not isinstance(self.resolution, Tensor):
            raise ValueError(f"{self.resolution} is not a Tensor for some reason.")  # To appease MyPy
        if self.training or not self.realistic_validation:
            g = self.get_gauss()
            res = self.resolution * torch.exp(g.log_prob(xy)) / torch.exp(g.log_prob(self.xy))
            res = torch.clamp_min(res, 1e-10)  # To avoid NaN gradients
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            res = torch.zeros((len(xy), 2), device=self.device)  # Zero detection outside detector
            res[mask] = self.resolution
        return res

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None, as_2d: bool = False) -> Tensor:
        if not isinstance(self.efficiency, Tensor):
            raise ValueError(f"{self.efficiency} is not a Tensor for some reason.")  # To appease MyPy
        if self.training or not self.realistic_validation:
            g = self.get_gauss()
            scale = torch.exp(g.log_prob(xy)) / torch.exp(g.log_prob(self.xy))
            if not as_2d:
                scale = torch.prod(scale, dim=-1)  # Maybe weight product by xy distance?
            eff = self.efficiency * scale
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            eff = torch.zeros(len(xy), device=self.device)  # Zero detection outside detector
            eff[mask] = self.efficiency
            if as_2d:
                eff = eff[:, None]
        return eff

    def _set_budget_scale(self, budget: Optional[Tensor] = None) -> None:
        # Hack to provide budget persistence; eventually encode res and eff in hits
        if budget is not None:
            self.budget_scale = torch.sqrt(budget / (self.m2_cost * self.xy_span.prod()))

    def get_hits(self, mu: MuonBatch, budget: Optional[Tensor] = None) -> Dict[str, Tensor]:
        self._set_budget_scale(budget)
        span = self.get_scaled_xy_span()
        mask = mu.get_xy_mask(self.xy - (span / 2), self.xy + (span / 2))  # Muons in panel

        xy0 = self.xy - (span / 2)  # Low-left of panel
        rel_xy = mu.xy - xy0
        res = self.get_resolution(mu.xy, mask)
        rel_xy = rel_xy + (torch.randn((len(mu), 2), device=self.device) / res)

        if not self.training and self.realistic_validation:  # Prevent reco hit from exiting panel
            np_span = span.detach().cpu().numpy()
            rel_xy[mask] = torch.stack([torch.clamp(rel_xy[mask][:, 0], 0, np_span[0]), torch.clamp(rel_xy[mask][:, 1], 0, np_span[1])], dim=-1)
        reco_xy = xy0 + rel_xy

        hits = {
            "reco_xy": reco_xy,
            "gen_xy": mu.xy.detach().clone(),
            "z": self.z.expand_as(mu.x)[:, None],
        }
        return hits

    def get_cost(self) -> Tensor:
        return self.m2_cost * self.xy_span.prod()

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        with torch.no_grad():
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            self.xy_span[0].clamp_(min=xyz_high[0] / 20, max=10 * xyz_high[0])
            self.xy_span[1].clamp_(min=xyz_high[1] / 20, max=10 * xyz_high[1])

    @property
    def x(self) -> Tensor:
        return self.xy[0]

    @property
    def y(self) -> Tensor:
        return self.xy[1]
