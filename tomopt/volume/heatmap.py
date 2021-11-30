from typing import Tuple, Callable, Optional, Dict
import numpy as np

import torch
from torch import nn, Tensor

from ..muon import MuonBatch
from ..core import DEVICE


__all__ = ["DetectorHeatMap"]


class DetectorHeatMap(nn.Module):
    def __init__(
        self,
        *,
        init_xyz: Tuple[float, float, float],
        init_xy_span: Tuple[float, float],
        area_cost_func: Callable[[Tensor], Tensor],
        realistic_validation: bool = False,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.area_cost_func = area_cost_func
        self.realistic_validation = realistic_validation
        self.device = device
        self.register_buffer(
            "resolution",
            torch.tensor(float(1000.0), requires_grad=True, device=self.device),
        )
        self.register_buffer(
            "efficiency",
            torch.tensor(float(1.0), requires_grad=True, device=self.device),
        )
        self.z = torch.tensor(init_xyz[2:3], device=self.device)

        # what to do with these?
        # self.xy = nn.Parameter(torch.tensor(init_xyz[:2], device=self.device))
        # self.xy_span = nn.Parameter(torch.tensor(init_xy_span, device=self.device))
        self.xy = torch.tensor(init_xyz[:2], device=self.device)
        self.xy_span = torch.tensor(init_xy_span, device=self.device)

        self.gaussian_mixture = GaussCluster(n_cluster=5)

    def __repr__(self) -> str:
        return f"""{self.__class__} located at xy={self.xy.data}, z={self.z.data}, and xy span {self.xy_span.data}"""

    def get_xy_mask(self, xy: Tensor) -> Tensor:
        xy_low = self.xy - (self.xy_span / 2)
        xy_high = self.xy + (self.xy_span / 2)
        return (xy[:, 0] >= xy_low[0]) * (xy[:, 0] < xy_high[0]) * (xy[:, 1] >= xy_low[1]) * (xy[:, 1] < xy_high[1])

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not isinstance(self.resolution, Tensor):
            raise ValueError(f"{self.resolution} is not a Tensor.")
        return self.resolution

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None, as_2d: bool = False) -> Tensor:
        if not isinstance(self.efficiency, Tensor):
            raise ValueError(f"{self.efficiency} is not a Tensor.")

        print("xy shape", xy.shape)
        gmm = self.gaussian_mixture
        res = gmm(xy)
        scale = torch.exp(res)
        # Note: Check if cutting off is best approach
        scale = torch.max(torch.tensor(0.0), scale)
        scale = torch.min(torch.tensor(1.0), scale)
        print("scale preds", scale.mean(), scale.max(), scale.std())

        if self.training or not self.realistic_validation:
            if not as_2d:
                scale = torch.prod(scale, dim=-1)  # Maybe weight product by xy distance?
            eff = self.efficiency * scale
        else:
            # Note: discretize: fix scale to n stacked detector eff values for inference
            eff = self.efficiency * scale
            if as_2d:
                eff = eff[:, None]

        return eff

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:
        print("Mu shape", mu)
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
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            self.xy_span[0].clamp_(min=xyz_high[0] / 20, max=xyz_high[0])
            self.xy_span[1].clamp_(min=xyz_high[1] / 20, max=xyz_high[1])
        pass

    @property
    def x(self) -> Tensor:
        return self.xy[0]

    @property
    def y(self) -> Tensor:
        return self.xy[1]


class GaussCluster(nn.Module):
    """Should be replaxed with torch.distributions.MixtureSameFamily for GMM."""

    def __init__(self, n_cluster: int = 2) -> None:
        super(GaussCluster, self).__init__()
        self._n_cluster = n_cluster

        self._mu = []
        self._sig = []
        self._norm = []
        for n in range(self._n_cluster):
            rand_mu = 1.0 * (0.5 - torch.rand(2, requires_grad=True))
            rand_sig = 1.0 * torch.eye(2, requires_grad=True)  # + 0.0001 * (0.5 - torch.rand((2, 2)))
            norm = torch.tensor(1.0, requires_grad=True)
            self._mu.append(torch.nn.Parameter(rand_mu))
            self._sig.append(torch.nn.Parameter(rand_sig))
            self._norm.append(torch.nn.Parameter(norm))

        # mypy does not like list comprehensions with different types, apparently
        # params = [self._mu, self._sig, self._norm]
        # params = [item for sublist in params for item in sublist]
        params = []
        for par_list in [self._mu, self._sig, self._norm]:
            for elem in par_list:
                params.append(elem)

        self.my_params = nn.ParameterList(params)

    def forward(self, x: Tensor) -> Tensor:
        res = torch.zeros(x.shape[0], 1)

        for n in range(self._n_cluster):
            mu = self._mu[n]
            sig = torch.abs(self._sig[n])
            g = torch.distributions.MultivariateNormal(mu, sig)

            int_res = g.log_prob(x)
            int_res = int_res.reshape((*int_res.shape, 1))
            res += torch.abs(self._norm[n]) * torch.exp(int_res)

        res = torch.max(torch.tensor(1e-12), res)
        print("scale raw preds", res.mean(), res.max(), res.std())

        return torch.log(res)
