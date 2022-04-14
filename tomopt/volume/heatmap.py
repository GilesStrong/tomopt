from typing import Tuple, Callable, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, Tensor

from ..muon import MuonBatch
from ..core import DEVICE

__all__ = ["DetectorHeatMap"]


class DetectorHeatMap(nn.Module):
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
        n_cluster: int = 30,
    ):
        if res <= 0:
            raise ValueError("Resolution must be positive")
        if eff <= 0:
            raise ValueError("Efficiency must be positive")

        super().__init__()
        self.area_cost_func = area_cost_func
        self.realistic_validation = realistic_validation
        self.device = device
        self.register_buffer("resolution", torch.tensor(float(res), device=self.device))
        self.register_buffer("efficiency", torch.tensor(float(eff), device=self.device))

        self.n_cluster = n_cluster
        if init_xy_span[1] < init_xy_span[0]:
            init_xy_span = (init_xy_span[1], init_xy_span[0])
        self.register_buffer("xy_fix", torch.tensor(init_xyz[:2], device=self.device))
        self.register_buffer("xy_span_fix", torch.tensor(init_xy_span, device=self.device))
        self.delta_xy = init_xy_span[1] - init_xy_span[0]

        self.gmm = GMM(n_cluster=self.n_cluster, init_xy=init_xyz[:2], device=device, init_xy_span=self.delta_xy)
        self.mu = self.gmm.mu
        self.sig = self.gmm.sig
        self.norm = self.gmm.norm
        self.z = nn.Parameter(torch.tensor(init_xyz[2:3], device=self.device))
        self.gmm.my_params.append(self.z)
        self.range_mult = 1.2

    def __repr__(self) -> str:
        return f"""{self.__class__} at av. xy={self.gmm.mu.T.mean(1)} with n_comp {self.n_cluster}, z={self.z.data}."""

    def get_xy_mask(self, xy: Tensor) -> Tensor:
        raise NotImplementedError("Realistic validation isn't yet supported for heatmap detectors")
        if not isinstance(self.xy_fix, Tensor):
            raise ValueError(f"{self.xy_fix} is not a Tensor for some reason.")  # To appease MyPy
        if not isinstance(self.xy_span_fix, Tensor):
            raise ValueError(f"{self.xy_span_fix} is not a Tensor for some reason.")  # To appease MyPy

        xy_low = self.xy_fix - self.range_mult * self.xy_span_fix
        xy_high = self.xy_fix + self.range_mult * self.xy_span_fix
        return (xy[:, 0] >= xy_low[0]) * (xy[:, 0] < xy_high[0]) * (xy[:, 1] >= xy_low[1]) * (xy[:, 1] < xy_high[1])

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not isinstance(self.resolution, Tensor):
            raise ValueError(f"{self.resolution} is not a Tensor for some reason.")  # To appease MyPy

        if self.training or not self.realistic_validation:
            res = self.resolution * self.gmm(xy) / torch.max(self.gmm(self.mu))
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
            scale = self.gmm(xy) / torch.max(self.gmm(self.mu))
            scale = torch.min(torch.tensor(1.0), scale)
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

    def get_cost(self) -> Tensor:
        return self.area_cost_func(self.sig.prod(1).mean())

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:
        if not isinstance(self.xy_fix, Tensor):
            raise ValueError(f"{self.xy_fix} is not a Tensor for some reason.")  # To appease MyPy
        if not isinstance(self.xy_span_fix, Tensor):
            raise ValueError(f"{self.xy_span_fix} is not a Tensor for some reason.")  # To appease MyPy

        mask = mu.get_xy_mask(self.xy_fix - self.range_mult * self.delta_xy, self.xy_fix + self.range_mult * self.delta_xy)  # Muons in panel
        xy0 = self.xy_fix - (self.delta_xy / 2)  # aprox. Low-left of panel
        rel_xy = mu.xy - xy0
        res = self.get_resolution(mu.xy, mask)
        rel_xy = rel_xy + (torch.randn((len(mu), 2), device=self.device) / res)

        if not self.training and self.realistic_validation:  # Prevent reco hit from exiting panel
            # fix this?
            span = self.xy_span_fix.detach().cpu().numpy()
            rel_xy[mask] = torch.stack([torch.clamp(rel_xy[mask][:, 0], 0, span[0]), torch.clamp(rel_xy[mask][:, 1], 0, span[1])], dim=-1)
        reco_xy = xy0 + rel_xy

        hits = {
            "reco_xy": reco_xy,
            "gen_xy": mu.xy.detach().clone(),
            "z": self.z.expand_as(mu.x)[:, None],
        }
        return hits

    def plot_map(self, bpixelate: bool = False, bsavefig: bool = False, filename: str = None) -> None:
        """"""

        if not isinstance(self.xy_fix, Tensor):
            raise ValueError(f"{self.xy_fix} is not a Tensor for some reason.")  # To appease MyPy
        if not isinstance(self.xy_span_fix, Tensor):
            raise ValueError(f"{self.xy_span_fix} is not a Tensor for some reason.")  # To appease MyPy

        def get_z_from_mesh(x: Tensor, y: Tensor) -> Tensor:
            stacked_t = torch.stack([x, y]).T
            reshaped = torch.reshape(stacked_t, (stacked_t.shape[0] * stacked_t.shape[1], stacked_t.shape[2]))
            reshaped = torch.unsqueeze(reshaped, 1)
            z = self.gmm(reshaped).prod(1) / torch.max(self.gmm(self.mu))
            torch.min(torch.tensor(1.0), z)
            z = torch.reshape(z, (stacked_t.shape[0], stacked_t.shape[1]))

            return z

        with sns.axes_style(style="whitegrid", rc={"patch.edgecolor": "none"}):
            x = self.xy_fix[0].detach().cpu().numpy()
            y = self.xy_fix[1].detach().cpu().numpy()
            xs = torch.linspace(x - 2 * self.delta_xy, x + 2 * self.delta_xy, steps=200)
            ys = torch.linspace(y - 2 * self.delta_xy, y + 2 * self.delta_xy, steps=200)
            x, y = torch.meshgrid(xs, ys)
            z = get_z_from_mesh(x, y).detach().cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            if bpixelate:
                cs = ax.scatter(x, y, c=z, cmap="plasma", s=450.0, marker="s")
            else:
                cs = ax.contourf(x, y, z, cmap="plasma", vmin=0.0, vmax=0.9)
            ax.set_aspect("equal")
            fig.colorbar(cs)

            if bsavefig:
                if filename is None:
                    filename = "heatmap_plot.png"
                plt.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()

    def clamp_params(self, musigz_low: Tuple[float, float, float], musigz_high: Tuple[float, float, float]) -> None:
        with torch.no_grad():
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            self.mu.clamp_(min=musigz_low[0], max=musigz_high[0])
            self.z.clamp_(min=musigz_low[2] + eps, max=musigz_high[2] - eps)
            self.sig.clamp_(min=musigz_low[1] / self.range_mult, max=self.range_mult * musigz_high[1])
            self.norm.clamp_(min=0.01, max=1.5)

    @property
    def x(self) -> Tensor:
        if not isinstance(self.xy_fix, Tensor):
            raise ValueError(f"{self.xy_fix} is not a Tensor for some reason.")  # To appease MyPy
        return self.xy_fix[0]

    @property
    def y(self) -> Tensor:
        if not isinstance(self.xy_fix, Tensor):
            raise ValueError(f"{self.xy_fix} is not a Tensor for some reason.")  # I just love MyPy
        return self.xy_fix[1]


class GMM(nn.Module):
    """"""

    def __init__(
        self,
        n_cluster: int = 20,
        init_xy: Tuple[float, float] = (0.0, 0.0),
        init_xy_span: float = 10.0,
        init_norm: float = 1.0,
        device: torch.device = DEVICE,
    ) -> None:
        super(GMM, self).__init__()
        self.n_cluster = n_cluster
        self.device = device
        self._init_xy = torch.tensor(init_xy, device=self.device)
        self._init_xy_span = torch.tensor(init_xy_span, device=self.device)

        rand_mu = self._init_xy_span * (0.5 - torch.rand(self.n_cluster, 2, device=self.device))
        self.mu = nn.Parameter(rand_mu + self._init_xy)

        rand_sig = torch.max(torch.rand(self.n_cluster, 2, device=self.device), torch.tensor(0.2))
        self.sig = nn.Parameter(self._init_xy_span * rand_sig)
        self.norm = nn.Parameter(torch.tensor([float(init_norm)], device=self.device))

        params = [self.mu, self.sig, self.norm]
        self.my_params = nn.ParameterList(params)

        mix = torch.distributions.Categorical(
            torch.ones(
                self.n_cluster,
            )
        )
        comp = torch.distributions.Independent(
            torch.distributions.Normal(
                self.mu,
                self.sig,
            ),
            1,
        )
        self.gmm = torch.distributions.MixtureSameFamily(mix, comp)

    def forward(self, x: Tensor) -> Tensor:
        res = self.norm * torch.exp(self.gmm.log_prob(x))
        res = res.reshape(res.shape[0], 1)
        res = res.expand(res.shape[0], 2)
        res = torch.sqrt(res)
        return res
