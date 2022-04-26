from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ...muon import MuonBatch
from ...volume.panel import DetectorPanel
from ...core import DEVICE


__all__ = ["PhiDetectorPanel"]


class PhiDetectorPanel(DetectorPanel):
    def __init__(self, *, init_phi: float, init_z: float, res: float, eff: float, device: torch.device = DEVICE):
        super().__init__(
            res=res,
            eff=eff,
            init_xyz=(0.0, 0.0, float(init_z)),
            init_xy_span=(1.0, 1.0),
            area_cost_func=lambda x: torch.zeros(1, 1),
            realistic_validation=False,
            device=device,
        )
        self.phi = nn.Parameter(torch.tensor(float(init_phi), device=self.device))

    def __repr__(self) -> str:
        return f"""{self.__class__} located at xy={self.xy.data}, z={self.z.data}, xy span {self.xy_span.data}, and phi angle {self.phi.data}"""

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not isinstance(self.resolution, Tensor):
            raise ValueError(f"{self.resolution} is not a Tensor for some reason.")  # To appease MyPy
        return self.resolution

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None, as_2d: bool = False) -> Tensor:
        if not isinstance(self.efficiency, Tensor):
            raise ValueError(f"{self.efficiency} is not a Tensor for some reason.")  # To appease MyPy
        return self.efficiency

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:
        gen_h = (mu.x[:, None] * self.phi.cos()) + (mu.y[:, None] * self.phi.sin())
        reco_h = gen_h + (torch.randn((len(mu), 1), device=self.device) / self.resolution)

        hits = {
            "reco_h": reco_h,
            "gen_h": gen_h,
            "gen_xy": mu.xy.detach().clone(),
            "z": self.z.expand_as(mu.x)[:, None],
            "phi": self.phi.expand_as(mu.x)[:, None],
        }
        return hits

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        super().clamp_params(xyz_low, xyz_high)
        with torch.no_grad():
            self.phi.clamp_(min=0, max=torch.pi * 2)
