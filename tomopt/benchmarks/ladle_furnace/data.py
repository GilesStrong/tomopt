from typing import Tuple, Callable, List

import torch
from torch import Tensor

from ...optimisation.data.passives import AbsPassiveGenerator
from ...volume import Volume
from ...core import X0

__all__ = ["LadleFurnacePassiveGenerator"]


class LadleFurnacePassiveGenerator(AbsPassiveGenerator):
    def __init__(
        self,
        volume: Volume,
        x0_furnace: float = X0["iron"],
        fill_materials: List[str] = ["aluminium"],
        slag_materials: List[str] = ["silicon", "carbon", "beryllium"],
    ) -> None:
        self.x0_furnace, self.fill_materials, self.slag_materials = x0_furnace, fill_materials, slag_materials
        super().__init__(volume=volume, materials=self.fill_materials + self.slag_materials)
        self.slag_x0s = Tensor([X0[m] for m in self.slag_materials], device=self.volume.device)
        self.fill_x0s = Tensor([X0[m] for m in self.fill_materials], device=self.volume.device)

        self.xy_shp = (self.lw / self.size).astype(int).tolist()
        self.fill_z_range = ((self.z_range[0]) + self.size, self.z_range[1])

    def _generate(self) -> Tuple[Callable[..., Tensor], Tensor]:
        mat_z = self.size + self.fill_z_range[0] + ((self.fill_z_range[1] - (self.fill_z_range[0] + self.size)) * torch.rand(1, device=self.volume.device))
        slag_z = mat_z + ((self.z_range[1] - mat_z) * torch.rand(1, device=self.volume.device))

        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            if z <= self.fill_z_range[0]:
                x0 = self.x0_furnace * torch.ones(self.xy_shp)
            elif z > self.fill_z_range[0] and z <= mat_z:
                x0 = self.fill_x0s[torch.randint(high=len(self.fill_x0s), size=(self.xy_shp[0] * self.xy_shp[1],), device=self.fill_x0s.device)].reshape(
                    self.xy_shp
                )
            elif z > mat_z and z <= slag_z:
                x0 = self.slag_x0s[torch.randint(high=len(self.slag_x0s), size=(self.xy_shp[0] * self.xy_shp[1],), device=self.slag_x0s.device)].reshape(
                    self.xy_shp
                )
            elif z > slag_z:
                x0 = X0["air"] * torch.ones(self.xy_shp)
            x0[0, :] = self.x0_furnace
            x0[-1, :] = self.x0_furnace
            x0[:, 0] = self.x0_furnace
            x0[:, -1] = self.x0_furnace
            return x0

        return generator, mat_z