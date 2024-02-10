from typing import Optional, Tuple

import torch
from torch import Tensor

from ...core import X0, RadLengthFunc
from ...optimisation.data.passives import AbsPassiveGenerator
from ...volume import Volume

__all__ = ["LadleFurnacePassiveGenerator"]


class LadleFurnacePassiveGenerator(AbsPassiveGenerator):
    r"""
    Research tested only: no unit tests
    """

    def __init__(
        self,
        volume: Volume,
        x0_furnace: float = X0["steel"],
        fill_material: str = "hot liquid steel",
        slag_material: str = "slag",
    ) -> None:
        self.x0_furnace, self.fill_material, self.slag_material = x0_furnace, fill_material, slag_material
        super().__init__(volume=volume, materials=[self.fill_material, self.slag_material])
        self.slag_x0 = X0[self.slag_material]
        self.fill_x0 = X0[self.fill_material]

        self.xy_shp = (self.lw / self.size).astype(int).tolist()
        self.fill_z_range = ((self.z_range[0]) + self.size, self.z_range[1])

    def _generate(self, fixed_mat_z: Optional[float] = None, fixed_slag_z: Optional[float] = None) -> Tuple[RadLengthFunc, Tensor]:
        if fixed_mat_z is None:
            mat_z = self.size + self.fill_z_range[0] + ((self.fill_z_range[1] - (self.fill_z_range[0] + self.size)) * torch.rand(1, device=self.volume.device))
        else:
            mat_z = Tensor([fixed_mat_z], device=self.volume.device)
        if fixed_slag_z is None:
            slag_z = mat_z + ((self.z_range[1] - mat_z) * torch.rand(1, device=self.volume.device))
        else:
            slag_z = Tensor([fixed_slag_z], device=self.volume.device)

        def generator(*, z: Tensor, lw: Tensor, size: float) -> Tensor:
            if z <= self.fill_z_range[0]:  # Bottom layer
                x0 = self.x0_furnace * torch.ones(self.xy_shp)

            elif z > self.fill_z_range[0] and z <= mat_z:  # fill material
                x0 = self.fill_x0 * torch.ones(self.xy_shp)

            elif z > mat_z and z <= slag_z:  # Slag
                x0 = self.slag_x0 * torch.ones(self.xy_shp)

            elif z > slag_z:
                x0 = X0["air"] * torch.ones(self.xy_shp)

            # Add furnace walls
            x0[0, :] = self.x0_furnace
            x0[-1, :] = self.x0_furnace
            x0[:, 0] = self.x0_furnace
            x0[:, -1] = self.x0_furnace
            return x0

        return generator, mat_z
