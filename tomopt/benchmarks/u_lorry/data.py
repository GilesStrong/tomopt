from typing import Tuple, Callable, List
import numpy as np

import torch
from torch import Tensor

from ...optimisation.data.passives import AbsPassiveGenerator
from ...volume import Volume
from ...core import X0

__all__ = ["ULorryPassiveGenerator"]


class ULorryPassiveGenerator(AbsPassiveGenerator):
    def __init__(
        self,
        volume: Volume,
        u_volume: float,
        u_prob: float = 0.5,
        fill_frac: float = 0.8,
        x0_lorry: float = X0["iron"],
        bkg_materials: List[str] = ["air", "iron"],
    ) -> None:
        super().__init__(volume=volume, materials=["iron", "uranium"])
        self.u_volume, self.u_prob, self.fill_frac, self.x0_lorry, self.bkg_materials = u_volume, u_prob, fill_frac, x0_lorry, bkg_materials
        self.bkg_x0s = Tensor([X0[m] for m in self.bkg_materials], device=self.volume.device)
        self.n_u_voxels = np.max((1, self.u_volume // (self.size**3)))
        self.xy_shp = (self.lw / self.size).astype(int).tolist()
        self.bkg_z_range = ((self.z_range[0]) + self.size, self.fill_frac * self.z_range[1])

    def _get_block_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        w = int(np.random.uniform(1, np.min(((self.u_volume / (self.size**2)) // self.size, self.xy_shp[0]))))
        h = int(
            np.random.uniform(1, np.min(((self.u_volume / (self.size * w * self.size)) // self.size, (self.bkg_z_range[1] - self.bkg_z_range[0]) / self.size)))
        )
        l = int(np.min(((self.u_volume / (w * self.size * h * self.size)) // self.size, self.xy_shp[1])))
        block_size = np.hstack((w, l, h)).astype(float)
        block_size[2] *= self.size

        block_low = np.hstack(
            (
                int(np.random.uniform(0, self.xy_shp[0] - block_size[0])),
                int(np.random.uniform(0, self.xy_shp[1] - block_size[1])),
                np.random.uniform(self.bkg_z_range[0], self.bkg_z_range[1] - block_size[2]),
            )
        )
        block_high = block_low + block_size

        return block_low, block_high

    def _generate(self) -> Tuple[Callable[..., Tensor], Tensor]:
        block_present = torch.rand(1) < self.u_prob
        if block_present:
            block_low, block_high = self._get_block_coords()

        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            if z <= self.bkg_z_range[0]:
                x0 = self.x0_lorry * torch.ones(self.xy_shp)
            elif z > self.bkg_z_range[0] and z <= self.bkg_z_range[1]:
                x0 = self.bkg_x0s[torch.randint(high=len(self.bkg_x0s), size=(self.xy_shp[0] * self.xy_shp[1],), device=self.bkg_x0s.device)].reshape(
                    self.xy_shp
                )
            elif z > self.bkg_z_range[1]:
                x0 = X0["air"] * torch.ones(self.xy_shp)

            if block_present:
                if z > block_low[2] and z <= block_high[2]:
                    x0[block_low[0].astype(int) : block_high[0].astype(int), block_low[1].astype(int) : block_high[1].astype(int)] = X0["uranium"]
            return x0

        return generator, block_present.long()