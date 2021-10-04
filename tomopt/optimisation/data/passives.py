from typing import Tuple, Callable, List, Optional, Union, Generator
from random import shuffle
from abc import abstractmethod, ABCMeta
import numpy as np

import torch
from torch import Tensor

from ...core import X0

__all__ = ["VoxelPassiveGenerator", "BlockPassiveGenerator", "PassiveYielder"]


class AbsPassiveGenerator(metaclass=ABCMeta):
    def __init__(self, materials: Optional[List[str]] = None) -> None:
        if materials is None:
            materials = [m for m in X0]
        self.materials = materials

    @abstractmethod
    def generate(self) -> Callable[..., Tensor]:
        pass


class VoxelPassiveGenerator(AbsPassiveGenerator):
    def generate(self) -> Callable[..., Tensor]:
        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            x0s = lw.new_tensor([X0[m] for m in self.materials])
            shp = (lw / size).long()
            return x0s[torch.randint(high=len(x0s), size=(shp.prod().numpy(),), device=x0s.device)].reshape(list(shp))

        return generator


class BlockPassiveGenerator(AbsPassiveGenerator):
    def __init__(
        self,
        lw: Tuple[float, float],
        z_range: Tuple[float, float],
        block_size: Tuple[float, float, float],
        sort_x0: bool,
        materials: Optional[List[str]] = None,
    ) -> None:
        super().__init__(materials=materials)
        self.lw, self.z_range, self.block_size, self.sort_x0 = lw, z_range, block_size, sort_x0

    def generate(self) -> Callable[..., Tensor]:
        mats = np.random.choice(self.materials, 2, replace=False)
        base_x0 = X0[mats[0]]
        block_x0 = X0[mats[1]]
        if self.sort_x0 and base_x0 < block_x0:  # Ensure block is denser material
            base_x0, block_x0 = block_x0, base_x0

        block_size = np.random.choice(self.block_size, 3, replace=False)
        block_low = np.hstack(
            (
                np.random.uniform(high=self.lw[0] - block_size[0]),
                np.random.uniform(high=self.lw[1] - block_size[1]),
                np.random.uniform(self.z_range[0], self.z_range[1] - block_size[2]),
            )
        )

        block_high = block_low + block_size

        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            shp = (lw / size).long()
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list(shp)) * base_x0
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = block_x0
            return rad_length

        return generator


class PassiveYielder:
    def __init__(self, passives: Union[List[Callable[..., Tensor]], AbsPassiveGenerator], n_passives: Optional[int] = None, shuffle: bool = True):
        self.passives, self.n_passives, self.shuffle = passives, n_passives, shuffle
        if isinstance(self.passives, AbsPassiveGenerator):
            if self.n_passives is None:
                raise ValueError("If a AbsPassiveGenerator class is used, n_passives must be specified")
        else:
            self.n_passives = len(self.passives)

    def __len__(self) -> int:
        return self.n_passives

    def __iter__(self) -> Generator[Callable[..., Tensor], None, None]:
        if isinstance(self.passives, AbsPassiveGenerator):
            for _ in range(self.n_passives):
                yield self.passives.generate()
        else:
            if self.shuffle:
                shuffle(self.passives)
            for p in self.passives:
                yield p
