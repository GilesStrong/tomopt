from typing import Tuple, Callable, List, Optional, Union, Generator
from random import shuffle
from abc import abstractmethod, ABCMeta
import numpy as np

import torch
from torch import Tensor

from ...core import X0
from ...volume import Volume

__all__ = ["VoxelPassiveGenerator", "RandomBlockPassiveGenerator", "BlockPresentPassiveGenerator", "PassiveYielder"]


class AbsPassiveGenerator(metaclass=ABCMeta):
    def __init__(
        self,
        volume: Volume,
        materials: Optional[List[str]] = None,
    ) -> None:
        self.volume = volume
        if materials is None:
            materials = [m for m in X0]
        self.materials = materials
        self.lw = volume.lw.detach().cpu().numpy()
        self.z_range = [z.detach().cpu().item() for z in self.volume.get_passive_z_range()]
        self.size = volume.passive_size

    def get_data(self) -> Tuple[Callable[..., Tensor], Optional[Tensor]]:
        return self._generate()

    def generate(self) -> Callable[..., Tensor]:
        f, _ = self._generate()
        return f

    @abstractmethod
    def _generate(self) -> Tuple[Callable[..., Tensor], Optional[Tensor]]:
        pass


class AbsBlockPassiveGenerator(AbsPassiveGenerator):
    def __init__(
        self,
        volume: Volume,
        block_size: Optional[Tuple[float, float, float]],
        block_size_max_half: Optional[bool] = None,
        materials: Optional[List[str]] = None,
    ) -> None:
        super().__init__(volume=volume, materials=materials)
        self.block_size = block_size
        self.block_size_max = [self.lw[0], self.lw[1], self.z_range[1] - self.z_range[0]]
        if self.block_size is None and block_size_max_half is None:
            raise ValueError("Random block size requested, but block_size_max_half is None, please set to True or False")
        if block_size_max_half:
            self.block_size_max = [x / 2 for x in self.block_size_max]

    def _get_block_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.block_size is None:
            block_size = np.hstack(
                (
                    np.random.uniform(self.size, self.block_size_max[0]),
                    np.random.uniform(self.size, self.block_size_max[1]),
                    np.random.uniform(self.size, self.block_size_max[2]),
                )
            )
        else:
            block_size = np.random.choice(self.block_size, 3, replace=False)

        block_low = np.hstack(
            (
                np.random.uniform(high=self.lw[0] - block_size[0]),
                np.random.uniform(high=self.lw[1] - block_size[1]),
                np.random.uniform(self.z_range[0], self.z_range[1] - block_size[2]),
            )
        )
        block_high = block_low + block_size

        return block_low, block_high


class RandomBlockPassiveGenerator(AbsBlockPassiveGenerator):
    def __init__(
        self,
        volume: Volume,
        block_size: Optional[Tuple[float, float, float]],
        sort_x0: bool,
        enforce_diff_mat: bool,
        block_size_max_half: Optional[bool] = None,
        materials: Optional[List[str]] = None,
    ) -> None:
        super().__init__(volume=volume, block_size=block_size, materials=materials, block_size_max_half=block_size_max_half)
        self.sort_x0, self.enforce_diff_mat = sort_x0, enforce_diff_mat

    def _generate(self) -> Tuple[Callable[..., Tensor], Tensor]:
        bkg_mat, block_mat = None, None
        while bkg_mat is None or block_mat is None or (bkg_mat == block_mat and self.enforce_diff_mat):
            bkg_mat = np.random.randint(0, len(self.materials))
            block_mat = np.random.randint(0, len(self.materials))
        base_x0 = X0[self.materials[bkg_mat]]
        block_x0 = X0[self.materials[block_mat]]
        if self.sort_x0 and block_x0 > base_x0:
            block_x0, base_x0 = base_x0, block_x0

        block_low, block_high = self._get_block_coords()

        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            shp = (lw / size).long()
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list(shp)) * base_x0
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = block_x0
            return rad_length

        return generator, Tensor([block_x0])


class BlockPresentPassiveGenerator(AbsBlockPassiveGenerator):
    def _generate(self) -> Tuple[Callable[..., Tensor], Tensor]:
        bkg_mat = 0
        block_mat = np.random.randint(0, len(self.materials))
        base_x0 = X0[self.materials[bkg_mat]]
        block_x0 = X0[self.materials[block_mat]]

        block_low, block_high = self._get_block_coords()

        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            shp = (lw / size).long()
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list(shp)) * base_x0
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = block_x0
            return rad_length

        return generator, Tensor([block_x0])


class VoxelPassiveGenerator(AbsPassiveGenerator):
    def _generate(self) -> Tuple[Callable[..., Tensor], None]:
        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            x0s = lw.new_tensor([X0[m] for m in self.materials])
            shp = (lw / size).long()
            return x0s[torch.randint(high=len(x0s), size=(shp.prod().numpy(),), device=x0s.device)].reshape(list(shp))

        return generator, None


class PassiveYielder:
    def __init__(
        self,
        passives: Union[List[Union[Tuple[Callable[..., Tensor], Optional[Tensor]], Callable[..., Tensor]]], AbsPassiveGenerator],
        n_passives: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.passives, self.n_passives, self.shuffle = passives, n_passives, shuffle
        if isinstance(self.passives, AbsPassiveGenerator):
            if self.n_passives is None:
                raise ValueError("If a AbsPassiveGenerator class is used, n_passives must be specified")
        else:
            self.n_passives = len(self.passives)

    def __len__(self) -> int:
        return self.n_passives

    def __iter__(self) -> Generator[Tuple[Callable[..., Tensor], Optional[Tensor]], None, None]:
        if isinstance(self.passives, AbsPassiveGenerator):
            for _ in range(self.n_passives):
                yield self.passives.get_data()
        else:
            if self.shuffle:
                shuffle(self.passives)
            for p in self.passives:
                if isinstance(p, tuple):
                    yield p
                else:
                    yield p, None
