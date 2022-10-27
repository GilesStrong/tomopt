from typing import Tuple, List, Optional, Union, Generator
from random import shuffle
from abc import abstractmethod, ABCMeta
import numpy as np

import torch
from torch import Tensor

from ...core import X0, RadLengthFunc
from ...volume import Volume

r"""
Provides classes that generate and yield passive volume layouts
"""

__all__ = ["VoxelPassiveGenerator", "RandomBlockPassiveGenerator", "BlockPresentPassiveGenerator", "PassiveYielder"]


class AbsPassiveGenerator(metaclass=ABCMeta):
    r"""
    Abstract base class for classes that generate new passive layouts.

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator._generate` method should be overridden to return:
        A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
        An optional "target" value for the layout

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.generate` method will return only the layout function and no target
    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.get_data` method will return both the layout function and the target
    """

    def __init__(
        self,
        volume: Volume,
        materials: Optional[List[str]] = None,
    ) -> None:
        r"""
        Initialises the generator for a given volume, in case any volume parameters are required by the inheriting generators

        Arguments:
            volume: Volume that the passive laypout will be loaded into
            materials: list of material names that can be used in the volume, None -> all materials known to TomOpt
        """

        self.volume = volume
        self.device = self.volume.device
        if materials is None:
            materials = [m for m in X0]
        self.materials = materials
        self.lw = volume.lw.detach().cpu().numpy()
        self.z_range = [z.detach().cpu().item() for z in self.volume.get_passive_z_range()]
        self.size = volume.passive_size

    @abstractmethod
    def _generate(self) -> Tuple[RadLengthFunc, Optional[Tensor]]:
        r"""
        Inheriting classes should override this.

        Returns:
            RadLengthFunc: A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
            Target: An optional "target" value for the layout
        """

        pass

    def get_data(self) -> Tuple[RadLengthFunc, Optional[Tensor]]:
        r"""
        Returns:
            RadLengthFunc: A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
            Target: An optional "target" value for the layout
        """

        return self._generate()

    def generate(self) -> RadLengthFunc:
        r"""
        Returns:
            The layout function and no target
        """

        f, _ = self._generate()
        return f


class AbsBlockPassiveGenerator(AbsPassiveGenerator):
    r"""
    Abstract base class for classes that generate new passive layouts which contain a single cuboid of material (block).

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator._generate` method should be overridden to return:
        A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
        An optional "target" value for the layout

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.generate` method will return only the layout function and no target
    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.get_data` method will return both the layout function and the target
    """

    def __init__(
        self,
        volume: Volume,
        block_size: Optional[Tuple[float, float, float]],
        block_size_max_half: Optional[bool] = None,
        materials: Optional[List[str]] = None,
    ) -> None:
        r"""
        Initialises the generator for a given volume, in case any volume parameters are required by the inheriting generators.
        The block will be centered radmonly in the volume, and can either be of fixed or random size.

        Arguments:
            volume: Volume that the passive laypout will be loaded into
            block_size: if set, will generate blocks of the specified size and random orientation, otherwise will randomly set the size of the blocks
            block_size_max_half: if True and block_size is None, the maximum size of blocks will be set to half the size of the passive volume
            materials: list of material names that can be used in the volume, None -> all materials known to TomOpt
        """

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
    r"""
    Generates new passive layouts which contain a single cuboid of material (block) of random material against a random background material.
    Blocks are always present, but can potentially be of the same material as the background.
    The target for the volumes is the X0 of the block material.

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.generate` method will return only the layout function and no target
    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.get_data` method will return both the layout function and the target
    """

    def __init__(
        self,
        volume: Volume,
        block_size: Optional[Tuple[float, float, float]],
        sort_x0: bool,
        enforce_diff_mat: bool,
        block_size_max_half: Optional[bool] = None,
        materials: Optional[List[str]] = None,
    ) -> None:
        r"""
        Initialises the generator for a given volume, in case any volume parameters are required by the inheriting generators.
        The block will be centered radmonly in the volume, and can either be of fixed or random size.

        Arguments:
            volume: Volume that the passive laypout will be loaded into
            block_size: if set, will generate blocks of the specified size and random orientation, otherwise will randomly set the size of the blocks
            sort_x0: if True, the block will always have a lower X0 than the background, unless they are of the same material
            enforce_diff_mat: if True, the block will always be of a different material to the background
            block_size_max_half: if True and block_size is None, the maximum size of blocks will be set to half the size of the passive volume
            materials: list of material names that can be used in the volume, None -> all materials known to TomOpt
        """

        super().__init__(volume=volume, block_size=block_size, materials=materials, block_size_max_half=block_size_max_half)
        self.sort_x0, self.enforce_diff_mat = sort_x0, enforce_diff_mat

    def _generate(self) -> Tuple[RadLengthFunc, Tensor]:
        r"""
        Generates passive layouts containing a (randomly sized) block of random material at a random location surrounded by a random background.

        Returns:
            RadLengthFunc: A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
            Target: The X0 of the block material
        """

        bkg_mat, block_mat = None, None
        while bkg_mat is None or block_mat is None or (bkg_mat == block_mat and self.enforce_diff_mat):
            bkg_mat = np.random.randint(0, len(self.materials))
            block_mat = np.random.randint(0, len(self.materials))
        base_x0 = X0[self.materials[bkg_mat]]
        block_x0 = X0[self.materials[block_mat]]
        if self.sort_x0 and block_x0 > base_x0:
            block_x0, base_x0 = base_x0, block_x0

        block_low, block_high = self._get_block_coords()

        def generator(*, z: Tensor, lw: Tensor, size: float) -> Tensor:
            shp = (lw / size).long()
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list(shp)) * base_x0
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = block_x0
            return rad_length

        return generator, Tensor([block_x0])


class BlockPresentPassiveGenerator(AbsBlockPassiveGenerator):
    r"""
    Generates new passive layouts which contain a single cuboid of material (block) of random material against a fixed background material.
    Blocks are always present, but can potentially be of the same material as the background.
    The target for the volumes is the X0 of the block material.
    The background material for the background will always be the zeroth material provided during initialisation.

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.generate` method will return only the layout function and no target
    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.get_data` method will return both the layout function and the target
    """

    def _generate(self) -> Tuple[RadLengthFunc, Tensor]:
        r"""
        Generates passive layouts containing a (randomly sized) block of random material at a random location surrounded by a fixed background.
        The background material for the background will always be the zeroth material provided during initialisation.

        Returns:
            RadLengthFunc: A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
            Target: The X0 of the block material
        """

        bkg_mat = 0
        block_mat = np.random.randint(0, len(self.materials))
        base_x0 = X0[self.materials[bkg_mat]]
        block_x0 = X0[self.materials[block_mat]]

        block_low, block_high = self._get_block_coords()

        def generator(*, z: Tensor, lw: Tensor, size: float) -> Tensor:
            shp = (lw / size).long()
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list(shp)) * base_x0
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = block_x0
            return rad_length

        return generator, Tensor([block_x0])


class VoxelPassiveGenerator(AbsPassiveGenerator):
    r"""
    Generates new passive layouts where every voxel is of a random material.

    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.generate` method will return only the layout function and no target
    The :meth:`~tomopt.optimisation.data.passives.AbsPassiveGenerator.get_data` method will return both the layout function and the target
    """

    def _generate(self) -> Tuple[RadLengthFunc, None]:
        r"""
        Generates new passive layouts where ever voxel is of a random material.

        Returns:
            RadLengthFunc: A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
            Target: None
        """

        def generator(*, z: Tensor, lw: Tensor, size: float) -> Tensor:
            x0s = lw.new_tensor([X0[m] for m in self.materials])
            shp = (lw / size).long()
            return x0s[torch.randint(high=len(x0s), size=(shp.prod().numpy(),), device=x0s.device)].reshape(list(shp))

        return generator, None


class PassiveYielder:
    r"""
    Dataset class that can either:
        Yield from a set of prespecified passive-volume layouts, and optional targets
        Generate and yield random layouts and optional targets from a provided generator
    """

    def __init__(
        self,
        passives: Union[List[Union[Tuple[RadLengthFunc, Optional[Tensor]], RadLengthFunc]], AbsPassiveGenerator],
        n_passives: Optional[int] = None,
        shuffle: bool = True,
    ):
        r"""
        Arguments:
            passives: Either a list of passive-volume functions (and optional targets together in a tuple), or a passive-volume generator
            n_passives: if a generator is used, this determines the number of volumes to generator per epoch in training, or in total when predicting
            shuffle: If a list of prespecified layouts is provided, their order will be shuffled if this is True
        """

        self.passives, self.n_passives, self.shuffle = passives, n_passives, shuffle
        if isinstance(self.passives, AbsPassiveGenerator):
            if self.n_passives is None:
                raise ValueError("If a AbsPassiveGenerator class is used, n_passives must be specified")
        else:
            self.n_passives = len(self.passives)

    def __len__(self) -> int:
        return self.n_passives

    def __iter__(self) -> Generator[Tuple[RadLengthFunc, Optional[Tensor]], None, None]:
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
