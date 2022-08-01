from typing import Tuple, Callable
import numpy as np

import torch
from torch import Tensor

from ...optimisation.data.passives import AbsPassiveGenerator
from ...volume import Volume
from ...core import X0, DENSITIES
from ...utils import x0_from_mixture

__all__ = ["SmallWallsPassiveGenerator"]


class SmallWallsPassiveGenerator(AbsPassiveGenerator):
    def __init__(
        self,
        volume: Volume,
        x0_soil: float = x0_from_mixture(
            [X0["SiO2"], X0["soft tissue"], X0["water"], X0["air"]],
            [DENSITIES["SiO2"], DENSITIES["soft tissue"], DENSITIES["water"], DENSITIES["air"]],
            volume_fracs=[0.44, 0.6, 0.25, 0.25],
        )[
            "X0"
        ],  # ~0.26m
        x0_wall: float = x0_from_mixture(
            [X0["SiO2"], X0["Al2O3"], X0["Fe2O3"], X0["MgO"], X0["CaO"], X0["Na2O"], X0["K2O"]],
            [DENSITIES["SiO2"], DENSITIES["Al2O3"], DENSITIES["Fe2O3"], DENSITIES["MgO"], DENSITIES["CaO"], DENSITIES["Na2O"], DENSITIES["K2O"]],
            volume_fracs=[56.0, 17.0, 11.2, 4.0, 4.2, 3.0, 4.6],
        )[
            "X0"
        ],  # 0.08m https://core.ac.uk/download/pdf/324142628.pdf
        stop_k: float = 10,
        turn_k: float = 5,
        min_length: int = 4,  # number of voxels
        min_height: int = 4,
    ) -> None:
        super().__init__(volume=volume, materials=["soil", "wall"])
        self.x0_soil, self.x0_wall, self.stop_k, self.turn_k, self.min_length, self.min_height = x0_soil, x0_wall, stop_k, turn_k, min_length, min_height
        self.zxy_shp = [int(np.round((self.z_range[1] - self.z_range[0]) / self.size))] + (self.lw / self.size).astype(int).tolist()
        self.wall_z_range = (self.z_range[0], self.z_range[1] - self.size)  # Atleast top layer is soil

    def _generate(self) -> Tuple[Callable[..., Tensor], Tensor]:
        n_walls = np.random.randint(1, 5)
        zxy_map = torch.zeros(self.zxy_shp)
        ground_z = np.random.randint(self.zxy_shp[0] // 2)
        wall_heights = []

        for _ in range(n_walls):
            wall_heights.append(np.random.randint(ground_z + self.min_height, self.zxy_shp[0]))
            x, y = np.random.randint(self.zxy_shp[1]), np.random.randint(self.zxy_shp[2])

            # Choose initial move direction, border start position limits direction
            moves = [0, 1, 2, 3]  # 0=up, 1=right, 2=down, 3=left
            if x == 0:
                moves.remove(3)
            elif x == self.zxy_shp[1] - 1:
                moves.remove(1)
            if y == 0:
                moves.remove(2)
            elif y == self.zxy_shp[2] - 1:
                moves.remove(0)
            move = np.random.choice(moves)

            length, n_since_last_turn = 0, 0
            while True:
                if (zxy_map[:, x, y] != 1).all():
                    zxy_map[ground_z : wall_heights[-1], x, y] = 1
                else:  # Wall meets another wall, or itself
                    break
                length += 1
                n_since_last_turn += 1

                if length >= self.min_length and np.random.random() < np.exp(-self.stop_k / length):  # Wall ends
                    break
                if np.random.random() < np.exp(-self.turn_k / n_since_last_turn):  # Wall turns
                    n_since_last_turn = 0
                    move = np.random.choice([0, 2] if move in [1, 3] else [1, 3])

                if move == 0:
                    y += 1
                elif move == 1:
                    x += 1
                elif move == 2:
                    y -= 1
                else:
                    x -= 1
                if (x >= self.zxy_shp[1]) or (x < 0) or (y >= self.zxy_shp[2]) or (y < 0):  # Wall moves outside volume
                    break

        x0_map = zxy_map * self.x0_wall
        x0_map[x0_map == 0] = self.x0_soil

        def generator(*, z: float, lw: Tensor, size: float) -> Tensor:
            return x0_map[np.round((z - self.z_range[0]) / self.size).long() - 1].squeeze()

        return generator, zxy_map
