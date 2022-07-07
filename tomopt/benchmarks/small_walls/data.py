from typing import Tuple, Callable
import numpy as np

import torch
from torch import Tensor

from ...optimisation.data.passives import AbsPassiveGenerator
from ...volume import Volume
from ...core import X0

__all__ = ["SmallWallsPassiveGenerator"]


class SmallWallsPassiveGenerator(AbsPassiveGenerator):
    def __init__(
        self,
        volume: Volume,
        x0_soil: float = (0.5 * X0["carbon"]) + (0.25 * X0["water"]) + (0.25 * X0["air"]),
        x0_wall: float = (0.55 * X0["SiO2"]) + (0.30 * X0["Al2O3"]) + (0.08 * X0["Fe2O3"]) + (0.05 * X0["MgO"]) + (0.01 * X0["CaO"]) + (0.01 * X0["carbon"]),
        stop_k: float = 10,
        turn_k: float = 5,
        min_lenght: int = 3,
        min_height: int = 3,
    ) -> None:
        super().__init__(volume=volume, materials=["soil", "wall"])
        self.x0_soil, self.x0_wall, self.stop_k, self.turn_k, self.min_lenght, self.min_height = x0_soil, x0_wall, stop_k, turn_k, min_lenght, min_height
        self.zxy_shp = [int(np.round((self.z_range[1] - self.z_range[0]) / self.size))] + (self.lw / self.size).astype(int).tolist()
        self.wall_z_range = (self.z_range[0], self.z_range[1] - self.size)  # Atleast top layer is soil

    def _generate(self) -> Tuple[Callable[..., Tensor], Tensor]:
        n_walls = np.random.randint(1, 5)
        zxy_map = torch.zeros(self.zxy_shp)
        ground_z = np.random.randint(self.zxy_shp[0] // 2)
        wall_heights = []

        for _ in range(n_walls):
            wall_heights.append(np.random.randint(ground_z + self.min_height, self.zxy_shp[0] - 1))
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

                if length >= self.min_lenght and np.random.random() < np.exp(-self.stop_k / length):  # Wall ends
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
