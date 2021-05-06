import math

import torch
from torch import Tensor

__all__ = ["generate_batch"]


def generate_batch(n: int, p: float = 5) -> Tensor:
    r"""
    Return tensor is (muons, coords),
    coords = (x~Uniform[0,1], y~Uniform[0,1], momentum (fixed), theta_x~cos2(a) a~Uniform[0,0.5pi], theta_y~Uniform[0,2pi])
    """

    batch = torch.stack(
        [
            torch.rand(n),
            torch.rand(n),
            torch.zeros(n) + p,
            torch.clamp(torch.randn(n) / 10, -math.pi / 2, math.pi / 2),  # Fix this
            torch.clamp(torch.randn(n) / 10, -math.pi / 2, math.pi / 2),  # Fix this
        ],
        dim=1,
    )
    return batch
