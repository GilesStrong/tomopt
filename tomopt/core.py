from typing import Callable, Iterator

import torch
from torch import nn

__all__ = ["DEVICE", "SCATTER_COEF_A", "SCATTER_COEF_B", "X0", "PartialOpt"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

SCATTER_COEF_A = 0.0136

SCATTER_COEF_B = 0.038

# TODO: Update to numbers and materials used in scattering model

X0 = {  # Is actually X0/density "inverse scaled scattering density"
    "beryllium": 0.3528,  # m
    "carbon": 0.1932,
    "silicon": 0.0937,
    "aluminium": 0.08897,
    "iron": 0.01757,
    "copper": 0.01436,
    "lead": 0.005612,
    "uranium": 3.141e-3
    # 'air':312.22
}

PartialOpt = Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
