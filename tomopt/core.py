from typing import Callable, Iterator

import torch
from torch import nn

__all__ = ["DEVICE", "SCATTER_COEF_A", "SCATTER_COEF_B", "X0", "PartialOpt"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf
SCATTER_COEF_A = 0.0136  # GeV
SCATTER_COEF_B = 0.038  # Dimensionless

X0 = {  # https://pdg.lbl.gov/2022/AtomicNuclearProperties/index.html
    "beryllium": 0.3528,  # m
    "carbon": 0.1932,
    "silicon": 0.0937,
    "aluminium": 0.08897,
    "iron": 0.01757,
    "copper": 0.01436,
    "lead": 0.005612,
    "uranium": 3.141e-3,
    "air": 312.22,
    "water": 0.3608,
    "SiO2": 0.116577,
    "Al2O3": 0.07038,
    "Fe2O3": 0.03242,
    "MgO": 0.07828,
    "CaO": 0.05760,
}


PartialOpt = Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
