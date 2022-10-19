from typing import Callable, Iterator

import torch
from torch import nn

r"""
Common global constants, custom variable types, etc.
"""

__all__ = ["DEVICE", "SCATTER_COEF_A", "SCATTER_COEF_B", "X0", "DENSITIES", "PartialOpt"]


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf
SCATTER_COEF_A = 0.0136  # GeV
SCATTER_COEF_B = 0.038  # Dimensionless

# TODO: Update to numbers and materials used in scattering model
X0 = {  # https://pdg.lbl.gov/2022/AtomicNuclearProperties/index.html
    "beryllium": 0.3528,  # m
    "graphite": 0.1932,
    "silicon": 0.0937,
    "aluminium": 0.08897,
    "iron": 0.01757,
    "copper": 0.01436,
    "lead": 0.005612,
    "uranium": 3.141e-3,
    "air": 303.9,
    "water": 0.3608,
    "SiO2": 0.1229,
    "Al2O3": 0.07038,
    "Fe2O3": 0.03242,
    "MgO": 0.07828,
    "CaO": 0.05760,
    "toluene": 0.5068,
    "benzene": 0.4984,
    "K2O": 0.08147,
    "Na2O": 0.1285,
    "soft tissue": 0.3763,
}

DENSITIES = {  # https://pdg.lbl.gov/2022/AtomicNuclearProperties/index.html
    "water": 1000.0,  # kg/m^3
    "SiO2": 2200.0,
    "Al2O3": 3970.0,
    "Fe2O3": 5200.0,
    "MgO": 3580.0,
    "CaO": 3300.0,
    "graphite": 2210.0,
    "air": 1.205,
    "toluene": 866.9,
    "benzene": 878.7,
    "K2O": 2320.0,
    "Na2O": 2270.0,
    "soft tissue": 1000.0,
}

PartialOpt = Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
