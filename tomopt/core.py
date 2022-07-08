from typing import Callable, Iterator, Union, List, Dict
import numpy as np

import torch
from torch import nn

__all__ = ["DEVICE", "SCATTER_COEF_A", "SCATTER_COEF_B", "X0", "DENSITIES", "PartialOpt", "x0_from_mixture"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf
SCATTER_COEF_A = 0.0136  # GeV
SCATTER_COEF_B = 0.038  # Dimensionless

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
}


def x0_from_mixture(x0s: Union[np.ndarray, List[float]], densities: Union[np.ndarray, List[float]], fracs: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    if not isinstance(x0s, np.ndarray):
        x0s = np.array(x0s)
    if not isinstance(densities, np.ndarray):
        densities = np.array(densities)
    if not isinstance(fracs, np.ndarray):
        fracs = np.array(fracs)
    fracs = fracs / fracs.sum()

    x0rho = 1 / (fracs / (x0s * densities)).sum()
    rho = 1 / (fracs / densities).sum()
    x0 = x0rho / rho
    return {"X0": x0, "density": rho}


PartialOpt = Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
