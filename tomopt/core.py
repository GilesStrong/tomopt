from typing import Callable, Iterator

import torch
from mypy_extensions import NamedArg
from torch import Tensor, nn

r"""
Common global constants, custom variable types, etc.
"""

__all__ = ["DEVICE", "SCATTER_COEF_A", "SCATTER_COEF_B", "X0", "DENSITIES", "Z", "A", "mean_excitation_E", "PartialOpt", "B", "PropertiesFunc", "RadLengthFunc"]


DEVICE = torch.device("cpu")  # Set to CPU even if CUDA available due to issue #53  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    "steel": 0.01782,
    "hot liquid steel": 0.01991,
    "slag": 0.08211,
}

DENSITIES = {  # https://pdg.lbl.gov/2022/AtomicNuclearProperties/index.html
    "iron": 7.87,  # g*cm^-3
    "aluminium": 2.699,
    "copper": 8.960,
    "lead": 11.35,
    "uranium": 19,
    "beryllium": 1.848,
    # "water": 1000.0,  # kg/m^3
    # "SiO2": 2200.0,
    # "Al2O3": 3970.0,
    # "Fe2O3": 5200.0,
    # "MgO": 3580.0,
    # "CaO": 3300.0,
    # "graphite": 2210.0,
    # "air": 1.205,
    # "toluene": 866.9,
    # "benzene": 878.7,
    # "K2O": 2320.0,
    # "Na2O": 2270.0,
    # "soft tissue": 1000.0,
    # "steel": 7818.0,
    # "hot liquid steel": 7000.0,
    # "slag": 2700.0,
}

A = {"iron": 55.845, "aluminium": 26.9815385, "copper": 63.546, "lead": 207.2, "uranium": 238.02891, "beryllium": 4}

Z = {"iron": 26, "aluminium": 13, "copper": 29, "lead": 82, "uranium": 92, "beryllium": 9}

B = {  # B parameter in Kuhn scattering model
    "iron": 9.669761283357756,
    "aluminium": 8.360943658395296,
    "copper": 9.800963860872987,
    "lead": 9.542446057727844,
    "uranium": 9.990760245428246,
    "beryllium": 9.60390753204349,
}

mean_excitation_E = {"iron": 286.0, "aluminium": 166.0, "copper": 322.0, "lead": 823.0, "uranium": 890.0, "beryllium": 63.7}  # eV

props = [X0, B, Z, A, DENSITIES, mean_excitation_E]

PartialOpt = Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
RadLengthFunc = Callable[[NamedArg(Tensor, "z"), NamedArg(Tensor, "lw"), NamedArg(float, "size")], Tensor]  # noqa F821
PropertiesFunc = Callable[[NamedArg(Tensor, "z"), NamedArg(Tensor, "lw"), NamedArg(float, "size")], Tensor]  # noqa F405
