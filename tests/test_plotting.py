from functools import partial
import numpy as np
from unittest.mock import patch

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

from tomopt.optimisation.data.passives import PassiveYielder
from tomopt.volume import VoxelDetectorLayer, PassiveLayer, Volume
from tomopt.optimisation.wrapper import VoxelVolumeWrapper
from tomopt.optimisation import VoxelX0Loss
from tomopt.plotting import plot_pred_true_x0, plot_scatter_density, plot_hit_density
from tomopt.optimisation.callbacks.diagnostic_callbacks import ScatterRecord, HitRecord
from tomopt.core import X0

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
    if z >= 0.4 and z <= 0.5:
        rad_length[5:, 5:] = X0["lead"]
    return rad_length


def get_layers(init_res: float = 1e4):
    def eff_cost(x: Tensor) -> Tensor:
        return torch.expm1(3 * F.relu(x))

    def res_cost(x: Tensor) -> Tensor:
        return F.relu(x / 100) ** 2

    layers = []
    init_eff = 0.5
    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(
                VoxelDetectorLayer(pos=pos, init_eff=init_eff, init_res=init_res, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
            )
        else:
            pos = "below"
            layers.append(PassiveLayer(lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


@patch("matplotlib.pyplot.show")
def test_plot_pred_true_x0(mock_show):
    volume = Volume(get_layers())
    vw = VoxelVolumeWrapper(
        volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15)
    )
    preds = vw.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=100, mu_bs=100)
    plot_pred_true_x0(*preds[0])


@patch("matplotlib.pyplot.show")
def test_plot_scatter_density(mock_show):
    volume = Volume(get_layers())
    vw = VoxelVolumeWrapper(
        volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15)
    )
    sr = ScatterRecord()
    vw.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=100, mu_bs=100, cbs=[sr])
    df = sr.get_record(True)
    plot_scatter_density(df)


@patch("matplotlib.pyplot.show")
def test_plot_hit_density(mock_show):
    volume = Volume(get_layers())
    vw = VoxelVolumeWrapper(
        volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15)
    )
    hr = HitRecord()
    vw.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=100, mu_bs=100, cbs=[hr])
    df = hr.get_record(True)
    plot_hit_density(df)
