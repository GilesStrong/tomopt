from functools import partial
from unittest.mock import patch

import torch
from torch import Tensor, nn

from tomopt.core import X0
from tomopt.optimisation import VoxelX0Loss
from tomopt.optimisation.callbacks.diagnostic_callbacks import HitRecord, ScatterRecord
from tomopt.optimisation.data.passives import PassiveYielder
from tomopt.optimisation.wrapper import PanelVolumeWrapper
from tomopt.plotting import plot_hit_density, plot_pred_true_x0, plot_scatter_density
from tomopt.volume import DetectorPanel, PanelDetectorLayer, PassiveLayer, Volume

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
    if z >= 0.4 and z <= 0.5:
        rad_length[5:, 5:] = X0["lead"]
    return rad_length


def get_layers(init_res: float = 1e5, init_eff: float = 0.9, n_panels: int = 4, init_xy_span=[3.0, 3.0]) -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=init_xy_span)
                for i in range(n_panels)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=init_xy_span)
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


@patch("matplotlib.pyplot.show")
def test_plot_pred_true_x0(mock_show):
    volume = Volume(get_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(torch.optim.SGD, lr=5e4),
        z_pos_opt=partial(torch.optim.SGD, lr=5e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e4),
        loss_func=VoxelX0Loss(target_budget=0),
    )
    preds = vw.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=100, mu_bs=100)
    plot_pred_true_x0(*preds[0])


@patch("matplotlib.pyplot.show")
def test_plot_scatter_density(mock_show):
    volume = Volume(get_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(torch.optim.SGD, lr=5e4),
        z_pos_opt=partial(torch.optim.SGD, lr=5e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e4),
        loss_func=VoxelX0Loss(target_budget=0),
    )
    sr = ScatterRecord()
    vw.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=100, mu_bs=100, cbs=[sr])
    df = sr.get_record(True)
    plot_scatter_density(df)


@patch("matplotlib.pyplot.show")
def test_plot_hit_density(mock_show):
    volume = Volume(get_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(torch.optim.SGD, lr=5e4),
        z_pos_opt=partial(torch.optim.SGD, lr=5e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e4),
        loss_func=VoxelX0Loss(target_budget=0),
    )
    hr = HitRecord()
    vw.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=100, mu_bs=100, cbs=[hr])
    df = hr.get_record(True)
    plot_hit_density(df)
