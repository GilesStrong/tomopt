from functools import partial
from pathlib import Path
import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.volume import Volume, PassiveLayer, PanelDetectorLayer, DetectorHeatMap
from tomopt.optimisation.wrapper.volume_wrapper import FitParams, HeatMapVolumeWrapper
from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks.diagnostic_callbacks import ScatterRecord
from tomopt.optimisation.data.passives import PassiveYielder
from tomopt.optimisation.loss import VoxelX0Loss
from tomopt.optimisation.callbacks.grad_callbacks import NoMoreNaNs
from tomopt.optimisation.callbacks.data_callbacks import MuonResampler
from tomopt.muon.generation import MuonGenerator2016
from tomopt.optimisation.callbacks.pred_callbacks import PredHandler

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
    if z >= 0.4 and z <= 0.5:
        rad_length[5:, 5:] = X0["lead"]
    return rad_length


def get_heatmap_layers(init_res: float = 1e3) -> nn.ModuleList:
    def area_cost(a: Tensor) -> Tensor:
        return F.relu(a)

    layers = []
    init_eff = 0.9
    n_panels = 4
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorHeatMap(
                    res=init_res,
                    eff=init_eff,
                    init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)],
                    init_xy_span=[-0.5, 0.5],
                    area_cost_func=area_cost,
                )
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
                DetectorHeatMap(
                    res=init_res,
                    eff=init_eff,
                    init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)],
                    init_xy_span=[-0.5, 0.5],
                    area_cost_func=area_cost,
                )
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


def test_panel_volume_wrapper_methods():
    volume = Volume(get_heatmap_layers())
    vw = HeatMapVolumeWrapper(
        volume,
        mu_opt=partial(optim.SGD, lr=2e0, momentum=0.95),
        z_pos_opt=partial(optim.Adam, lr=2e-2),
        sig_opt=partial(optim.Adam, lr=2e-0),
        norm_opt=partial(optim.Adam, lr=2e-0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )

    # _build_opt
    for p, xy, z, s, n in zip(
        [p for l in volume.get_detectors() for p in l.panels],
        vw.opts["mu_opt"].param_groups[0]["params"],
        vw.opts["z_pos_opt"].param_groups[0]["params"],
        vw.opts["sig_opt"].param_groups[0]["params"],
        vw.opts["norm_opt"].param_groups[0]["params"],
    ):
        assert torch.all(p.mu == xy)
        assert torch.all(p.z == z)
        assert torch.all(p.sig == s)
        assert torch.all(p.norm == n)

    # get_detectors
    for i, j, k in zip(volume.get_detectors(), vw.get_detectors(), (volume.layers[0], volume.layers[-1])):
        assert i == j == k

    # get_param_count
    assert vw.get_param_count() == 4.0 * 2.0 * (30.0 * 4.0 + 1 + 1)

    # save
    def zero_params(v, vw):
        for l in v.get_detectors():
            for p in l.panels:
                nn.init.zeros_(p.mu)
                nn.init.zeros_(p.z)
                nn.init.zeros_(p.sig)
                nn.init.zeros_(p.norm)
                nn.init.zeros_(p.resolution)
                nn.init.zeros_(p.efficiency)
                nn.init.zeros_(p.xy_fix)
                nn.init.zeros_(p.xy_span_fix)

                assert p.mu.sum() == 0
                assert p.z.sum() == 0
                assert p.sig.sum() == 0
                assert p.norm.sum() == 0
                assert p.resolution.sum() == 0
                assert p.efficiency.sum() == 0
                assert p.xy_fix.sum() == 0
                assert p.xy_span_fix.abs().sum() == 0

        vw.set_opt_lr(0, "mu_opt")
        vw.set_opt_lr(0, "z_pos_opt")
        vw.set_opt_lr(0, "sig_opt")
        vw.set_opt_lr(0, "norm_opt")

    path = Path("tests/test_panel_save.pt")
    vw.save("tests/test_panel_save.pt")
    assert path.exists()
    zero_params(volume, vw)

    vw.load(path)
    for l in volume.get_detectors():
        for p in l.panels:
            assert p.mu.sum() != 0
            assert p.z.sum() != 0
            assert p.sig.sum() != 0
            assert p.norm.sum() != 0
            assert p.resolution.sum() != 0
            assert p.efficiency.sum() != 0
            assert p.xy_fix.sum() != 0
            assert p.xy_span_fix.abs().sum() != 0
    vw.get_opt_lr("mu_opt") != 0
    vw.get_opt_lr("z_pos_opt") != 0
    vw.get_opt_lr("sig_opt") != 0
    vw.get_opt_lr("norm_opt") != 0

    # from_save
    zero_params(volume, vw)
    vw = HeatMapVolumeWrapper.from_save(
        path,
        volume=volume,
        mu_opt=partial(optim.SGD, lr=0),
        z_pos_opt=partial(optim.Adam, lr=0),
        sig_opt=partial(optim.Adam, lr=0),
        norm_opt=partial(optim.Adam, lr=0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )
    for l in volume.get_detectors():
        for p in l.panels:
            assert p.mu.sum() != 0
            assert p.z.sum() != 0
            assert p.sig.sum() != 0
            assert p.norm.sum() != 0
            assert p.resolution.sum() != 0
            assert p.efficiency.sum() != 0
            assert p.xy_fix.sum() != 0
            assert p.xy_span_fix.abs().sum() != 0
    vw.get_opt_lr("mu_opt") != 0
    vw.get_opt_lr("z_pos_opt") != 0
    vw.get_opt_lr("sig_opt") != 0
    vw.get_opt_lr("norm_opt") != 0
