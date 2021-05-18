from functools import partial
from pathlib import Path
from tomopt.loss.loss import DetectorLoss

import numpy as np

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.volume import Volume, PassiveLayer, DetectorLayer
from tomopt.optimisation.wrapper.volume_wrapper import VolumeWrapper

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
            layers.append(DetectorLayer(pos=pos, init_eff=init_eff, init_res=init_res, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost))
        else:
            pos = "below"
            layers.append(PassiveLayer(lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


def test_volume_wrapper_methods():
    volume = Volume(get_layers())
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))

    # _build_opt
    for l, r, e in zip(volume.get_detectors(), vw.res_opt.param_groups[0]["params"], vw.eff_opt.param_groups[0]["params"]):
        assert torch.all(l.resolution == r)
        assert torch.all(l.efficiency == e)

    # get_detectors
    for i, j in zip(volume.get_detectors(), vw.get_detectors()):
        assert i == j

    # get_param_count
    assert vw.get_param_count() == 4 * 2 * 10 * 10

    # save
    def zero_params(v, vw):
        for l in v.get_detectors():
            nn.init.zeros_(l.resolution)
            nn.init.zeros_(l.efficiency)
        assert l.resolution.sum() == 0
        assert l.efficiency.sum() == 0
        vw.res_lr = 0
        vw.eff_lr = 0

    p = Path("test_save.pt")
    vw.save("test_save.pt")
    assert p.exists()
    zero_params(volume, vw)

    vw.load(p)
    for l in volume.get_detectors():
        assert l.resolution.sum() != 0
        assert l.efficiency.sum() != 0
    vw.res_lr != 0
    vw.eff_lr != 0

    # from_save
    zero_params(volume, vw)
    vw = VolumeWrapper.from_save(p, volume=volume, res_opt=partial(optim.SGD, lr=0), eff_opt=partial(optim.Adam, lr=0), loss_func=DetectorLoss(0.15))
    for l in volume.get_detectors():
        assert l.resolution.sum() != 0
        assert l.efficiency.sum() != 0
    vw.res_lr != 0
    vw.eff_lr != 0


def test_volume_wrapper_parameters():
    volume = Volume(get_layers())
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))

    assert vw.eff_lr == 2e-5
    assert vw.res_lr == 2e1
    assert vw.eff_mom == 0.9
    assert vw.res_mom == 0.95

    vw.eff_lr = 2
    vw.res_lr = 2
    vw.eff_mom = 0.8
    vw.res_mom = 0.8

    assert vw.eff_lr == 2
    assert vw.res_lr == 2
    assert vw.eff_mom == 0.8
    assert vw.res_mom == 0.8
