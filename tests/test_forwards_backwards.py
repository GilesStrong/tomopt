import pytest
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.volume import Volume, PassiveLayer, VoxelDetectorLayer
from tomopt.muon import MuonBatch, generate_batch
from tomopt.inference import VoxelScatterBatch, X0Inferer
from tomopt.optimisation.loss import DetectorLoss

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["aluminium"]
    if z >= 0.5:
        rad_length[3:7, 3:7] = X0["lead"]
    return rad_length


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def get_layers(init_res: float = 1e3):
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
            layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


@pytest.fixture
def inferer():
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_layers())
    volume(mu)
    sb = VoxelScatterBatch(mu=mu, volume=volume)
    return X0Inferer(sb)


def test_forwards(inferer):
    pred, weight = inferer.pred_x0()
    loss_func = DetectorLoss(1e-5)
    loss_val = loss_func(pred, weight, inferer.volume)

    for l in inferer.volume.get_detectors():
        assert torch.nan_to_num(torch.autograd.grad(loss_val, l.resolution, retain_graph=True, allow_unused=True)[0].abs(), 0).sum() > 0
        assert torch.autograd.grad(loss_val, l.efficiency, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


def test_backwards(inferer):
    pred, weight = inferer.pred_x0()
    loss_func = DetectorLoss(0.15)
    loss_val = loss_func(pred, weight, inferer.volume)
    opt = torch.optim.SGD(inferer.volume.parameters(), lr=1)
    opt.zero_grad()
    loss_val.backward()
    for p in inferer.volume.parameters():
        assert p.grad is not None
    opt.step()
    for l in inferer.volume.get_detectors():
        assert l.resolution.mean() != 1e4
        assert l.efficiency.mean() != 0.5
