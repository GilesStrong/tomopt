import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.volume.layer import Layer
from tomopt.volume import PassiveLayer, DetectorLayer, Volume
from tomopt.muon import MuonBatch, generate_batch
from tomopt.core import X0
from tomopt.utils import jacobian


LW = Tensor([1, 1])
SZ = 0.1
N = 1000
Z = 1


@pytest.fixture
def batch():
    return MuonBatch(generate_batch(N), init_z=1)


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    r"""Test"""
    rad_length = torch.ones(list((lw / size).long())) * X0["aluminium"]
    if z >= 0.3 and z <= 0.6:
        rad_length[3:7, 3:7] = X0["lead"]
    return rad_length


def test_layer(batch):
    l = Layer(lw=LW, z=1, size=SZ)
    batch.x = 0.5
    batch.y = 0.7
    assert torch.all(l.mu_abs2idx(batch)[0] == Tensor([5, 7]))


def test_passive_layer_forwards(batch):
    # Normal scattering
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=Z, size=SZ)
    start = batch.copy()
    pl(batch)
    assert batch.z - Tensor([Z - SZ]) < 1e-5
    assert torch.all(batch.dr(start) > 0)
    assert torch.all(batch.xy != start.xy)

    # Small scattering
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=Z, size=1e-4)
    batch = start.copy()
    pl(batch, 1)
    assert batch.z - Tensor([Z]) <= 1e-4
    assert torch.all(batch.dr(start) < 1e-5)
    assert torch.all(batch.xy - start.xy < 1e-3)


@pytest.mark.parametrize("n", [(1), (2), (5)])
def test_passive_layer_scattering(mocker, batch, n):  # noqa: F811
    for m in ["propagate", "get_xy_mask"]:
        mocker.patch.object(MuonBatch, m)
    mock_getters = {}
    for m in ["theta_x", "theta_y", "x", "y", "p"]:
        mock_getters[m] = mocker.PropertyMock(return_value=getattr(batch, m))
        mocker.patch.object(MuonBatch, m, mock_getters[m])

    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, size=SZ, z=Z)
    pl(batch, n)
    assert batch.propagate.call_count == n
    assert batch.propagate.called_with(SZ / n)
    assert batch.get_xy_mask.call_count == n
    assert mock_getters["p"].call_count == n
    for m in ["x", "y"]:
        assert mock_getters[m].call_count == 2 * n
    for m in ["theta_x", "theta_y"]:
        assert mock_getters[m].call_count == 4 * n


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def test_detector_layer(batch):
    dl = DetectorLayer(pos="above", init_eff=1, init_res=1e3, lw=LW, z=Z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
    assert dl.resolution.mean() == Tensor([1e3])
    assert dl.efficiency.mean() == Tensor([1])

    start = batch.copy()
    dl(batch)
    assert batch.z - Tensor([Z - SZ]) < 1e-5
    assert torch.all(batch.dr(start) == 0)  # Detector layers don't scatter
    assert torch.all(batch.xy != start.xy)

    hits = batch.get_hits(LW)
    assert len(hits) == 1
    assert hits["above"]["xy"].shape == torch.Size([batch.get_xy_mask(LW).sum(), 1, 2])
    assert hits["above"]["z"][0, 0, 0] == Z - (SZ / 2)  # Hits located at z-centre of layer

    grad = jacobian(hits["above"]["xy"][:, 0], dl.resolution).sum((-1, -2))
    assert ((grad == grad) * (grad != 0)).sum() == 2 * len(grad)  # every reco hit (x,y) is function of resolution


def get_layers():
    layers = []
    init_eff = 0.5
    init_res = 10000
    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(DetectorLayer(pos=pos, init_eff=init_eff, init_res=init_res, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost))
        else:
            pos = "below"
            layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


def test_volume(batch):
    layers = get_layers()
    volume = Volume(layers=layers)
    assert volume.get_detectors()[-1] == layers[-1]
    assert volume.get_passives()[-1] == layers[-3]
    assert torch.all(volume.lw == LW)
    assert volume.size == Tensor([SZ])
    assert volume.h == 10 * SZ
    with pytest.raises(AttributeError):
        volume.lw = 0
        volume.size = 0
        volume.h = 0
    zr = volume.get_passive_z_range()
    assert zr[0] - 0.2 < 1e-5
    assert zr[1] - 0.8 < 1e-5
    assert volume.get_cost() == 4001392.7500
