import pytest
from pytest_mock import mocker

import torch
from torch import Tensor

from tomopt.volume.layer import Layer
from tomopt.volume import PassiveLayer
from tomopt.muon import MuonBatch, generate_batch
from tomopt.core import X0


LW = Tensor([1, 1])
SZ = 0.1
N = 1000


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
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=1, size=SZ)
    start = batch.copy()
    pl(batch)
    assert batch.z - Tensor([0.9]) < 1e-5
    assert torch.all(batch.dr(start) > 0)
    assert torch.all(batch.xy != start.xy)

    # Small scattering
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=1, size=1e-4)
    batch = start.copy()
    pl(batch, 1)
    assert batch.z - Tensor([1]) <= 1e-4
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

    # batch = MuonBatch(generate_batch(100), init_z=1)
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, size=SZ, z=1)
    pl(batch, n)
    assert batch.propagate.call_count == n
    assert batch.propagate.called_with(SZ / n)
    assert batch.get_xy_mask.call_count == n
    assert mock_getters["p"].call_count == n
    for m in ["x", "y"]:
        assert mock_getters[m].call_count == 2 * n
    for m in ["theta_x", "theta_y"]:
        assert mock_getters[m].call_count == 4 * n
