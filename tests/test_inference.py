import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.volume import PassiveLayer, DetectorLayer, Volume
from tomopt.muon import MuonBatch, generate_batch
from tomopt.core import X0
from tomopt.inference import ScatterBatch, X0Inferer

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


def get_layers(init_res: float = 1e4):
    layers = []
    init_eff = 0.5
    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(DetectorLayer(pos=pos, init_eff=init_eff, init_res=init_res, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost))
        else:
            pos = "below"
            layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


@pytest.fixture
def scatter_batch():
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_layers())
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    return mu, volume, sb


def test_scatter_batch_properties(scatter_batch):
    mu, volume, sb = scatter_batch

    assert sb.hits["above"]["z"].shape == mu.get_hits(LW)["above"]["z"].shape

    assert (loc_unc := sb.location_unc.mean() / sb.location.abs().mean()) < 10
    assert (dxy_unc := (sb.dxy_unc / sb.dxy).abs().mean()) < 10
    assert (dtheta_unc := (sb.dtheta_unc / sb.dtheta).mean()) < 10
    assert (theta_out_unc := sb.theta_out_unc.mean() / sb.theta_out.abs().mean()) < 10
    assert (theta_in_unc := sb.theta_in_unc.mean() / sb.theta_in.abs().mean()) < 10

    sb.plot_scatter(0)

    mask = sb.get_scatter_mask()
    assert sb.location[mask][:, 2].max() < 0.8
    assert sb.location[mask][:, 2].min() > 0.2
    assert mask.sum() > N / 4  # At least a quarter of the muons stay inside volume and scatter loc inside passive volume

    # Resolution increase improves location uncertainty
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_layers(init_res=1e7))
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    assert sb.location_unc.mean() / sb.location.abs().mean() < loc_unc
    assert sb.dxy_unc.mean() / sb.dxy.abs().mean() < dxy_unc
    assert sb.dtheta_unc.mean() / sb.dtheta.abs().mean() < dtheta_unc
    assert sb.theta_out_unc.mean() / sb.theta_out.abs().mean() < theta_out_unc
    assert sb.theta_in_unc.mean() / sb.theta_in.abs().mean() < theta_in_unc


def test_scatter_batch_compute(mocker, scatter_batch):  # noqa F811
    mu, volume = scatter_batch[0], scatter_batch[1]
    hits = {
        "above": {
            "xy": Tensor([[[0.0, 0.0], [0.1, 0.1]]]),
            "z": Tensor(
                [
                    [[1.0], [0.9]],
                ]
            ),
        },
        "below": {
            "xy": Tensor(
                [
                    [[0.1, 0.1], [0.0, 0.0]],
                ]
            ),
            "z": Tensor(
                [
                    [[0.1], [0.0]],
                ]
            ),
        },
    }
    mocker.patch.object(mu, "get_hits", return_value=hits)
    mocker.patch("tomopt.volume.layer.Layer.abs2idx", return_value=torch.zeros((3, 2), dtype=torch.long))

    def mock_jac(y: Tensor, x: Tensor, create_graph: bool = False, allow_unused: bool = True) -> Tensor:
        return torch.zeros(y.shape + x.shape)

    mocker.patch("tomopt.inference.scattering.jacobian", mock_jac)

    sb = ScatterBatch(mu=mu, volume=volume)
    assert (sb.location - Tensor([[0.5, 0.5, 0.5]])).sum().abs() < 1e-5
    assert (sb.dxy - Tensor([[0.0, 0.0]])).sum().abs() < 1e-5
    assert (sb.theta_in - Tensor([[-math.pi / 4, -math.pi / 4]])).sum().abs() < 1e-3
    assert (sb.theta_out - Tensor([[math.pi / 4, math.pi / 4]])).sum().abs() < 1e-3
    assert (sb.dtheta - Tensor([[math.pi / 2, math.pi / 2]])).sum().abs() < 1e-5


def test_x0_inferer_properties(scatter_batch):
    mu, volume, sb = scatter_batch
    inferer = X0Inferer(scatters=sb, default_pred=X0["beryllium"])

    assert inferer.mu == mu
    assert inferer.volume == volume
    assert len(inferer.hits["above"]["z"]) == len(mu.get_hits(LW)["above"]["z"])
    assert torch.all(inferer.lw == LW)
    assert inferer.size == SZ


def test_x0_inferer_methods(scatter_batch):
    mu, volume, sb = scatter_batch
    inferer = X0Inferer(scatters=sb, default_pred=X0["beryllium"])

    pt, pt_unc = inferer.x0_from_dtheta()
    assert len(pt) == len(sb.location[sb.get_scatter_mask()])
    assert pt.shape == pt_unc.shape
    assert (pt_unc / pt).mean() < 10

    pxy, pxy_unc = inferer.x0_from_dxy()
    assert pxy is None and pxy_unc is None  # modify tests when  dxy predictions implemented

    eff = inferer.compute_efficiency()
    assert (eff - 0.0625).abs().mean() < 1e-5

    p, w = inferer.average_preds(x0_dtheta=pt, x0_dtheta_unc=pt_unc, x0_dxy=pxy, x0_dxy_unc=pxy_unc, efficiency=eff)
    true = volume.get_rad_cube()
    assert p.shape == true.shape
    assert w.shape == true.shape
    assert (((p - true)[p == p]).abs() / true[p == p]).mean() < 100

    p2, w2 = inferer.pred_x0()
    assert p2.shape == true.shape
    assert w2.shape == true.shape
    assert (p2 != p2).sum() == 0  # No NaNs
    assert (((p2 - true)).abs() / true).mean() < 100
