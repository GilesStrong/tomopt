import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np
import math
from unittest.mock import patch
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.volume import PassiveLayer, VoxelDetectorLayer, Volume, PanelDetectorLayer, DetectorPanel
from tomopt.muon import MuonBatch, generate_batch
from tomopt.core import X0
from tomopt.inference import VoxelScatterBatch, VoxelX0Inferer, PanelX0Inferer, PanelScatterBatch
from tomopt.volume.layer import Layer

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


def area_cost(x: Tensor) -> Tensor:
    return F.relu(x / 1000) ** 2


def get_voxel_layers(init_res: float = 1e4, init_eff: float = 0.5) -> nn.ModuleList:
    layers = []
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


def get_panel_layers(init_res: float = 1e4, init_eff: float = 0.5, n_panels: int = 4) -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=[0.5, 0.5], area_cost_func=area_cost)
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
                DetectorPanel(
                    res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=[0.5, 0.5], area_cost_func=area_cost
                )
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def voxel_scatter_batch() -> Tuple[MuonBatch, Volume, VoxelScatterBatch]:
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_voxel_layers())
    volume(mu)
    sb = VoxelScatterBatch(mu=mu, volume=volume)
    return mu, volume, sb


@pytest.fixture
def panel_scatter_batch() -> Tuple[MuonBatch, Volume, PanelScatterBatch]:
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_panel_layers())
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)
    return mu, volume, sb


@pytest.mark.flaky(max_runs=3, min_passes=2)
@patch("matplotlib.pyplot.show")
def test_scatter_batch_properties_voxel(mock_show, voxel_scatter_batch):
    mu, volume, sb = voxel_scatter_batch

    assert sb.hits["above"]["z"].shape == mu.get_hits(LW)["above"]["z"].shape

    assert (loc_xy_unc := sb.location_unc[:, :2].mean()) < 0.5
    assert (loc_z_unc := sb.location_unc[:, 2].mean()) < 1.5
    assert (dxy_unc := sb.dxy_unc.mean()) < 1.0
    assert (dtheta_unc := (sb.dtheta_unc / sb.dtheta).mean()) < 10
    assert (theta_out_unc := sb.theta_out_unc.mean() / sb.theta_out.abs().mean()) < 10
    assert (theta_in_unc := sb.theta_in_unc.mean() / sb.theta_in.abs().mean()) < 10

    sb.plot_scatter(0)

    mask = sb.get_scatter_mask()
    assert sb.location[mask][:, 2].max() < 0.8
    assert sb.location[mask][:, 2].min() > 0.2
    assert mask.sum() > N / 4  # At least a quarter of the muons stay inside volume and scatter loc inside passive volume

    for l in volume.get_detectors():
        assert torch.autograd.grad(sb.dtheta.sum(), l.resolution, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    # Resolution increase improves location uncertainty
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_voxel_layers(init_res=1e7))
    volume(mu)
    sb = VoxelScatterBatch(mu=mu, volume=volume)
    assert sb.location_unc[:, :2].mean() < loc_xy_unc
    assert sb.location_unc[:, 2].mean() < loc_z_unc
    assert sb.dxy_unc.mean() < dxy_unc
    assert sb.dtheta_unc.mean() / sb.dtheta.abs().mean() < dtheta_unc
    assert sb.theta_out_unc.mean() / sb.theta_out.abs().mean() < theta_out_unc
    assert sb.theta_in_unc.mean() / sb.theta_in.abs().mean() < theta_in_unc


# @pytest.mark.flaky(max_runs=3, min_passes=2)
@patch("matplotlib.pyplot.show")
def test_scatter_batch_properties_panel(mock_show, panel_scatter_batch):
    mu, volume, sb = panel_scatter_batch

    assert sb.hits["above"]["z"].shape == mu.get_hits(LW)["above"]["z"].shape

    assert (loc_xy_unc := sb.location_unc[:, :2].mean()) < 0.5
    assert (loc_z_unc := sb.location_unc[:, 2].mean()) < 1.5
    assert (dxy_unc := sb.dxy_unc.mean()) < 1.0
    assert (dtheta_unc := (sb.dtheta_unc / sb.dtheta).mean()) < 10
    assert (theta_out_unc := sb.theta_out_unc.mean() / sb.theta_out.abs().mean()) < 10
    assert (theta_in_unc := sb.theta_in_unc.mean() / sb.theta_in.abs().mean()) < 10

    sb.plot_scatter(0)

    mask = sb.get_scatter_mask()
    assert sb.location[mask][:, 2].max() < 0.8
    assert sb.location[mask][:, 2].min() > 0.2
    assert mask.sum() > N / 4  # At least a quarter of the muons stay inside volume and scatter loc inside passive volume

    for l in volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(sb.dtheta.sum(), p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(sb.dtheta.sum(), p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(sb.dtheta.sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    # Resolution increase improves location uncertainty
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_panel_layers(init_res=1e7))
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)
    assert sb.location_unc[:, :2].mean() < loc_xy_unc
    assert sb.location_unc[:, 2].mean() < loc_z_unc
    assert sb.dxy_unc.mean() < dxy_unc
    assert sb.dtheta_unc.mean() / sb.dtheta.abs().mean() < dtheta_unc
    assert sb.theta_out_unc.mean() / sb.theta_out.abs().mean() < theta_out_unc
    assert sb.theta_in_unc.mean() / sb.theta_in.abs().mean() < theta_in_unc


def test_scatter_batch_trajectory_fit():
    # 2 Hits
    xa0 = Tensor([[0, 0, 1]])
    xa1 = Tensor([[1, 1, 0]])
    # Same unc
    traj = VoxelScatterBatch.get_muon_trajectory([xa0, xa1], [Tensor([[1, 1]]), Tensor([[1, 1]])], lw=Tensor([1, 1]))
    assert (traj == Tensor([[1, 1, -1]])).all()
    # Different unc
    traj = VoxelScatterBatch.get_muon_trajectory([xa0, xa1], [Tensor([[10, 10]]), Tensor([[1, 1]])], lw=Tensor([1, 1]))
    assert (traj == Tensor([[1, 1, -1]])).all()

    # 3 Hits inline
    xa2 = Tensor([[0.5, 0.5, 0.5]])
    # Same unc
    traj = VoxelScatterBatch.get_muon_trajectory([xa0, xa1, xa2], [Tensor([[1, 1]]), Tensor([[1, 1]]), Tensor([[1, 1]])], lw=Tensor([1, 1]))
    assert (traj == Tensor([[1, 1, -1]])).all()
    # Different unc
    traj = VoxelScatterBatch.get_muon_trajectory([xa0, xa1, xa2], [Tensor([[10, 10]]), Tensor([[1, 1]]), Tensor([[1, 1]])], lw=Tensor([1, 1]))
    assert (traj == Tensor([[1, 1, -1]])).all()

    # 3 Hits offline
    xa0 = Tensor([[0, 0, 1]])
    xa1 = Tensor([[0, 1, 0.5]])
    xa2 = Tensor([[1, 0, 0.5]])
    # Same unc
    traj = VoxelScatterBatch.get_muon_trajectory([xa0, xa1, xa2], [Tensor([[1, 1]]), Tensor([[1, 1]]), Tensor([[1, 1]])], lw=Tensor([1, 1]))
    assert (traj - Tensor([[0.5, 0.5, -0.5]])).sum() < 1e-5
    # Different unc
    traj = VoxelScatterBatch.get_muon_trajectory([xa0, xa1, xa2], [Tensor([[1, 1]]), Tensor([[1e9, 1e9]]), Tensor([[1, 1]])], lw=Tensor([1, 1]))
    assert (traj - Tensor([[1, 0, -0.5]])).sum() < 1e-5


def test_scatter_batch_compute(mocker, voxel_scatter_batch):  # noqa F811
    mu, volume = voxel_scatter_batch[0], voxel_scatter_batch[1]
    hits = {
        "above": {
            "reco_xy": Tensor([[[0.0, 0.0], [0.1, 0.1]]]),
            "gen_xy": Tensor([[[0.0, 0.0], [0.1, 0.1]]]),
            "z": Tensor(
                [
                    [[1.0], [0.9]],
                ]
            ),
        },
        "below": {
            "reco_xy": Tensor(
                [
                    [[0.1, 0.1], [0.0, 0.0]],
                ]
            ),
            "gen_xy": Tensor(
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
    mocker.patch("tomopt.volume.layer.Layer.abs2idx", return_value=torch.zeros((1, 3), dtype=torch.long))

    def mock_jac(y: Tensor, x: Tensor, create_graph: bool = False, allow_unused: bool = True) -> Tensor:
        return torch.zeros(y.shape + x.shape)

    mocker.patch("tomopt.inference.scattering.jacobian", mock_jac)

    sb = VoxelScatterBatch(mu=mu, volume=volume)
    assert (sb.location - Tensor([[0.5, 0.5, 0.5]])).sum().abs() < 1e-3
    assert (sb.dxy - Tensor([[0.0, 0.0]])).sum().abs() < 1e-3
    assert (sb.theta_in - Tensor([[-math.pi / 4, -math.pi / 4]])).sum().abs() < 1e-3
    assert (sb.theta_out - Tensor([[math.pi / 4, math.pi / 4]])).sum().abs() < 1e-3
    assert (sb.dtheta - Tensor([[math.pi / 2, math.pi / 2]])).sum().abs() < 1e-3


def test_x0_inferer_properties(voxel_scatter_batch):
    mu, volume, sb = voxel_scatter_batch
    inferer = VoxelX0Inferer(scatters=sb, default_pred=X0["beryllium"])

    assert inferer.mu == mu
    assert inferer.volume == volume
    assert len(inferer.hits["above"]["z"]) == len(mu.get_hits(LW)["above"]["z"])
    assert torch.all(inferer.lw == LW)
    assert inferer.size == SZ


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_x0_inferer_methods_voxel():
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_voxel_layers(init_res=1e3))
    volume(mu)
    sb = VoxelScatterBatch(mu=mu, volume=volume)
    inferer = VoxelX0Inferer(scatters=sb, default_pred=X0["beryllium"])

    pt, pt_unc = inferer.x0_from_dtheta()
    assert len(pt) == len(sb.location[sb.get_scatter_mask()])
    assert pt.shape == pt_unc.shape
    assert (pt_unc / pt).mean() < 10

    for l in volume.get_detectors():
        assert torch.autograd.grad(pt.abs().sum(), l.resolution, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
        assert torch.autograd.grad(pt_unc.abs().sum(), l.resolution, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    pxy, pxy_unc = inferer.x0_from_dxy()
    assert pxy is None and pxy_unc is None  # modify tests when dxy predictions implemented

    eff = inferer.compute_efficiency()
    assert (eff - 0.0625).abs().mean() < 1e-5

    p, w = inferer.average_preds(x0_dtheta=pt, x0_dtheta_unc=pt_unc, x0_dxy=pxy, x0_dxy_unc=pxy_unc, efficiency=eff)
    true = volume.get_rad_cube()
    assert p.shape == true.shape
    assert w.shape == true.shape
    mask = p == p
    assert (((p - true)[mask]).abs() / true[mask]).mean() < 100

    p2, w2 = inferer.pred_x0()
    assert p2.shape == true.shape
    assert w2.shape == true.shape
    assert (p2 != p2).sum() == 0  # NaNs replaced with default prediction
    assert (p2_mse := (((p2 - true)).abs() / true).mean()) < 100  # noqa F841
    print(torch.autograd.grad(p2.abs().sum(), l.resolution, retain_graph=True, allow_unused=True))

    for l in volume.get_detectors():
        assert torch.autograd.grad(p2.abs().sum(), l.resolution, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
        assert torch.autograd.grad(p2.abs().sum(), l.efficiency, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
        assert torch.autograd.grad(w2.abs().sum(), l.resolution, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
        assert torch.autograd.grad(w2.abs().sum(), l.efficiency, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    p2, w2 = inferer.pred_x0(inc_default=False)
    assert (p2 != p2).sum() >= 0  # NaNs NOT replaced with default prediction


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_x0_inferer_methods_panel():
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_panel_layers(init_res=1e3))
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)
    inferer = PanelX0Inferer(scatters=sb, default_pred=X0["beryllium"])

    pt, pt_unc = inferer.x0_from_dtheta()
    assert len(pt) == len(sb.location[sb.get_scatter_mask()])
    assert pt.shape == pt_unc.shape
    assert (pt_unc / pt).mean() < 10

    for l in volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(pt.abs().sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(pt_unc.abs().sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    pxy, pxy_unc = inferer.x0_from_dxy()
    assert pxy is None and pxy_unc is None  # modify tests when dxy predictions implemented

    eff = inferer.compute_efficiency()
    assert (eff > 0).all()

    p, w = inferer.average_preds(x0_dtheta=pt, x0_dtheta_unc=pt_unc, x0_dxy=pxy, x0_dxy_unc=pxy_unc, efficiency=eff)
    true = volume.get_rad_cube()
    assert p.shape == true.shape
    assert w.shape == true.shape
    mask = p == p
    assert (((p - true)[mask]).abs() / true[mask]).mean() < 100

    p2, w2 = inferer.pred_x0()
    assert p2.shape == true.shape
    assert w2.shape == true.shape
    assert (p2 != p2).sum() == 0  # NaNs replaced with default prediction
    assert (p2_mse := (((p2 - true)).abs() / true).mean()) < 100  # noqa F841

    for l in volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(p2.abs().sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(p2.abs().sum(), p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(p2.abs().sum(), p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w2.abs().sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w2.abs().sum(), p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w2.abs().sum(), p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    p2, w2 = inferer.pred_x0(inc_default=False)
    assert (p2 != p2).sum() >= 0  # NaNs NOT replaced with default prediction


def test_x0_inferer_scatter_inversion(mocker, voxel_scatter_batch):  # noqa F811
    layer = Layer(LW, Z, SZ)
    mu, volume, sb = voxel_scatter_batch
    inferer = VoxelX0Inferer(scatters=sb, default_pred=X0["beryllium"])
    x0 = X0["lead"]
    n_x0 = layer._compute_n_x0(x0=x0, deltaz=SZ, theta=mu.theta)
    mocker.patch("tomopt.volume.layer.torch.randn", lambda n, device: torch.ones(n, device=device))  # remove randomness
    dx, dy, dtheta_x, dtheta_y = layer._compute_displacements(n_x0=n_x0, deltaz=SZ, theta_x=mu.theta_x, theta_y=mu.theta_y, mom=mu.mom)
    dtheta = torch.stack([dtheta_x, dtheta_y], dim=1)

    sb._dtheta = dtheta
    sb._dtheta_unc = torch.ones_like(dtheta)
    sb._theta_in = mu.theta_xy
    sb._theta_in_unc = torch.ones_like(dtheta)
    sb._theta_out = mu.theta_xy + dtheta
    sb._theta_out_unc = torch.ones_like(dtheta)
    mask = torch.ones_like(n_x0) > 0
    inferer.mask = mask
    mocker.patch.object(mu, "get_xy_mask", return_value=mask)

    mocker.patch("tomopt.inference.rad_length.jacobian", lambda i, j: torch.ones((len(i), 1, 2), device=i.device))  # remove randomness
    pred, _ = inferer.x0_from_dtheta()
    assert (pred.mean() - x0) < 1e-5
