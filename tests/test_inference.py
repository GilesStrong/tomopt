import math
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from pytest_lazyfixture import lazy_fixture
from pytest_mock import mocker  # noqa F401
from torch import Tensor, nn

from tomopt.core import X0
from tomopt.inference import (
    DenseBlockClassifierFromX0s,
    GenScatterBatch,
    PanelX0Inferrer,
    ScatterBatch,
)
from tomopt.inference.volume import AbsIntClassifierFromX0
from tomopt.muon import MuonBatch, MuonGenerator2016
from tomopt.optimisation import MuonResampler
from tomopt.utils import jacobian
from tomopt.volume import (
    DetectorPanel,
    PanelDetectorLayer,
    PassiveLayer,
    SigmoidDetectorPanel,
    Volume,
)

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


def get_panel_layers(init_res: float = 1e5, init_eff: float = 0.9, n_panels: int = 4, init_xy_span=[3.0, 3.0]) -> nn.ModuleList:
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


def get_sigmoid_panel_layers(smooth=1.0, init_res: float = 1e5, init_eff: float = 0.9, n_panels: int = 4, init_xy_span=[3.0, 3.0]) -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                SigmoidDetectorPanel(smooth=smooth, res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=init_xy_span)
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
                SigmoidDetectorPanel(smooth=smooth, res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=init_xy_span)
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def panel_scatter_batch() -> Tuple[MuonBatch, Volume, ScatterBatch]:
    volume = Volume(get_panel_layers())
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    return mu, volume, sb


@pytest.fixture
def sigmoid_panel_scatter_batch() -> Tuple[MuonBatch, Volume, ScatterBatch]:
    volume = Volume(get_sigmoid_panel_layers())
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    return mu, volume, sb


@pytest.mark.flaky(max_runs=3, min_passes=2)
@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("scatter_batch", [lazy_fixture("panel_scatter_batch"), lazy_fixture("sigmoid_panel_scatter_batch")])
def test_panel_scatter_batch(mock_show, scatter_batch):
    mu, volume, sb = scatter_batch
    assert len(sb) == len(mu)

    # hits
    hits = mu.get_hits()
    assert sb.n_hits_above == 4
    assert sb.n_hits_below == 4
    for i in range(4):
        assert (sb.above_hits[:, i] == hits["above"]["reco_xyz"][:, i]).all()
        assert (sb.below_hits[:, i] == hits["below"]["reco_xyz"][:, i]).all()
        assert (sb.above_gen_hits[:, i] == hits["above"]["gen_xyz"][:, i]).all()
        assert (sb.below_gen_hits[:, i] == hits["below"]["gen_xyz"][:, i]).all()

    assert (loc_xy_unc := sb.poca_xyz_unc[:, :2].mean()) < 1.0
    assert (loc_z_unc := sb.poca_xyz_unc[:, 2].mean()) < 1.5
    assert (dxy_unc := sb.dxy_unc.mean()) < 1.0
    assert sb.dtheta_unc.mean() / sb.dtheta.mean() < 10
    assert sb.dphi_unc.mean() / sb.dphi.mean() < 10
    assert (theta_msc_unc := (sb.theta_msc_unc / sb.theta_msc).mean()) < 10
    assert (theta_out_unc := sb.theta_out_unc.mean() / sb.theta_out.abs().mean()) < 10
    assert (theta_in_unc := sb.theta_in_unc.mean() / sb.theta_in.abs().mean()) < 10
    assert sb.phi_out_unc.mean() / sb.phi_out.abs().mean() < 10
    assert sb.phi_in_unc.mean() / sb.phi_in.abs().mean() < 10

    # range check
    assert (sb.theta_in >= 0).all() and (sb.theta_in < torch.pi / 2).all()
    assert (sb.theta_out >= 0).all() and (sb.theta_out < torch.pi / 2).all()
    assert (sb.phi_in >= 0).all() and (sb.phi_in < 2 * torch.pi).all() and (sb.phi_in.min() < torch.pi) and (sb.phi_in.max() > torch.pi)
    assert (sb.phi_out >= 0).all() and (sb.phi_out < 2 * torch.pi).all() and (sb.phi_out.min() < torch.pi) and (sb.phi_out.max() > torch.pi)
    assert (sb.theta_xy_in > -torch.pi / 2).all() and (sb.theta_xy_in < torch.pi / 2).all() and (sb.theta_xy_in.min() < 0) and (sb.theta_xy_in.max() > 0)
    assert (sb.theta_xy_out > -torch.pi / 2).all() and (sb.theta_xy_out < torch.pi / 2).all() and (sb.theta_xy_out.min() < 0) and (sb.theta_xy_out.max() > 0)
    assert (sb.dtheta_xy >= 0).all() and (sb.dtheta_xy < torch.pi).all()
    assert (sb.dtheta >= 0).all() and (sb.dtheta < torch.pi).all()
    assert (sb.dphi >= 0).all() and (sb.dphi < torch.pi).all()
    assert (sb.theta_msc.abs() >= 0).all() and (sb.theta_msc < torch.pi).all()

    # uncertainties
    uncs = sb.hit_uncs
    assert (uncs[0][:, 2] == 0).all()
    xy_unc = uncs[0][:, :2]
    assert xy_unc.min().item() > 0
    assert xy_unc.max().item() < 0.1
    assert xy_unc.std().item() > 0

    # efficiencies
    effs = sb.hit_effs
    assert effs.min().item() > 0
    assert effs.max().item() < 1
    assert effs.std().item() > 0

    sb.plot_scatter(0)

    mask = sb.get_scatter_mask()
    assert sb.poca_xyz[mask][:, 2].max() < 0.8
    assert sb.poca_xyz[mask][:, 2].min() > 0.2
    assert mask.sum() > N / 4  # At least a quarter of the muons stay inside volume and scatter loc inside passive volume

    for l in volume.get_detectors():
        for p in l.panels:
            assert jacobian(sb.theta_msc, p.xy).abs().sum() > 0
            assert jacobian(sb.theta_msc, p.z).abs().sum() > 0
            assert jacobian(sb.theta_msc, p.xy_span).abs().sum() > 0

    # Resolution increase improves location uncertainty
    volume = Volume(get_panel_layers(init_res=1e7))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    assert sb.poca_xyz_unc[:, :2].mean() < loc_xy_unc
    assert sb.poca_xyz_unc[:, 2].mean() < loc_z_unc
    assert sb.dxy_unc.mean() < dxy_unc
    assert sb.theta_msc_unc.mean() / sb.theta_msc.abs().mean() < theta_msc_unc
    assert sb.theta_out_unc.mean() / sb.theta_out.abs().mean() < theta_out_unc
    assert sb.theta_in_unc.mean() / sb.theta_in.abs().mean() < theta_in_unc


def test_scatter_batch_trajectory_fit():
    # 2 Hits
    xa0 = Tensor([[0, 0, 1]])
    xa1 = Tensor([[1, 1, 0]])
    # Same unc
    traj, start = ScatterBatch.get_muon_trajectory(torch.stack([xa0, xa1], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1]))
    assert (traj == Tensor([[1, 1, -1]])).all()
    assert (start == xa0).all()
    # Different unc
    traj, _ = ScatterBatch.get_muon_trajectory(torch.stack([xa0, xa1], dim=1), torch.stack([Tensor([[10, 10]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1]))
    assert (traj == Tensor([[1, 1, -1]])).all()

    # 3 Hits inline
    xa2 = Tensor([[0.5, 0.5, 0.5]])
    # Same unc
    traj, _ = ScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj == Tensor([[1, 1, -1]])).all()
    # Different unc
    traj, _ = ScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[10, 10]]), Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj == Tensor([[1, 1, -1]])).all()

    # 3 Hits offline
    xa0 = Tensor([[0, 0, 1]])
    xa1 = Tensor([[0, 1, 0.5]])
    xa2 = Tensor([[1, 0, 0.5]])
    # Same unc
    traj, _ = ScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj - Tensor([[0.5, 0.5, -0.5]])).sum() < 1e-5
    # Different unc
    traj, _ = ScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1e9, 1e9]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj - Tensor([[1, 0, -0.5]])).sum() < 1e-5


def test_scatter_batch_compute(mocker, panel_scatter_batch):  # noqa F811
    mu, volume = panel_scatter_batch[0], panel_scatter_batch[1]
    hits = {
        "above": {
            "reco_xyz": Tensor([[[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]]]),
            "gen_xyz": Tensor([[[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]]]),
            "unc_xyz": Tensor([[[1, 1, 1], [1, 1, 1]]]),
            "eff": Tensor([[[1], [1]]]),
        },
        "below": {
            "reco_xyz": Tensor(
                [
                    [[0.1, 0.0, 0.1], [0.0, 0.0, 0.0]],
                ]
            ),
            "gen_xyz": Tensor(
                [
                    [[0.1, 0.0, 0.1], [0.0, 0.0, 0.0]],
                ]
            ),
            "unc_xyz": Tensor([[[1, 1, 1], [1, 1, 1]]]),
            "eff": Tensor([[[1], [1]]]),
        },
    }
    mocker.patch.object(mu, "get_hits", return_value=hits)
    mocker.patch("tomopt.volume.layer.PassiveLayer.abs2idx", return_value=torch.zeros((1, 3), dtype=torch.long))

    def mock_jac(y: Tensor, x: Tensor, create_graph: bool = False, allow_unused: bool = True) -> Tensor:
        return torch.zeros(y.shape + x.shape)

    mocker.patch("tomopt.inference.scattering.jacobian", mock_jac)

    sb = ScatterBatch(mu=mu, volume=volume)
    assert (sb.poca_xyz - Tensor([[0.0, 0.5, 0.5]])).sum().abs() < 1e-3
    assert (sb.dxy - Tensor([[0.0, 0.0]])).sum().abs() < 1e-3
    assert (sb.theta_in - (torch.pi / 4)).sum().abs() < 1e-3
    assert (sb.theta_out - (torch.pi / 4)).sum().abs() < 1e-3
    assert (sb.dtheta - (torch.pi / 2)).sum().abs() < 1e-3  # Smallest scattering is in dtheta, rather than dphi
    assert (sb.phi_in).sum().abs() < 1e-3
    assert (sb.phi_out - torch.pi).sum().abs() < 1e-3
    assert (sb.dphi).sum().abs() < 1e-3
    assert (sb.dtheta_xy - Tensor([[0, torch.pi / 2]])).sum().abs() < 1e-3
    assert (sb.theta_msc - Tensor([torch.pi / 2])).sum().abs() < 1e-3

    # Entry exit points
    assert sb.xyz_in[:, 1].sum().abs() < 1e-3
    assert sb.xyz_out[:, 1].sum().abs() < 1e-3
    assert (sb.xyz_in[:, 0] - Tensor([0.2])).sum().abs() < 1e-3
    assert (sb.xyz_out[:, 0] - Tensor([0.2])).sum().abs() < 1e-3
    assert (sb.xyz_in[:, 2] - Tensor([0.8])).sum().abs() < 1e-3
    assert (sb.xyz_out[:, 2] - Tensor([0.2])).sum().abs() < 1e-3


def test_gen_scatter_batch_compute(mocker, panel_scatter_batch):  # noqa F811
    mu, volume = panel_scatter_batch[0], panel_scatter_batch[1]
    hits = {
        "above": {
            "reco_xyz": Tensor([[[10.0, -2.0, 1.0], [1, 0.3, 0.9]]]),
            "gen_xyz": Tensor([[[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]]]),
            "unc_xyz": Tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "eff": Tensor([[[1], [1]]]),
        },
        "below": {
            "reco_xyz": Tensor(
                [
                    [[np.nan, 0.1, 0.1], [10.0, -20.0, 0.0]],
                ]
            ),
            "gen_xyz": Tensor(
                [
                    [[0.1, 0.0, 0.1], [0.0, 0.0, 0.0]],
                ]
            ),
            "unc_xyz": Tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "eff": Tensor([[[1], [1]]]),
        },
    }
    mocker.patch.object(mu, "get_hits", return_value=hits)
    mocker.patch("tomopt.volume.layer.PassiveLayer.abs2idx", return_value=torch.zeros((1, 3), dtype=torch.long))

    sb = GenScatterBatch(mu=mu, volume=volume)

    assert (sb.poca_xyz - Tensor([[0.0, 0.5, 0.5]])).sum().abs() < 1e-3
    assert (sb.dxy - Tensor([[0.0, 0.0]])).sum().abs() < 1e-3
    assert (sb.theta_in - (torch.pi / 4)).sum().abs() < 1e-3
    assert (sb.theta_out - (torch.pi / 4)).sum().abs() < 1e-3
    assert (sb.dtheta - (torch.pi / 2)).sum().abs() < 1e-3  # Smallest scattering is in dtheta, rather than dphi
    assert (sb.phi_in).sum().abs() < 1e-3
    assert (sb.phi_out - torch.pi).sum().abs() < 1e-3
    assert (sb.dphi).sum().abs() < 1e-3
    assert (sb.dtheta_xy - Tensor([[0, torch.pi / 2]])).sum().abs() < 1e-3
    assert (sb.theta_msc - Tensor([torch.pi / 2])).sum().abs() < 1e-3

    # Entry exit points
    assert sb.xyz_in[:, 1].sum().abs() < 1e-3
    assert sb.xyz_out[:, 1].sum().abs() < 1e-3
    assert (sb.xyz_in[:, 0] - Tensor([0.2])).sum().abs() < 1e-3
    assert (sb.xyz_out[:, 0] - Tensor([0.2])).sum().abs() < 1e-3
    assert (sb.xyz_in[:, 2] - Tensor([0.8])).sum().abs() < 1e-3
    assert (sb.xyz_out[:, 2] - Tensor([0.2])).sum().abs() < 1e-3


def test_abs_volume_inferrer_properties(panel_scatter_batch):
    mu, volume, sb = panel_scatter_batch
    inferrer = PanelX0Inferrer(volume=volume)

    assert inferrer.volume == volume
    assert torch.all(inferrer.lw == LW)
    assert inferrer.size == SZ


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_panel_x0_inferrer_methods(mocker):  # noqa F811
    volume = Volume(get_panel_layers(init_res=1e5))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inferrer = PanelX0Inferrer(volume=volume)

    pt = inferrer.x0_from_scatters(deltaz=SZ, total_scatter=sb.total_scatter, theta_in=sb.theta_in, theta_out=sb.theta_out, mom=sb.mu.reco_mom[:, None])
    assert len(pt) == len(sb.poca_xyz)
    assert pt.shape == torch.Size([len(sb.mu), 1])

    mask = ~pt.isnan()
    for l in volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(pt[mask].abs().sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    eff = inferrer.compute_efficiency(scatters=sb)
    assert eff[:, None].shape == pt.shape
    assert (eff > 0).all()

    # Single batch
    inferrer.add_scatters(sb)
    assert len(inferrer.scatter_batches) == 1
    for p in [
        inferrer._n_mu,
        inferrer._muon_scatter_vars,
        inferrer._muon_scatter_var_uncs,
        inferrer._muon_probs_per_voxel_zxy,
        inferrer._muon_efficiency,
        inferrer._vox_zxy_x0_preds,
        inferrer._vox_zxy_x0_pred_uncs,
    ]:
        assert p is None

    inferrer._combine_scatters()
    assert inferrer.n_mu == len(sb)
    for p in [inferrer._muon_scatter_vars, inferrer._muon_scatter_var_uncs, inferrer._muon_efficiency]:
        assert isinstance(p, Tensor) and len(p) == len(sb)
    for p in [inferrer._muon_probs_per_voxel_zxy, inferrer._vox_zxy_x0_preds, inferrer._vox_zxy_x0_pred_uncs]:
        assert p is None
    inferrer._reset_vars()
    for p in [
        inferrer._n_mu,
        inferrer._muon_scatter_vars,
        inferrer._muon_scatter_var_uncs,
        inferrer._muon_probs_per_voxel_zxy,
        inferrer._muon_efficiency,
        inferrer._vox_zxy_x0_preds,
        inferrer._vox_zxy_x0_pred_uncs,
    ]:
        assert p is None

    mocker.spy(inferrer, "_combine_scatters")
    mocker.spy(inferrer, "_get_voxel_zxy_x0_preds")
    inferrer.vox_zxy_x0_preds
    assert inferrer._combine_scatters.call_count == 1
    assert inferrer._get_voxel_zxy_x0_preds.call_count == 1

    p1, w1 = inferrer.get_prediction()
    assert inferrer._combine_scatters.call_count == 1
    assert inferrer._get_voxel_zxy_x0_preds.call_count == 1
    assert isinstance(inferrer._muon_probs_per_voxel_zxy, Tensor)

    true = volume.get_rad_cube()
    assert p1.shape == true.shape
    assert w1.shape == torch.Size([])
    assert inferrer._muon_probs_per_voxel_zxy.shape == torch.Size([len(sb)]) + true.shape
    assert (p1 != p1).sum() == 0  # No NaNs
    assert (((p1 - true)).abs() / true).mean() < 100

    for l in volume.get_detectors():
        for panel in l.panels:
            assert torch.autograd.grad(p1.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(p1.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(p1.abs().sum(), panel.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w1.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w1.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    # Multiple batches
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb2 = ScatterBatch(mu=mu, volume=volume)
    inferrer.add_scatters(sb2)
    for p in [
        inferrer._n_mu,
        inferrer._muon_scatter_vars,
        inferrer._muon_scatter_var_uncs,
        inferrer._muon_probs_per_voxel_zxy,
        inferrer._muon_efficiency,
        inferrer._vox_zxy_x0_preds,
        inferrer._vox_zxy_x0_pred_uncs,
    ]:
        assert p is None

    assert len(inferrer.scatter_batches) == 2
    assert inferrer.n_mu == len(sb) + len(sb2)
    assert inferrer._combine_scatters.call_count == 2

    p2, w2 = inferrer.get_prediction()  # Averaged prediction slightly changes with new batch
    assert (p2 - p1).abs().sum() > 1e-2
    assert (w2 - w1).abs().sum() > 1e-2
    assert inferrer._get_voxel_zxy_x0_preds.call_count == 2


def test_panel_inferrer_multi_batch():
    volume = Volume(get_panel_layers(init_res=1e5))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(1000), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)

    # one batch
    inf = PanelX0Inferrer(volume=volume)
    inf.add_scatters(ScatterBatch(mu=mu, volume=volume))
    pred1 = inf.get_prediction()

    # multi-batch
    inf = PanelX0Inferrer(volume=volume)
    for i in range(4):
        mask = torch.zeros(len(mu))
        mask[250 * i : 250 * (i + 1)] = 1
        mask = mask.bool()
        mu_batch = mu.copy()
        mu_batch.filter_muons(mask)
        for pos in mu._hits:
            for var in mu._hits[pos]:
                for xy_pos in mu._hits[pos][var]:
                    mu_batch._hits[pos][var].append(xy_pos[mask])
        inf.add_scatters(ScatterBatch(mu=mu_batch, volume=volume))
    pred4 = inf.get_prediction()

    assert (((pred1 - pred4) / pred1).abs() < 1e-4).all()


def test_panel_x0_inferrer_efficiency(mocker, panel_scatter_batch):  # noqa F811
    mu, volume, sb = panel_scatter_batch
    inferrer = PanelX0Inferrer(volume=volume)
    effs = Tensor([0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    true_eff = 0.4263  # from MC sim

    for i, e in enumerate(effs):
        sb._hit_effs[:, i] = e

    assert (inferrer.compute_efficiency(scatters=sb)[0] - true_eff).abs() < 1e-3


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_x0_inferrer_scatter_inversion(mocker, panel_scatter_batch):  # noqa F811
    mu, volume, sb = panel_scatter_batch
    inferrer = PanelX0Inferrer(volume=volume)
    inferrer.size = SZ
    x0 = torch.ones_like(mu.theta) * X0["lead"]
    mocker.patch("tomopt.volume.layer.torch.randn", lambda n, device: torch.ones(n, device=device))  # remove randomness
    layer = PassiveLayer(LW, Z, SZ, step_sz=SZ / mu.theta.cos())
    scatters = layer._pdg_scatter(
        x0=x0, theta=mu.theta, phi=mu.phi, theta_x=mu.theta_x, theta_y=mu.theta_y, mom=mu.mom, log_term=False, step_sz=SZ / mu.theta.cos()
    )
    dtheta_x_m, dtheta_y_m = scatters["dtheta_x_m"], scatters["dtheta_x_m"]

    mu_start = mu.copy()
    mu.scatter_dtheta_xy(dtheta_x_vol=scatters["dtheta_x_vol"], dtheta_y_vol=scatters["dtheta_y_vol"])
    pred = inferrer.x0_from_scatters(
        deltaz=SZ,
        total_scatter=torch.sqrt((dtheta_x_m**2) + (dtheta_y_m**2)) / math.sqrt(2),
        theta_in=mu_start.theta,
        theta_out=mu.theta,
        mom=mu.mom,
    )

    assert (pred.mean() - X0["lead"]).abs() < 1e-4


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_x0_inferrer_scatter_inversion_via_scatter_batch():
    gen = MuonGenerator2016((0, 1), (0, 1), fixed_mom=5)
    muons = MuonBatch(gen(10000), init_z=1)
    x0 = torch.ones(len(muons)) * X0["lead"]
    dz = 0.01
    layer = PassiveLayer(lw=Tensor([1, 1]), z=1, size=0.1, step_sz=dz / muons.theta.cos())
    pdg_scattering = layer._pdg_scatter(
        x0=x0, theta=muons.theta, theta_x=muons.theta_x, theta_y=muons.theta_y, mom=muons.mom, log_term=False, phi=muons.phi, step_sz=dz / muons.theta.cos()
    )

    xyz = F.pad(muons.xy.detach().clone(), (0, 1))
    xyz[:, 2] = muons.z.detach().clone()
    muons.append_hits(
        {
            "reco_xyz": xyz,
            "gen_xyz": xyz,
            "unc_xyz": torch.zeros(len(muons), 3),
            "eff": torch.ones(len(muons), 1),
        },
        pos="above",
    )
    muons.propagate_dz(0.1)
    xyz = F.pad(muons.xy.detach().clone(), (0, 1))
    xyz[:, 2] = muons.z.detach().clone()
    muons.append_hits(
        {
            "reco_xyz": xyz,
            "gen_xyz": xyz,
            "unc_xyz": torch.zeros(len(muons), 3),
            "eff": torch.ones(len(muons), 1),
        },
        pos="above",
    )
    muons.propagate_dz(dz)
    muons.scatter_dtheta_xy(dtheta_x_vol=pdg_scattering["dtheta_x_vol"], dtheta_y_vol=pdg_scattering["dtheta_y_vol"])
    xyz = F.pad(muons.xy.detach().clone(), (0, 1))
    xyz[:, 2] = muons.z.detach().clone()
    muons.append_hits(
        {
            "reco_xyz": xyz,
            "gen_xyz": xyz,
            "unc_xyz": torch.zeros(len(muons), 3),
            "eff": torch.ones(len(muons), 1),
        },
        pos="below",
    )
    muons.propagate_dz(0.1)
    xyz = F.pad(muons.xy.detach().clone(), (0, 1))
    xyz[:, 2] = muons.z.detach().clone()
    muons.append_hits(
        {
            "reco_xyz": xyz,
            "gen_xyz": xyz,
            "unc_xyz": torch.zeros(len(muons), 3),
            "eff": torch.ones(len(muons), 1),
        },
        pos="below",
    )

    class MockVolume:
        lw = Tensor([10, 10])
        passive_size = dz
        device = torch.device("cpu")

        def get_passive_z_range(self):
            return (0.89, 0.9)

        def get_passives(self):
            return [1]

    volume = MockVolume()
    sb = GenScatterBatch(muons, volume=volume)
    inferrer = PanelX0Inferrer(volume=volume)

    pred = inferrer.x0_from_scatters(
        deltaz=dz,
        total_scatter=(sb.total_scatter).square().mean().sqrt() / math.sqrt(2),
        theta_in=sb.theta_in.square().mean().sqrt(),
        theta_out=sb.theta_out.square().mean().sqrt(),
        mom=sb.mu.mom[:, None].square().mean().sqrt(),
    )

    assert (pred - X0["lead"]).abs() < 3e-3


# @pytest.mark.flaky(max_runs=2, min_passes=1)
# @pytest.mark.parametrize("dvi_class, weighted", [[DeepVolumeInferrer, False], [WeightedDeepVolumeInferrer, True]])
# def test_deep_volume_inferrer(dvi_class: Type[DeepVolumeInferrer], weighted: bool):
#     volume = Volume(get_panel_layers(init_res=1e4))
#     gen = MuonGenerator2016.from_volume(volume)
#     mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
#     mu = MuonBatch(mus, init_z=volume.h)
#     volume(mu)
#     sb = ScatterBatch(mu=mu, volume=volume)
#     nvalid = len(mu)

#     grp_feats = ["pred_x0", "track_xy", "delta_angles", "theta_msc", "track_angles", "poca", "dpoca", "voxels"]  # 1  # 4  # 2  # 1  # 4  # 3  # 3->4  # 0->3
#     n_infeats = 18

#     class MockModel(nn.Module):
#         def __init__(self) -> None:
#             super().__init__()
#             self.layer = nn.Linear(600 * (n_infeats + 4 + weighted), 1)
#             self.act = nn.Sigmoid()

#         def forward(self, x: Tensor) -> Tensor:
#             return self.act(self.layer(x.mean(2).flatten()[None]))

#     inferrer = dvi_class(model=MockModel(), base_inferrer=PanelX0Inferrer(volume=volume), volume=volume, grp_feats=grp_feats, include_unc=True)

#     pt, pt_unc = inferrer.get_base_predictions(scatters=sb)
#     assert len(pt) == len(sb.poca_xyz)
#     assert pt.shape == pt_unc.shape

#     assert len(inferrer.in_vars) == 0
#     assert len(inferrer.in_var_uncs) == 0
#     assert len(inferrer.efficiencies) == 0
#     inferrer.add_scatters(sb)
#     assert len(inferrer.in_vars) == 1
#     assert len(inferrer.in_var_uncs) == 1
#     assert len(inferrer.efficiencies) == 1
#     assert inferrer.in_vars[-1].shape == torch.Size((nvalid, n_infeats))
#     assert inferrer.in_var_uncs[-1].shape == torch.Size((nvalid, n_infeats))
#     assert len(inferrer.efficiencies[-1]) == nvalid

#     inputs = inferrer._build_inputs(inferrer.in_vars[0])
#     assert inputs.shape == torch.Size((600, nvalid, n_infeats + 4))  # +4 since voxels and dpoca_r

#     pred = inferrer.get_prediction()
#     assert pred.shape == torch.Size((1, 1))

#     for l in volume.get_detectors():
#         for panel in l.panels:
#             assert torch.autograd.grad(pred.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(pred.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(pred.abs().sum(), panel.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_dense_block_classifier_from_x0s():
    volume = Volume(get_panel_layers(init_res=1e4))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)

    def u_rad_length(*, z: float, lw: Tensor, size: float) -> Tensor:
        rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
        if z > 0.4 and z <= 0.5:
            rad_length[7:, 6:] = X0["uranium"]
        return rad_length

    volume.load_rad_length(u_rad_length)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inferrer = DenseBlockClassifierFromX0s(12, PanelX0Inferrer, volume=volume)
    inferrer.add_scatters(sb)

    p, w = inferrer.get_prediction()
    for l in volume.get_detectors():
        for panel in l.panels:
            assert torch.autograd.grad(p.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(p.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(p.abs().sum(), panel.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


def test_abs_int_classifier_from_x0():
    volume = Volume(get_panel_layers(init_res=1e4))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)

    class Inf(AbsIntClassifierFromX0):
        def x02probs(self, vox_preds: Tensor) -> Tensor:
            return F.softmax(vox_preds.mean([1, 2]), dim=-1)

    # Raw probs
    inferrer = Inf(partial_x0_inferrer=PanelX0Inferrer, volume=volume, output_probs=True)
    inferrer.add_scatters(sb)

    p, w = inferrer.get_prediction()
    assert p.shape == torch.Size([6])
    assert w.shape == torch.Size([])

    for l in volume.get_detectors():
        for panel in l.panels:
            assert jacobian(p, panel.xy_span).abs().sum() > 0
            assert jacobian(p, panel.xy).abs().sum() > 0
            assert jacobian(p, panel.z).abs().sum() > 0
            assert torch.autograd.grad(w.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(w.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    # Single prediction
    inferrer = Inf(partial_x0_inferrer=PanelX0Inferrer, volume=volume, output_probs=False)
    inferrer.add_scatters(sb)
    p, w = inferrer.get_prediction()
    assert p.type() == "torch.LongTensor"
    assert p.shape == torch.Size([])
    assert w.shape == torch.Size([])

    # Single float prediction
    inferrer = Inf(partial_x0_inferrer=PanelX0Inferrer, volume=volume, output_probs=False, class2float=lambda x, v: 3.5 * x)
    inferrer.add_scatters(sb)
    p, w = inferrer.get_prediction()
    assert p.type() == "torch.FloatTensor"
    assert p.shape == torch.Size([])
    assert w.shape == torch.Size([])
