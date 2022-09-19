import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np
from unittest.mock import patch
from typing import Tuple
import types
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.volume import PassiveLayer, Volume, PanelDetectorLayer, DetectorPanel
from tomopt.muon import MuonBatch, MuonGenerator2016
from tomopt.core import X0
from tomopt.inference import (
    PanelX0Inferer,
    PanelScatterBatch,
    GenScatterBatch,
    DenseBlockClassifierFromX0s,
)
from tomopt.inference.volume import AbsIntClassifierFromX0
from tomopt.volume.layer import Layer
from tomopt.optimisation import MuonResampler
from tomopt.utils import jacobian

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


@pytest.fixture
def panel_scatter_batch() -> Tuple[MuonBatch, Volume, PanelScatterBatch]:
    volume = Volume(get_panel_layers())
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)
    return mu, volume, sb


@pytest.mark.flaky(max_runs=3, min_passes=2)
@patch("matplotlib.pyplot.show")
def test_panel_scatter_batch(mock_show, panel_scatter_batch):
    mu, volume, sb = panel_scatter_batch
    assert len(sb) == len(mu)

    # hits
    hits = mu.get_hits()
    assert sb.hits["above"]["z"].shape == hits["above"]["z"].shape
    assert sb.n_hits_above == 4
    assert sb.n_hits_below == 4
    for i in range(4):
        assert (sb.above_hits[:, i, :2] == hits["above"]["reco_xy"][:, i]).all()
        assert (sb.below_hits[:, i, :2] == hits["below"]["reco_xy"][:, i]).all()
        assert (sb.above_gen_hits[:, i, :2] == hits["above"]["gen_xy"][:, i]).all()
        assert (sb.below_gen_hits[:, i, :2] == hits["below"]["gen_xy"][:, i]).all()

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
    _, panel = next(volume.get_detectors()[0].yield_zordered_panels())
    uncs = sb._get_hit_uncs([panel], sb.reco_hits[:, 0:1])
    assert (uncs[0][:, 2] == 0).all()
    xy_unc = uncs[0][:, :2]
    assert xy_unc.min().item() > 0
    assert xy_unc.max().item() < 0.1
    assert xy_unc.std().item() > 0

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
    sb = PanelScatterBatch(mu=mu, volume=volume)
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
    traj, start = PanelScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj == Tensor([[1, 1, -1]])).all()
    assert (start == xa0).all()
    # Different unc
    traj, _ = PanelScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1], dim=1), torch.stack([Tensor([[10, 10]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj == Tensor([[1, 1, -1]])).all()

    # 3 Hits inline
    xa2 = Tensor([[0.5, 0.5, 0.5]])
    # Same unc
    traj, _ = PanelScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj == Tensor([[1, 1, -1]])).all()
    # Different unc
    traj, _ = PanelScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[10, 10]]), Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj == Tensor([[1, 1, -1]])).all()

    # 3 Hits offline
    xa0 = Tensor([[0, 0, 1]])
    xa1 = Tensor([[0, 1, 0.5]])
    xa2 = Tensor([[1, 0, 0.5]])
    # Same unc
    traj, _ = PanelScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1, 1]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj - Tensor([[0.5, 0.5, -0.5]])).sum() < 1e-5
    # Different unc
    traj, _ = PanelScatterBatch.get_muon_trajectory(
        torch.stack([xa0, xa1, xa2], dim=1), torch.stack([Tensor([[1, 1]]), Tensor([[1e9, 1e9]]), Tensor([[1, 1]])], dim=1), lw=Tensor([1, 1])
    )
    assert (traj - Tensor([[1, 0, -0.5]])).sum() < 1e-5


def test_scatter_batch_compute(mocker, panel_scatter_batch):  # noqa F811
    mu, volume = panel_scatter_batch[0], panel_scatter_batch[1]
    hits = {
        "above": {
            "reco_xy": Tensor([[[0.0, 0.0], [0.1, 0.0]]]),
            "gen_xy": Tensor([[[0.0, 0.0], [0.1, 0.0]]]),
            "z": Tensor(
                [
                    [[1.0], [0.9]],
                ]
            ),
        },
        "below": {
            "reco_xy": Tensor(
                [
                    [[0.1, 0.0], [0.0, 0.0]],
                ]
            ),
            "gen_xy": Tensor(
                [
                    [[0.1, 0.0], [0.0, 0.0]],
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

    sb = PanelScatterBatch(mu=mu, volume=volume)
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
            "reco_xy": Tensor([[[10.0, -2.0], [1, 0.3]]]),
            "gen_xy": Tensor([[[0.0, 0.0], [0.1, 0.0]]]),
            "z": Tensor(
                [
                    [[1.0], [0.9]],
                ]
            ),
        },
        "below": {
            "reco_xy": Tensor(
                [
                    [[np.nan, 0.1], [10.0, -20.0]],
                ]
            ),
            "gen_xy": Tensor(
                [
                    [[0.1, 0.0], [0.0, 0.0]],
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


def test_abs_volume_inferer_properties(panel_scatter_batch):
    mu, volume, sb = panel_scatter_batch
    inferer = PanelX0Inferer(volume=volume)

    assert inferer.volume == volume
    assert torch.all(inferer.lw == LW)
    assert inferer.size == SZ


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_panel_x0_inferer_methods(mocker):  # noqa F811
    volume = Volume(get_panel_layers(init_res=1e5))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)
    inferer = PanelX0Inferer(volume=volume)

    pt = inferer.x0_from_scatters(deltaz=SZ, total_scatter=sb.total_scatter, theta_in=sb.theta_in, theta_out=sb.theta_out, mom=sb.mu.reco_mom[:, None])
    assert len(pt) == len(sb.poca_xyz)
    assert pt.shape == torch.Size([len(sb.mu), 1])

    mask = ~pt.isnan()
    for l in volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(pt[mask].abs().sum(), p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0

    eff = inferer.compute_efficiency(scatters=sb)
    assert eff[:, None].shape == pt.shape
    assert (eff > 0).all()

    # Single batch
    inferer.add_scatters(sb)
    assert len(inferer.scatter_batches) == 1
    for p in [
        inferer._n_mu,
        inferer._muon_scatter_vars,
        inferer._muon_scatter_var_uncs,
        inferer._muon_probs_per_voxel_zxy,
        inferer._muon_efficiency,
        inferer._vox_zxy_x0_preds,
        inferer._vox_zxy_x0_pred_uncs,
    ]:
        assert p is None

    inferer._combine_scatters()
    assert inferer.n_mu == len(sb)
    for p in [inferer._muon_scatter_vars, inferer._muon_scatter_var_uncs, inferer._muon_efficiency]:
        assert isinstance(p, Tensor) and len(p) == len(sb)
    for p in [inferer._muon_probs_per_voxel_zxy, inferer._vox_zxy_x0_preds, inferer._vox_zxy_x0_pred_uncs]:
        assert p is None
    inferer._reset_vars()
    for p in [
        inferer._n_mu,
        inferer._muon_scatter_vars,
        inferer._muon_scatter_var_uncs,
        inferer._muon_probs_per_voxel_zxy,
        inferer._muon_efficiency,
        inferer._vox_zxy_x0_preds,
        inferer._vox_zxy_x0_pred_uncs,
    ]:
        assert p is None

    mocker.spy(inferer, "_combine_scatters")
    mocker.spy(inferer, "get_voxel_zxy_x0_preds")
    inferer.vox_zxy_x0_preds
    assert inferer._combine_scatters.call_count == 1
    assert inferer.get_voxel_zxy_x0_preds.call_count == 1

    p1, w1 = inferer.get_prediction()
    assert inferer._combine_scatters.call_count == 1
    assert inferer.get_voxel_zxy_x0_preds.call_count == 1
    assert isinstance(inferer._muon_probs_per_voxel_zxy, Tensor)

    true = volume.get_rad_cube()
    assert p1.shape == true.shape
    assert w1.shape == torch.Size([])
    assert inferer._muon_probs_per_voxel_zxy.shape == torch.Size([len(sb)]) + true.shape
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
    sb2 = PanelScatterBatch(mu=mu, volume=volume)
    inferer.add_scatters(sb2)
    for p in [
        inferer._n_mu,
        inferer._muon_scatter_vars,
        inferer._muon_scatter_var_uncs,
        inferer._muon_probs_per_voxel_zxy,
        inferer._muon_efficiency,
        inferer._vox_zxy_x0_preds,
        inferer._vox_zxy_x0_pred_uncs,
    ]:
        assert p is None

    assert len(inferer.scatter_batches) == 2
    assert inferer.n_mu == len(sb) + len(sb2)
    assert inferer._combine_scatters.call_count == 2

    p2, w2 = inferer.get_prediction()  # Averaged prediction slightly changes with new batch
    assert (p2 - p1).abs().sum() > 1e-2
    assert (w2 - w1).abs().sum() > 1e-2
    assert inferer.get_voxel_zxy_x0_preds.call_count == 2


def test_panel_inferer_multi_batch():
    volume = Volume(get_panel_layers(init_res=1e5))
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(1000), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)

    # one batch
    inf = PanelX0Inferer(volume=volume)
    inf.add_scatters(PanelScatterBatch(mu=mu, volume=volume))
    pred1, weight1 = inf.get_prediction()

    # multi-batch
    inf = PanelX0Inferer(volume=volume)
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
        inf.add_scatters(PanelScatterBatch(mu=mu_batch, volume=volume))
    pred4, weight4 = inf.get_prediction()

    assert (((pred1 - pred4) / pred1).abs() < 1e-4).all()
    assert (((weight1 - weight4) / weight1).abs() < 1e-4).all()


def test_panel_x0_inferer_efficiency(mocker, panel_scatter_batch):  # noqa F811
    mu, volume, sb = panel_scatter_batch
    inferer = PanelX0Inferer(volume=volume)
    a_effs = Tensor([0.6, 0.2, 0.3, 0.4])
    b_effs = Tensor([0.5, 0.6, 0.7, 0.8])
    true_eff = 0.4263  # from MC sim

    def get_efficiency(self, xy):
        return self.eff.expand(len(xy)).clone()

    for i, d in enumerate(volume.get_detectors()):
        for j, p in enumerate(d.panels):
            p.eff = a_effs[j] if i == 0 else b_effs[j]
            p.get_efficiency = types.MethodType(get_efficiency, p)

    assert (inferer.compute_efficiency(scatters=sb)[0] - true_eff).abs() < 1e-3


def test_x0_inferer_scatter_inversion(mocker, panel_scatter_batch):  # noqa F811
    layer = Layer(LW, Z, SZ)
    mu, volume, sb = panel_scatter_batch
    inferer = PanelX0Inferer(volume=volume)
    inferer.size = SZ
    x0 = X0["lead"]
    mocker.patch("tomopt.volume.layer.torch.randn", lambda n, device: torch.ones(n, device=device))  # remove randomness
    scatters = layer._pdg_scatter(x0=x0, deltaz=SZ, theta=mu.theta, theta_x=mu.theta_x, theta_y=mu.theta_y, mom=mu.mom, log_term=False)
    dtheta, dphi = scatters["dtheta_vol"], scatters["dphi_vol"]

    mu_start = mu.copy()
    mu.scatter_dtheta_dphi(dtheta_vol=dtheta, dphi_vol=dphi)
    pred = inferer.x0_from_scatters(
        deltaz=SZ,
        total_scatter=torch.sqrt((dtheta**2) + (dphi**2))[:, None] / math.sqrt(2),
        theta_in=mu_start.theta[:, None],
        theta_out=mu.theta[:, None],
        mom=mu.mom[:, None],
    )

    assert (pred.mean() - x0).abs() < 1e-4


def test_x0_inferer_scatter_inversion_via_scatter_batch():
    gen = MuonGenerator2016((0, 1), (0, 1), fixed_mom=5)
    muons = MuonBatch(gen(10000), init_z=1)
    x0 = torch.ones(len(muons)) * X0["lead"]
    layer = PassiveLayer(lw=Tensor([1, 1]), z=1, size=0.1)
    pdg_scattering = layer._pdg_scatter(x0=x0[:, None], deltaz=0.01, theta_xy=muons.theta_xy, mom=muons.mom[:, None])

    muons.append_hits({"reco_xy": muons.xy.detach().clone(), "gen_xy": muons.xy.detach().clone(), "z": muons.z.expand_as(muons.x)[:, None]}, pos="above")
    muons.propagate(0.1)
    muons.append_hits({"reco_xy": muons.xy.detach().clone(), "gen_xy": muons.xy.detach().clone(), "z": muons.z.expand_as(muons.x)[:, None]}, pos="above")
    muons.propagate(0.01)
    muons.scatter_dtheta_dphi(dtheta_vol=pdg_scattering["dtheta_vol"], dphi_vol=pdg_scattering["dphi_vol"])
    muons.append_hits({"reco_xy": muons.xy.detach().clone(), "gen_xy": muons.xy.detach().clone(), "z": muons.z.expand_as(muons.x)[:, None]}, pos="below")
    muons.propagate(0.1)
    muons.append_hits({"reco_xy": muons.xy.detach().clone(), "gen_xy": muons.xy.detach().clone(), "z": muons.z.expand_as(muons.x)[:, None]}, pos="below")

    class MockVolume:
        lw = Tensor([10, 10])
        passive_size = 0.01
        device = torch.device("cpu")

        def get_passive_z_range(self):
            return (0.89, 0.9)

        def get_passives(self):
            return [1]

    volume = MockVolume()
    sb = GenScatterBatch(muons, volume=volume)
    inferer = PanelX0Inferer(volume=volume)

    print(sb.total_scatter.square().mean().sqrt() / math.sqrt(2))
    pred = inferer.x0_from_scatters(
        deltaz=0.01,
        total_scatter=sb.total_scatter.square().mean().sqrt() / math.sqrt(2),
        theta_in=sb.theta_in.square().mean().sqrt(),
        theta_out=sb.theta_out.square().mean().sqrt(),
        mom=sb.mu.mom[:, None].square().mean().sqrt(),
    )

    assert (pred - X0["lead"]).abs() < 1e-4


# @pytest.mark.flaky(max_runs=2, min_passes=1)
# @pytest.mark.parametrize("dvi_class, weighted", [[DeepVolumeInferer, False], [WeightedDeepVolumeInferer, True]])
# def test_deep_volume_inferer(dvi_class: Type[DeepVolumeInferer], weighted: bool):
#     volume = Volume(get_panel_layers(init_res=1e4))
#     gen = MuonGenerator2016.from_volume(volume)
#     mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
#     mu = MuonBatch(mus, init_z=volume.h)
#     volume(mu)
#     sb = PanelScatterBatch(mu=mu, volume=volume)
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

#     inferer = dvi_class(model=MockModel(), base_inferer=PanelX0Inferer(volume=volume), volume=volume, grp_feats=grp_feats, include_unc=True)

#     pt, pt_unc = inferer.get_base_predictions(scatters=sb)
#     assert len(pt) == len(sb.poca_xyz)
#     assert pt.shape == pt_unc.shape

#     assert len(inferer.in_vars) == 0
#     assert len(inferer.in_var_uncs) == 0
#     assert len(inferer.efficiencies) == 0
#     inferer.add_scatters(sb)
#     assert len(inferer.in_vars) == 1
#     assert len(inferer.in_var_uncs) == 1
#     assert len(inferer.efficiencies) == 1
#     assert inferer.in_vars[-1].shape == torch.Size((nvalid, n_infeats))
#     assert inferer.in_var_uncs[-1].shape == torch.Size((nvalid, n_infeats))
#     assert len(inferer.efficiencies[-1]) == nvalid

#     inputs = inferer._build_inputs(inferer.in_vars[0])
#     assert inputs.shape == torch.Size((600, nvalid, n_infeats + 4))  # +4 since voxels and dpoca_r

#     pred, weight = inferer.get_prediction()
#     assert pred.shape == torch.Size((1, 1))
#     assert weight.shape == torch.Size(())

#     for l in volume.get_detectors():
#         for panel in l.panels:
#             assert torch.autograd.grad(pred.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(pred.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(pred.abs().sum(), panel.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(weight.abs().sum(), panel.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(weight.abs().sum(), panel.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(weight.abs().sum(), panel.z, retain_graph=True, allow_unused=True)[0].abs().sum() == 0


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
    sb = PanelScatterBatch(mu=mu, volume=volume)
    inferer = DenseBlockClassifierFromX0s(12, PanelX0Inferer, volume=volume)
    inferer.add_scatters(sb)

    p, w = inferer.get_prediction()
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
    sb = PanelScatterBatch(mu=mu, volume=volume)

    class Inf(AbsIntClassifierFromX0):
        def x02probs(self, vox_preds: Tensor) -> Tensor:
            return F.softmax(vox_preds.mean([1, 2]), dim=-1)

    # Raw probs
    inferer = Inf(partial_x0_inferer=PanelX0Inferer, volume=volume, output_probs=True)
    inferer.add_scatters(sb)

    p, w = inferer.get_prediction()
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
    inferer = Inf(partial_x0_inferer=PanelX0Inferer, volume=volume, output_probs=False)
    inferer.add_scatters(sb)
    p, w = inferer.get_prediction()
    assert p.type() == "torch.LongTensor"
    assert p.shape == torch.Size([])
    assert w.shape == torch.Size([])

    # Single float prediction
    inferer = Inf(partial_x0_inferer=PanelX0Inferer, volume=volume, output_probs=False, class2float=lambda x, v: 3.5 * x)
    inferer.add_scatters(sb)
    p, w = inferer.get_prediction()
    assert p.type() == "torch.FloatTensor"
    assert p.shape == torch.Size([])
    assert w.shape == torch.Size([])
