import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.volume import PassiveLayer, PanelDetectorLayer, DetectorHeatMap, Volume
from tomopt.optimisation import MuonResampler
from tomopt.muon import MuonBatch, MuonGenerator2016
from tomopt.core import X0
from tomopt.utils import jacobian


LW = Tensor([1, 1])
SZ = 0.1
N = 1000
Z = 1


@pytest.fixture
def batch():
    mg = MuonGenerator2016(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))
    return MuonBatch(mg(N), init_z=1)


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["lead"]
    if z < 0.5:
        rad_length[...] = X0["beryllium"]
    return rad_length


def area_cost(a: Tensor) -> Tensor:
    return F.relu(a)


def test_heatmap_detector_layer(batch):
    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorHeatMap(
                init_xyz=[0.5, 0.5, 0.9],
                init_xy_span=[-1.0, 1.0],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            )
        ],
    )
    assert dl.type_label == "heatmap"

    for p in dl.panels:
        assert p.resolution == Tensor([1e3])
        assert p.efficiency == Tensor([1])
        assert p.n_cluster == 30

    start = batch.copy()
    dl(batch)
    assert torch.abs(batch.z - Tensor([Z - (2 * SZ)])) < 1e-5
    assert torch.all(batch.dtheta(start.theta) == 0)  # Detector layers don't scatter
    assert torch.all(batch.xy != start.xy)

    hits = batch.get_hits()
    assert len(hits) == 1
    assert hits["above"]["reco_xy"].shape == torch.Size([len(batch), 1, 2])
    assert hits["above"]["gen_xy"].shape == torch.Size([len(batch), 1, 2])

    # every reco hit (x,y) is function of GMM mean and sigma
    for v in [dl.panels[0].mu, dl.panels[0].sig]:
        grad = jacobian(hits["above"]["reco_xy"][:, 0], v).sum((-1))
        assert not grad.isnan().any()
        assert (grad != 0).sum() > 0

    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorHeatMap(
                init_xyz=[0.5, 0.5, 0.9],
                init_xy_span=[-0.5, 0.5],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
            DetectorHeatMap(
                init_xyz=[3.0, 0.5, 2.0],
                init_xy_span=[-1.0, 1.0],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
            DetectorHeatMap(
                init_xyz=[0.5, -0.5, -0.3],
                init_xy_span=[-1.0, 1.0],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
            DetectorHeatMap(
                init_xyz=[0.5, 0.5, 0.4],
                init_xy_span=[0.0, 0.5],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
        ],
    )

    # detector conform
    # TODO: Does the test still make sense for HeatMap?
    dl.conform_detector()
    for p in dl.panels:
        xy_low = p.xy_fix[0] - p.range_mult * p.delta_xy
        xy_high = p.xy_fix[1] + p.range_mult * p.delta_xy
        xy_low = torch.max(torch.tensor(0.0), xy_low)
        xy_high = torch.min(LW[0], xy_high)
        for mu_comp in p.mu:
            assert mu_comp[0] <= xy_high
            assert mu_comp[1] <= xy_high
            assert mu_comp[0] >= xy_low
            assert mu_comp[1] >= xy_low
        for sig_comp in p.sig:
            assert sig_comp[0] <= LW[1] * p.range_mult
            assert sig_comp[1] <= LW[1] * p.range_mult
            assert sig_comp[0] >= 0.0
            assert sig_comp[1] >= 0.0
        assert p.z[0] <= 1
        assert p.z[0] >= 0.8
        assert p.norm >= 0.01
        assert p.norm <= 1.5

    # cost
    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorHeatMap(
                init_xyz=[0.5, 0.5, 0.9],
                init_xy_span=[0.1, 0.2],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
            DetectorHeatMap(
                init_xyz=[3.0, 0.5, 2.0],
                init_xy_span=[0.3, 0.4],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
            DetectorHeatMap(
                init_xyz=[0.5, -0.5, -0.3],
                init_xy_span=[0.5, 0.6],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
            DetectorHeatMap(
                init_xyz=[0.5, 0.5, 0.4],
                init_xy_span=[0.7, 0.8],
                area_cost_func=area_cost,
                res=1e3,
                eff=1.0,
                n_cluster=30,
            ),
        ],
    )
    assert dl.get_cost().detach().cpu().numpy() == np.sum([p.get_cost().detach().cpu().numpy() for p in dl.panels])


def get_panel_layers(init_res: float = 1e4, init_eff: float = 0.5, n_panels: int = 4) -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorHeatMap(
                    init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)],
                    init_xy_span=[-0.5, 0.5],
                    area_cost_func=area_cost,
                    res=init_res,
                    eff=init_eff,
                    n_cluster=30,
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
                    init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)],
                    init_xy_span=[-0.5, 0.5],
                    area_cost_func=area_cost,
                    res=init_res,
                    eff=init_eff,
                    n_cluster=30,
                )
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.mark.flaky(max_runs=3, min_passes=2)
def test_volume_forward_panel():
    layers = get_panel_layers(n_panels=4)
    volume = Volume(layers=layers)
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(100), volume=volume, gen=gen)
    batch = MuonBatch(mus, init_z=volume.h)
    start = batch.copy()
    volume(batch)

    assert torch.abs(batch.z) <= 1e-5  # Muons traverse whole volume
    mask = batch.get_xy_mask((0, 0), LW)
    assert torch.all(batch.dtheta(start.theta)[mask] > 0)  # All masked muons scatter

    hits = batch.get_hits()
    assert "above" in hits and "below" in hits
    assert hits["above"]["reco_xy"].shape[1] == 4
    assert hits["below"]["reco_xy"].shape[1] == 4
    assert hits["above"]["gen_xy"].shape[1] == 4
    assert hits["below"]["gen_xy"].shape[1] == 4

    # every reco hit (x,y) is function of gmm mu and sig
    for i, l in enumerate(volume.get_detectors()):
        for j, p in enumerate(l.yield_zordered_panels()):
            for v in [p.mu, p.sig]:
                grad = jacobian(hits["above" if l.z > 0.5 else "below"]["reco_xy"][:, j], v).nansum((-1))
                assert grad.isnan().sum() == 0
                assert (grad != 0).sum() > 0


def test_detector_panel_methods():
    panel = DetectorHeatMap(
        init_xyz=[0.0, 0.01, 0.9],
        init_xy_span=[-0.25, 0.25],
        area_cost_func=area_cost,
        res=10.0,
        eff=0.5,
        n_cluster=30,
    )

    with torch.no_grad():
        panel.mu[:, 0] = 0.0
        panel.mu[:, 1] = 0.01
        panel.sig[:, 0] = 0.5
        panel.sig[:, 1] = 0.51

    # get_xy_mask

    with pytest.raises(NotImplementedError):
        panel.get_xy_mask(Tensor([[0, 0], [0.1, 0.1], [0.25, 0.25], [0.5, 0.5], [1, 1], [0.1, 1], [1, 0.1], [-1, -1]]))
    # TODO: wait until realistic validation
    # mask = panel.get_xy_mask(Tensor([[0, 0], [0.1, 0.1], [0.25, 0.25], [0.5, 0.5], [1, 1], [0.1, 1], [1, 0.1], [-1, -1]]))
    # assert (mask.int() == Tensor([1, 1, 0, 0, 0, 0, 0, 0])).all()

    # get_resolution
    res = panel.get_resolution(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert res[0].mean() == 10
    assert res[1].mean() < res[0].mean()
    assert res[2].mean() > 0

    # get_efficiency
    eff = panel.get_efficiency(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert eff[0] == 0.5
    assert eff[1] < eff[0]
    assert eff[2] > 0
    assert 0 < eff[3] < 0.5

    panel.realistic_validation = True
    res = panel.get_resolution(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert res[0].mean() == 10
    assert res[1].mean() < res[0].mean()
    assert res[2].mean() > 0

    eff = panel.get_efficiency(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert eff[0] == 0.5
    assert eff[1] < eff[0]
    assert eff[2] > 0
    assert 0 < eff[3] < 0.5

    # TODO: wait until realistic validation
    # panel.eval()
    # res = panel.get_resolution(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    # assert res[0].mean() == 10
    # assert res[1].mean() == 10
    # assert res[2].mean() == 0

    # eff = panel.get_efficiency(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    # assert eff[0] == 0.5
    # assert eff[1] == 0.5
    # assert eff[2] == 0
    # assert eff[3] == 0.5

    panel.realistic_validation = False
    panel.train()

    # get_hits
    panel = DetectorHeatMap(
        init_xyz=[0.5, 0.5, 0.9],
        init_xy_span=[-0.25, 0.25],
        area_cost_func=area_cost,
        res=10.0,
        eff=0.5,
        n_cluster=30,
    )
    mg = MuonGenerator2016(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))
    mu = MuonBatch(mg(100), 1)
    mu._xy = torch.ones_like(mu.xy) / 2
    hits = panel.get_hits(mu)
    assert (hits["gen_xy"] == mu.xy).all()
    assert (hits["z"].mean() - 0.9).abs() < 1e-5
    assert (hits["reco_xy"].mean(0) - Tensor([0.5, 0.5]) < 0.25).all()

    panel.realistic_validation = True
    mu._xy = torch.zeros_like(mu.xy)
    hits = panel.get_hits(mu)
    assert hits["reco_xy"].isinf().sum() == 0

    # TODO: wait until realistic validation
    # panel.eval()
    # hits = panel.get_hits(mu)
    # assert hits["reco_xy"].isinf().sum() == 2 * len(mu)
    # mu = MuonBatch(mg(100), 1)
    # hits = panel.get_hits(mu)
    # mask = hits["reco_xy"].isinf().prod(1) - 1 < 0
    # # Reco hits can't leave panel
    # assert (Tensor([0.25, 0.25]) <= hits["reco_xy"][mask]).all()
    # assert (hits["reco_xy"][mask] <= Tensor([0.75, 0.75])).all()

    # get_cost
    cost = panel.get_cost()
    assert (torch.autograd.grad(cost, panel.sig, retain_graph=True, allow_unused=True)[0] > 0).all()

    panel = DetectorHeatMap(
        init_xyz=[2.0, -2.0, 2.0],
        init_xy_span=[-10.0, 10.0],
        area_cost_func=area_cost,
        res=10.0,
        eff=0.5,
        n_cluster=30,
    )
    panel.clamp_params((0, 0, 0), (1, 1, 1))
    assert panel.z - 1 < 0
    assert (panel.z - 1).abs() < 5e-3
