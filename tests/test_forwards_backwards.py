from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F
from pytest_lazyfixture import lazy_fixture
from torch import Tensor, nn

from tomopt.core import X0
from tomopt.inference import PanelX0Inferrer, ScatterBatch
from tomopt.muon import MuonBatch, MuonGenerator2016
from tomopt.optimisation import MuonResampler, VoxelX0Loss
from tomopt.volume import (
    DetectorHeatMap,
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
INIT_RES = 1e4
INIT_EFF = 0.5
N_PANELS = 4


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["aluminium"]
    if z >= 0.5:
        rad_length[3:7, 3:7] = X0["lead"]
    return rad_length


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def get_panel_layers() -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[1.0, 1.0]) for i in range(N_PANELS)
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
                DetectorPanel(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[1.0, 1.0])
                for i in range(N_PANELS)
            ],
        )
    )

    return nn.ModuleList(layers)


def get_sigmoid_panel_layers(smooth=1.0, init_res: float = INIT_RES, init_eff: float = INIT_EFF, n_panels: int = 4, init_xy_span=[3.0, 3.0]) -> nn.ModuleList:
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


def get_heatmap_layers() -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorHeatMap(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[-0.5, 0.5])
                for i in range(N_PANELS)
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
                DetectorHeatMap(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[-0.5, 0.5])
                for i in range(N_PANELS)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def panel_inferrer() -> PanelX0Inferrer:
    volume = Volume(get_panel_layers())
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inf = PanelX0Inferrer(volume=volume)
    inf.add_scatters(sb)
    return inf


@pytest.fixture
def fixed_budget_panel_inferrer() -> PanelX0Inferrer:
    volume = Volume(get_panel_layers(), budget=32)
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inf = PanelX0Inferrer(volume=volume)
    inf.add_scatters(sb)
    return inf


@pytest.fixture
def heatmap_inferrer() -> PanelX0Inferrer:
    volume = Volume(get_heatmap_layers())
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inf = PanelX0Inferrer(volume=volume)
    inf.add_scatters(sb)
    return inf


@pytest.fixture
def sigmoid_panel_inferrer() -> PanelX0Inferrer:
    volume = Volume(get_sigmoid_panel_layers())
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inf = PanelX0Inferrer(volume=volume)
    inf.add_scatters(sb)
    return inf


@pytest.fixture
def fixed_budget_sigmoid_panel_inferrer() -> PanelX0Inferrer:
    volume = Volume(get_sigmoid_panel_layers(), budget=8)
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    mu = MuonBatch(mus, init_z=volume.h)
    volume(mu)
    sb = ScatterBatch(mu=mu, volume=volume)
    inf = PanelX0Inferrer(volume=volume)
    inf.add_scatters(sb)
    return inf


# @pytest.fixture
# def deep_inferrer() -> DeepVolumeInferrer:
#     volume = Volume(get_panel_layers())
#     gen = MuonGenerator2016.from_volume(volume)
#     mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
#     mu = MuonBatch(mus, init_z=volume.h)
#     volume._target = Tensor([1])
#     volume(mu)
#     sb = ScatterBatch(mu=mu, volume=volume)
#     grp_feats = ["pred_x0", "delta_angles", "theta_msc", "voxels"]
#     n_infeats = 4

#     class MockModel(nn.Module):
#         def __init__(self) -> None:
#             super().__init__()
#             self.layer = nn.Linear(600 * (n_infeats + 3), 1)
#             self.act = nn.Sigmoid()

#         def forward(self, x: Tensor) -> Tensor:
#             return self.act(self.layer(x.mean(2).flatten()[None]))

#     inf = DeepVolumeInferrer(model=MockModel(), base_inferrer=PanelX0Inferrer(volume=volume), volume=volume, grp_feats=grp_feats)
#     inf.add_scatters(sb)
#     return inf


@pytest.mark.parametrize(
    "mode, inferrer",
    [
        ["gauss panel", lazy_fixture("panel_inferrer")],
        ["fixed-budget gauss panel", lazy_fixture("fixed_budget_panel_inferrer")],
        ["sigmoid panel", lazy_fixture("sigmoid_panel_inferrer")],
        ["fixed-budget sigmoid panel", lazy_fixture("fixed_budget_sigmoid_panel_inferrer")],
        ["heatmap", lazy_fixture("heatmap_inferrer")],
    ],
)
def test_forwards_panel(mode, inferrer):
    pred = inferrer.get_prediction()
    loss_func = VoxelX0Loss(target_budget=None, cost_coef=None)
    loss_val = loss_func(pred, inferrer.volume)

    if "fixed-budget" in mode:
        assert torch.autograd.grad(loss_val, inferrer.volume.budget_weights, retain_graph=True, allow_unused=False)[0].abs().sum() > 0

    if "panel" in mode:
        for l in inferrer.volume.get_detectors():
            for p in l.panels:
                assert torch.autograd.grad(loss_val, p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
                assert torch.autograd.grad(loss_val, p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
                assert torch.autograd.grad(loss_val, p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
    elif "heatmap" in mode:
        for l in inferrer.volume.get_detectors():
            for p in l.panels:
                assert torch.autograd.grad(loss_val, p.mu, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
                assert torch.autograd.grad(loss_val, p.sig, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
                assert torch.autograd.grad(loss_val, p.norm, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
                assert torch.autograd.grad(loss_val, p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


# def test_forwards_deep_panel(deep_inferrer):
#     pred = deep_inferrer.get_prediction()
#     loss_func = VolumeClassLoss(target_budget=1, cost_coef=1e-5, x02id={1: 1})
#     loss_val = loss_func(pred, deep_inferrer.volume)

#     for l in deep_inferrer.volume.get_detectors():
#         for p in l.panels:
#             assert torch.autograd.grad(loss_val, p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(loss_val, p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
#             assert torch.autograd.grad(loss_val, p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


@pytest.mark.parametrize(
    "mode, inferrer",
    [
        ["gauss panel", lazy_fixture("panel_inferrer")],
        ["fixed-budget gauss panel", lazy_fixture("fixed_budget_panel_inferrer")],
        ["sigmoid panel", lazy_fixture("sigmoid_panel_inferrer")],
        ["fixed-budget sigmoid panel", lazy_fixture("fixed_budget_sigmoid_panel_inferrer")],
    ],
)
def test_backwards_panel(mode, inferrer):
    pred = inferrer.get_prediction()
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=0.15)
    loss_val = loss_func(pred, inferrer.volume)
    opt = torch.optim.SGD(inferrer.volume.parameters(), lr=1)
    opt.zero_grad()
    loss_val.backward()
    for p in inferrer.volume.parameters():
        assert p.grad is not None
    opt.step()

    if "fixed_budget" in mode:
        assert (inferrer.volume.budget_weights != torch.zeros(2 * N_PANELS)).all()
    for l in inferrer.volume.get_detectors():
        for i, p in enumerate(l.panels):
            assert (p.xy != Tensor([0.5, 0.5])).all()
            if l.pos == "above":
                assert (p.z != Tensor([1 - (i * (2 * SZ) / N_PANELS)])).all()
            else:
                assert (p.z != Tensor([0.2 - (i * (2 * SZ) / N_PANELS)])).all()
            assert (p.xy_span != Tensor([0.5, 0.5])).all()
            assert p.resolution == Tensor([INIT_RES])
            assert p.efficiency == Tensor([INIT_EFF])


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_backwards_heatmap(heatmap_inferrer):
    pred = heatmap_inferrer.get_prediction()
    init_params = defaultdict(lambda: defaultdict(dict))
    for i, l in enumerate(heatmap_inferrer.volume.get_detectors()):
        for j, p in enumerate(l.panels):
            init_params[i][j]["mu"] = p.mu.detach().clone()
            init_params[i][j]["sig"] = p.mu.detach().clone()

    loss_func = VoxelX0Loss(target_budget=1, cost_coef=0.15)
    loss_val = loss_func(pred, heatmap_inferrer.volume)
    opt = torch.optim.SGD(heatmap_inferrer.volume.parameters(), lr=1)
    opt.zero_grad()
    loss_val.backward()
    for p in heatmap_inferrer.volume.parameters():
        assert p.grad is not None
    opt.step()
    for i, l in enumerate(heatmap_inferrer.volume.get_detectors()):
        for j, p in enumerate(l.panels):
            if l.pos == "above":
                assert (p.z != Tensor([1 - (i * (2 * SZ) / N_PANELS)])).all()
            else:
                assert (p.z != Tensor([0.2 - (i * (2 * SZ) / N_PANELS)])).all()
            assert p.resolution == Tensor([INIT_RES])
            assert p.efficiency == Tensor([INIT_EFF])
            assert (init_params[i][j]["mu"] != p.mu).sum() > len(p.mu) / 2
            assert (init_params[i][j]["sig"] != p.sig).sum() > len(p.sig) / 2
            assert p.norm != Tensor([1])


# def test_backwards_deep_panel(deep_inferrer):
#     pred = deep_inferrer.get_prediction()
#     loss_func = VolumeClassLoss(target_budget=1, cost_coef=0.15, x02id={1: 1})
#     loss_val = loss_func(pred, deep_inferrer.volume)
#     opt = torch.optim.SGD(deep_inferrer.volume.parameters(), lr=1)
#     opt.zero_grad()
#     loss_val.backward()
#     for p in deep_inferrer.volume.parameters():
#         assert p.grad is not None
#     opt.step()
#     for l in deep_inferrer.volume.get_detectors():
#         for i, p in enumerate(l.panels):
#             assert (p.xy != Tensor([0.5, 0.5])).all()
#             if l.pos == "above":
#                 assert (p.z != Tensor([1 - (i * (2 * SZ) / N_PANELS)])).all()
#             else:
#                 assert (p.z != Tensor([0.2 - (i * (2 * SZ) / N_PANELS)])).all()
#             assert (p.xy_span != Tensor([0.5, 0.5])).all()
#             assert p.resolution == Tensor([INIT_RES])
#             assert p.efficiency == Tensor([INIT_EFF])
