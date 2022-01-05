import pytest
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.volume import Volume, PassiveLayer, VoxelDetectorLayer, PanelDetectorLayer, DetectorPanel
from tomopt.muon import MuonBatch, generate_batch
from tomopt.inference import VoxelScatterBatch, VoxelX0Inferer, PanelScatterBatch, PanelX0Inferer, DeepVolumeInferer
from tomopt.optimisation.loss import VoxelX0Loss, VolumeClassLoss

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


def area_cost(x: Tensor) -> Tensor:
    return F.relu(x) ** 2


def get_voxel_layers() -> nn.ModuleList:
    layers = []

    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(
                VoxelDetectorLayer(pos=pos, init_eff=INIT_EFF, init_res=INIT_RES, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
            )
        else:
            pos = "below"
            layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


def get_panel_layers() -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[0.5, 0.5], area_cost_func=area_cost)
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
                DetectorPanel(
                    res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[0.5, 0.5], area_cost_func=area_cost
                )
                for i in range(N_PANELS)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def voxel_inferer() -> VoxelX0Inferer:
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_voxel_layers())
    volume(mu)
    sb = VoxelScatterBatch(mu=mu, volume=volume)
    inf = VoxelX0Inferer(volume=volume)
    inf.add_scatters(sb)
    return inf


@pytest.fixture
def panel_inferer() -> PanelX0Inferer:
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_panel_layers())
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)
    inf = PanelX0Inferer(volume=volume)
    inf.add_scatters(sb)
    return inf


@pytest.fixture
def deep_inferer() -> DeepVolumeInferer:
    mu = MuonBatch(generate_batch(N), init_z=1)
    volume = Volume(get_panel_layers())
    volume._target = Tensor([1])
    volume(mu)
    sb = PanelScatterBatch(mu=mu, volume=volume)

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(600 * 9, 1)
            self.act = nn.Sigmoid()

        def forward(self, x: Tensor) -> Tensor:
            return self.act(self.layer(x.mean(2).flatten()[None]))

    inf = DeepVolumeInferer(model=MockModel(), base_inferer=PanelX0Inferer(volume=volume), volume=volume)
    inf.add_scatters(sb)
    return inf


def test_forwards_voxel(voxel_inferer):
    pred, weight = voxel_inferer.get_prediction()
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=1e-5)
    loss_val = loss_func(pred, weight, voxel_inferer.volume)

    for l in voxel_inferer.volume.get_detectors():
        assert torch.nan_to_num(torch.autograd.grad(loss_val, l.resolution, retain_graph=True, allow_unused=True)[0].abs(), 0).sum() > 0
        assert torch.autograd.grad(loss_val, l.efficiency, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


def test_forwards_panel(panel_inferer):
    pred, weight = panel_inferer.get_prediction()
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=1e-5)
    loss_val = loss_func(pred, weight, panel_inferer.volume)

    for l in panel_inferer.volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(loss_val, p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(loss_val, p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(loss_val, p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


def test_forwards_deep_panel(deep_inferer):
    pred, weight = deep_inferer.get_prediction()
    loss_func = VolumeClassLoss(target_budget=1, cost_coef=1e-5, x02id={1: 1})
    loss_val = loss_func(pred, weight, deep_inferer.volume)

    for l in deep_inferer.volume.get_detectors():
        for p in l.panels:
            assert torch.autograd.grad(loss_val, p.xy, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(loss_val, p.z, retain_graph=True, allow_unused=True)[0].abs().sum() > 0
            assert torch.autograd.grad(loss_val, p.xy_span, retain_graph=True, allow_unused=True)[0].abs().sum() > 0


def test_backwards_voxel(voxel_inferer):
    pred, weight = voxel_inferer.get_prediction()
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=0.15)
    loss_val = loss_func(pred, weight, voxel_inferer.volume)
    opt = torch.optim.SGD(voxel_inferer.volume.parameters(), lr=1)
    opt.zero_grad()
    loss_val.backward()
    for p in voxel_inferer.volume.parameters():
        assert p.grad is not None
    opt.step()
    for l in voxel_inferer.volume.get_detectors():
        assert l.resolution.mean() != Tensor([INIT_RES])
        assert l.efficiency.mean() != Tensor([INIT_EFF])


def test_backwards_panel(panel_inferer):
    pred, weight = panel_inferer.get_prediction()
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=0.15)
    loss_val = loss_func(pred, weight, panel_inferer.volume)
    opt = torch.optim.SGD(panel_inferer.volume.parameters(), lr=1)
    opt.zero_grad()
    loss_val.backward()
    for p in panel_inferer.volume.parameters():
        assert p.grad is not None
    opt.step()
    for l in panel_inferer.volume.get_detectors():
        for i, p in enumerate(l.panels):
            assert (p.xy != Tensor([0.5, 0.5])).all()
            if l.pos == "above":
                assert (p.z != Tensor([1 - (i * (2 * SZ) / N_PANELS)])).all()
            else:
                assert (p.z != Tensor([0.2 - (i * (2 * SZ) / N_PANELS)])).all()
            assert (p.xy_span != Tensor([0.5, 0.5])).all()
            assert p.resolution == Tensor([INIT_RES])
            assert p.efficiency == Tensor([INIT_EFF])


def test_backwards_deep_panel(deep_inferer):
    pred, weight = deep_inferer.get_prediction()
    loss_func = VolumeClassLoss(target_budget=1, cost_coef=0.15, x02id={1: 1})
    loss_val = loss_func(pred, weight, deep_inferer.volume)
    opt = torch.optim.SGD(deep_inferer.volume.parameters(), lr=1)
    opt.zero_grad()
    loss_val.backward()
    for p in deep_inferer.volume.parameters():
        assert p.grad is not None
    opt.step()
    for l in deep_inferer.volume.get_detectors():
        for i, p in enumerate(l.panels):
            assert (p.xy != Tensor([0.5, 0.5])).all()
            if l.pos == "above":
                assert (p.z != Tensor([1 - (i * (2 * SZ) / N_PANELS)])).all()
            else:
                assert (p.z != Tensor([0.2 - (i * (2 * SZ) / N_PANELS)])).all()
            assert (p.xy_span != Tensor([0.5, 0.5])).all()
            assert p.resolution == Tensor([INIT_RES])
            assert p.efficiency == Tensor([INIT_EFF])
