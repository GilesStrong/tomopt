from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from pytest_mock import mocker  # noqa F401
from torch import Tensor, nn

from tomopt.core import props
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
N = 1000
Z = 1


def get_panel_layers(init_res: float = 1e4, init_eff: float = 0.5, n_panels: int = 4) -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0]) for i in range(n_panels)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(properties_func=arb_properties, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0])
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


def get_sigmoid_panel_layers(smooth=Tensor([1.0]), init_res: float = 1e4, init_eff: float = 0.5, n_panels: int = 4) -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                SigmoidDetectorPanel(smooth=smooth, res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0])
                for i in range(n_panels)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(properties_func=arb_properties, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                SigmoidDetectorPanel(smooth=smooth, res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0])
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def batch():
    mg = MuonGenerator2016(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))
    return MuonBatch(mg(N), init_z=1)


def arb_properties(*, z: float, lw: Tensor, size: float) -> Tensor:
    prop = lw.new_empty((6, int(lw[0].item() / size), int(lw[1].item() / size)))
    for i, p in enumerate(props):
        prop[i] = torch.ones(list((lw / size).long())) * p["lead"]
        if z < 0.5:
            prop[i] = p["beryllium"]
    return prop


def test_layer(batch):
    l = PassiveLayer(lw=LW, z=1, size=SZ)
    batch._x = 0.5
    batch._y = 0.7
    assert torch.all(l.mu_abs2idx(batch)[0] == Tensor([5, 7]))


def test_passive_layer_methods():
    pl = PassiveLayer(lw=LW, z=Z, size=SZ)
    assert pl.properties is None

    pl.load_properties(arb_properties)
    assert torch.all(pl.properties == arb_properties(z=Z, lw=LW, size=SZ))


def test_passive_layer_forwards(batch):
    # Normal scattering
    pl = PassiveLayer(properties_func=arb_properties, lw=LW, z=Z, size=SZ, step_sz=SZ / 10)
    start = batch.copy()
    pl(batch)
    assert (torch.abs(batch.z - Tensor([Z - SZ])) < 1e-5).all()
    assert torch.all(batch.dtheta(start.theta[batch._keep_mask]) > 0)
    assert torch.all(batch.xy != start.xy[batch._keep_mask])

    # X0 affects scattering
    pl = PassiveLayer(properties_func=arb_properties, lw=LW, z=0, size=SZ, step_sz=SZ / 10)
    batch2 = start.copy()
    pl(batch2)
    assert batch2.dtheta(start.theta[batch._keep_mask]).mean() < batch.dtheta(start.theta[batch._keep_mask]).mean()

    # Small scattering
    pl = PassiveLayer(properties_func=arb_properties, lw=LW, z=Z, size=1e-4, step_sz=SZ / 10)
    batch = start.copy()
    pl(batch)
    assert (torch.abs(batch.z - Tensor([Z - 1e-4])) <= 1e-3).all()
    assert (batch.dtheta(start.theta[batch._keep_mask]) < 1e-2).sum() / len(batch) > 0.9
    assert (torch.abs(batch.xy - start.xy[batch._keep_mask]) < 1e-3).sum() / len(batch) > 0.9


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_passive_layer_scatter_and_propagate(mocker):  # noqa: F811
    mg = MuonGenerator2016(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))

    prev_count = 0
    for n in (1, 2, 5):
        batch = MuonBatch(mg(N), init_z=Z)
        for m in ["propagate_d", "get_xy_mask", "scatter_dxyz", "scatter_dtheta_dphi"]:
            mocker.spy(batch, m)

        pl = PassiveLayer(properties_func=arb_properties, lw=LW, size=SZ, z=Z, scatter_model="pdg", step_sz=SZ / n)
        pl(batch)
        curr_count = batch.propagate_d.call_count
        assert curr_count > prev_count
        assert batch.scatter_dxyz.call_count == curr_count
        assert batch.get_xy_mask.call_count == curr_count
        prev_count = curr_count


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def test_panel_detector_layer(mocker, batch):  # noqa F811
    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[1.0, 1.0])],
    )
    assert dl.type_label == "panel"

    for p in dl.panels:
        assert p.resolution == Tensor([1e3])
        assert p.efficiency == Tensor([1])

    start = batch.copy()
    dl(batch)
    assert (torch.abs(batch.z - Tensor([Z - (2 * SZ)]))).all() < 1e-5
    assert torch.all(batch.dtheta(start.theta) == 0)  # Detector layers don't scatter
    assert torch.all(batch.xy != start.xy)

    hits = batch.get_hits()
    assert len(hits) == 1
    assert hits["above"]["reco_xyz"].shape == torch.Size([len(batch), 1, 3])
    assert hits["above"]["gen_xyz"].shape == torch.Size([len(batch), 1, 3])
    assert hits["above"]["unc_xyz"].shape == torch.Size([len(batch), 1, 3])
    assert hits["above"]["eff"].shape == torch.Size([len(batch), 1, 1])

    # every reco hit (x,y) is function of panel position and size
    for var in ["reco_xyz", "unc_xyz", "eff"]:
        for dep_var in [dl.panels[0].xy, dl.panels[0].xy_span]:
            grad = jacobian(hits["above"][var][:, 0], dep_var).sum((-1))
            assert not grad.isnan().any()
            assert (grad != 0).sum() > 0

    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.5, 5.0]),
            DetectorPanel(res=1e3, eff=1, init_xyz=[3.0, 0.5, 2.0], init_xy_span=[1.0, 1.0]),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, -0.5, -0.3], init_xy_span=[1.0, 1.0]),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.4], init_xy_span=[0.0, 0.5]),
        ],
    )

    # z-ordering
    zorder = dl.get_panel_zorder()
    assert (zorder == np.array([1, 0, 3, 2])).all()
    for i, (_, p) in enumerate(dl.yield_zordered_panels()):
        assert p is dl.panels[zorder[i]]

    # detector conform
    dl.conform_detector()
    for p in dl.panels:
        assert p.xy[0] <= 1
        assert p.xy[1] <= 1
        assert p.xy[0] >= 0
        assert p.xy[1] >= 0
        assert p.z[0] <= 1
        assert p.z[0] >= 0.8
        assert p.xy_span[0] <= 1
        assert p.xy_span[1] <= 10
        assert p.xy_span[0] >= 0
        assert p.xy_span[1] >= 0

    # cost
    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.1, 0.2]),
            DetectorPanel(res=1e3, eff=1, init_xyz=[3.0, 0.5, 2.0], init_xy_span=[0.3, 0.4]),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, -0.5, -0.3], init_xy_span=[0.5, 0.6]),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.4], init_xy_span=[0.7, 0.8]),
        ],
    )
    assert dl.get_cost().detach().cpu().numpy() == np.sum([p.xy_span.prod().detach().cpu().numpy() for p in dl.panels])

    # budget checks
    assert dl._n_costs == 4
    for p in dl.panels:
        mocker.patch.object(p, "assign_budget")
    dl.assign_budget(None)
    for p in dl.panels:
        assert p.assign_budget.call_count == 0
    dl.assign_budget(Tensor([1, 2, 3, 4]))
    for i, p in zip([2, 1, 4, 3], dl.panels):  # Panels are called in z order
        assert p.assign_budget.call_count == 1
        print(p.z)
        p.assign_budget.assert_called_with(Tensor([i]))


def test_volume_properties():
    layers = get_panel_layers()
    volume = Volume(layers=layers)

    assert volume.layers == layers
    with pytest.raises(AttributeError):
        volume._n_layer_costs == 8
    with pytest.raises(AttributeError):
        volume.budget_weights.shape == torch.Size([8])

    volume = Volume(layers=layers, budget=10)
    assert volume._n_layer_costs == [4, 4]  # 4 panels per layer
    assert volume.budget_weights.shape == torch.Size([8])
    assert volume.budget_weights.sum() == 0  # Equal budget split at start


def test_volume_draw(mocker):  # noqa F811
    layers = get_panel_layers()
    volume = Volume(layers=layers)
    mocker.patch("matplotlib.pyplot.show")
    mocker.spy(plt, "title")
    mocker.spy(plt, "figure")
    volume.draw(xlim=(-1, 2), ylim=(-1, 2), zlim=(0, 1.2))

    # Assert plt.title has been called with expected arg
    plt.title.assert_called_once_with("Volume layers")

    # Assert plt.figure got called
    assert plt.figure.called


def test_volume_methods(mocker):  # noqa F811
    layers = get_panel_layers()
    volume = Volume(layers=layers)
    assert volume.get_detectors()[-1] == layers[-1]
    assert volume.get_passives()[-1] == layers[-2]
    assert torch.all(volume.lw == LW)
    assert volume.passive_size == Tensor([SZ])
    assert volume.h == 10 * SZ
    with pytest.raises(AttributeError):
        volume.lw = 0
    with pytest.raises(AttributeError):
        volume.passive_size = 0
    with pytest.raises(AttributeError):
        volume.h = 0

    zr = volume.get_passive_z_range()
    assert torch.abs(zr[0] - 0.2) < 1e-5
    assert torch.abs(zr[1] - 0.8) < 1e-5
    assert volume.get_cost() == 8.0

    cube = volume.get_rad_cube()
    assert cube.shape == torch.Size([6] + list((LW / SZ).long()))
    assert torch.all(cube[0] == arb_properties(z=SZ, lw=LW, size=SZ)[0])  # cube reversed to match lookup_passive_xyz_coords: layer zero = bottom layer
    assert torch.all(cube[-1] == arb_properties(z=Z, lw=LW, size=SZ)[0])

    assert torch.all(volume.lookup_passive_xyz_coords(Tensor([0.55, 0.63, 0.31])) == Tensor([[5, 6, 1]]))
    assert torch.all(volume.lookup_passive_xyz_coords(Tensor([[0.55, 0.63, 0.31], [0.12, 0.86, 0.45]])) == Tensor([[5, 6, 1], [1, 8, 2]]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([0.55, 1.2, 0.31]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([0.55, 0.63, 13]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([-1, 0.63, 0.31]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([0.55, -1.2, 0.31]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([0.55, 0.63, -13]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([0.55, 0.63, 0]))
    with pytest.raises(ValueError):
        volume.lookup_passive_xyz_coords(Tensor([0.55, 0.63, 1]))

    def arb_properties2(*, z: float, lw: Tensor, size: float) -> Tensor:
        prop = lw.new_empty((6, int(lw[0].item() / size), int(lw[1].item() / size)))
        for i, p in enumerate(props):
            prop[i] = torch.ones(list((lw / size).long())) * p["aluminium"]
            if z < 0.5:
                prop[i] = p["lead"]
        return prop

    volume.load_properties(arb_properties2)
    cube = volume.get_rad_cube()
    assert cube.shape == torch.Size([6] + list((LW / SZ).long()))
    assert torch.all(cube[0] == arb_properties2(z=SZ, lw=LW, size=SZ)[0])  # cube reversed to match lookup_passive_xyz_coords: layer zero = bottom layer
    assert torch.all(cube[-1] == arb_properties2(z=Z, lw=LW, size=SZ)[0])

    edges = volume.edges
    assert edges.shape == torch.Size((600, 3))
    assert (edges[0] == Tensor([0, 0, 2 * SZ])).all()
    assert (edges[-1] == Tensor([LW[0] - SZ, LW[1] - SZ, 7 * SZ])).all()
    centres = volume.centres
    assert centres.shape == torch.Size((600, 3))
    assert (centres[0] == Tensor([0, 0, 2 * SZ]) + (SZ / 2)).all()
    assert (centres[-1] == Tensor([LW[0] - SZ, LW[1] - SZ, 7 * SZ]) + (SZ / 2)).all()

    # Budget assignment
    mocker.patch.object(layers[0], "assign_budget")
    mocker.patch.object(layers[-1], "assign_budget")
    volume = Volume(layers=layers, budget=8)
    for i in [0, -1]:
        assert layers[i].assign_budget.call_count == 1
        assert (layers[i].assign_budget.call_args.args[0] == Tensor([1, 1, 1, 1])).all()


@pytest.mark.flaky(max_runs=3, min_passes=2)
@pytest.mark.parametrize("panel", ["gauss", "sigmoid"])
def test_volume_forward_panel(panel):
    if panel == "gauss":
        layers = get_panel_layers(n_panels=4)
    elif panel == "sigmoid":
        layers = get_sigmoid_panel_layers(n_panels=4)
    else:
        raise ValueError(f"Panel model {panel} not recognised")

    volume = Volume(layers=layers)
    gen = MuonGenerator2016.from_volume(volume)
    mus = MuonResampler.resample(gen(N), volume=volume, gen=gen)
    batch = MuonBatch(mus, init_z=volume.h)
    start = batch.copy()
    volume(batch)

    assert (torch.abs(batch.z) <= 1e-5).all()  # Muons traverse whole volume
    mask = batch.get_xy_mask((0, 0), LW)
    assert torch.all(batch.dtheta(start.theta)[mask] > 0)  # All masked muons scatter

    hits = batch.get_hits()
    assert "above" in hits and "below" in hits
    assert hits["above"]["reco_xyz"].shape == torch.Size([N, 4, 3])
    assert hits["below"]["reco_xyz"].shape == torch.Size([N, 4, 3])
    assert hits["above"]["gen_xyz"].shape == torch.Size([N, 4, 3])
    assert hits["below"]["gen_xyz"].shape == torch.Size([N, 4, 3])
    assert hits["above"]["unc_xyz"].shape == torch.Size([N, 4, 3])
    assert hits["below"]["unc_xyz"].shape == torch.Size([N, 4, 3])
    assert hits["above"]["eff"].shape == torch.Size([N, 4, 1])
    assert hits["below"]["eff"].shape == torch.Size([N, 4, 1])

    # uncertainties
    for pos in hits:
        assert hits[pos]["unc_xyz"][:, :, :2].min().item() > 0
        assert hits[pos]["unc_xyz"][:, :, :2].std().item() > 0
        assert (hits[pos]["unc_xyz"][:, :, 2] == 0).all()

    # efficiencies
    for pos in hits:
        assert hits[pos]["eff"].min().item() > 0
        assert hits[pos]["eff"].max().item() < 1
        assert hits[pos]["eff"].std().item() > 0

    # every reco hit (x,y) is function of panel position and size
    for i, l in enumerate(volume.get_detectors()):
        for j, (_, p) in enumerate(l.yield_zordered_panels()):
            for dep_var in [p.xy, p.xy_span]:
                for var in ["reco_xyz", "unc_xyz", "eff"]:
                    grad = jacobian(hits["above" if l.z > 0.5 else "below"][var][:, j], dep_var).nansum((-1))
                    assert grad.isnan().sum() == 0
                    assert (grad != 0).sum() > 0


@pytest.mark.parametrize("model, partial_panel", [["gauss", DetectorPanel], ["sigmoid", partial(SigmoidDetectorPanel, smooth=Tensor([1.0]))]])
def test_detector_panel_properties(model, partial_panel):
    panel = partial_panel(res=1, eff=0.5, init_xyz=[0.5, 0.4, 0.9], init_xy_span=[0.3, 0.5], realistic_validation=False, m2_cost=4)
    assert panel.m2_cost == Tensor([4])
    assert panel.budget_scale == Tensor([1])
    assert panel.resolution == Tensor([1])
    assert panel.efficiency == Tensor([0.5])
    assert (panel.xy == Tensor([0.5, 0.4])).all()
    assert panel.z == Tensor([0.9])
    assert (panel.xy_span == Tensor([0.3, 0.5])).all()
    assert (panel.get_scaled_xy_span() == Tensor([0.3, 0.5])).all()
    assert panel.x == Tensor([0.5])
    assert panel.y == Tensor([0.4])
    if model == "sigmoid":
        assert (panel.smooth - 1.0).abs() < 1e-5

    panel = partial_panel(res=1, eff=0.5, init_xyz=[0.5, 0.4, 0.9], init_xy_span=[0.8, 0.5], realistic_validation=False, m2_cost=10, budget=64)
    assert panel.budget_scale == Tensor([4])
    assert (panel.xy_span == Tensor([0.8, 0.5])).all()
    assert (panel.get_scaled_xy_span() == Tensor([3.2, 2.0])).all()


@pytest.mark.parametrize("model, partial_panel", [["gauss", DetectorPanel], ["sigmoid", partial(SigmoidDetectorPanel, smooth=Tensor([1.0]))]])
def test_detector_panel_methods(model, partial_panel):
    panel = partial_panel(res=10, eff=0.5, init_xyz=[0.0, 0.01, 0.9], init_xy_span=[0.5, 0.51])

    # get_xy_mask
    mask = panel.get_xy_mask(Tensor([[0, 0], [0.1, 0.1], [0.25, 0.25], [0.5, 0.5], [1, 1], [0.1, 1], [1, 0.1], [-1, -1]]))
    assert (mask.int() == Tensor([1, 1, 0, 0, 0, 0, 0, 0])).all()

    if model == "gauss":
        # get_gauss
        with pytest.raises(ValueError):
            partial_panel(res=1, eff=0.5, init_xyz=[np.NaN, 0.0, 0.9], init_xy_span=[1.0, 1.0]).get_gauss()
        with pytest.raises(ValueError):
            partial_panel(res=1, eff=0.5, init_xyz=[0.0, 0.0, 0.9], init_xy_span=[0.5, np.NaN]).get_gauss()
        gauss = panel.get_gauss()
        assert (gauss.loc == Tensor([0.0, 0.01])).all()
        assert (gauss.scale == Tensor([0.5 / 4, 0.51 / 4])).all()
    elif model == "sigmoid":
        with pytest.raises(ValueError):
            SigmoidDetectorPanel(smooth=-1, res=1, eff=0.5, init_xyz=[0.0, 0.0, 0.9], init_xy_span=[1.0, 1.0])
        with pytest.raises(ValueError):
            SigmoidDetectorPanel(smooth=torch.nan, res=1, eff=0.5, init_xyz=[0.0, 0.0, 0.9], init_xy_span=[1.0, 1.0])

    # get_resolution
    res = panel.get_resolution(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert res[0].mean() == 10
    assert res[1].mean() < res[0].mean()
    assert res[2].mean() > 0
    assert res[3, 0] == 10

    panel.realistic_validation = True
    res = panel.get_resolution(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert res[0].mean() == 10
    assert res[1].mean() < res[0].mean()
    assert res[2].mean() > 0
    assert res[3, 0] == 10

    panel.eval()
    res = panel.get_resolution(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert res[0].mean() == 10
    assert res[1].mean() == 10
    assert res[2].mean() == 0
    assert res[3, 0] == 10
    panel.realistic_validation = False
    panel.train()

    # get_efficiency
    eff = panel.get_efficiency(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert eff[0] == 0.5
    assert eff[1] < eff[0]
    assert eff[2] > 0
    assert 0 < eff[3] < 0.5

    panel.realistic_validation = True
    eff = panel.get_efficiency(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert eff[0] == 0.5
    assert eff[1] < eff[0]
    assert eff[2] > 0
    assert 0 < eff[3] < 0.5

    panel.eval()
    eff = panel.get_efficiency(Tensor([[0, 0.01], [0.1, 0.1], [0.5, 0.5], [0, 0.1]]))
    assert eff[0] == 0.5
    assert eff[1] == 0.5
    assert eff[2] == 0
    assert eff[3] == 0.5
    panel.realistic_validation = False
    panel.train()

    # get_hits
    panel = partial_panel(res=10, eff=0.5, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.5, 0.5])
    mg = MuonGenerator2016(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))
    mu = MuonBatch(mg(100), 1)
    mu._xy = torch.ones_like(mu.xy) / 2
    hits = panel.get_hits(mu)
    assert (hits["gen_xyz"][:, :2] == mu.xy).all()
    assert (hits["gen_xyz"][:, 2].mean() - 0.9).abs() < 1e-5
    assert (hits["reco_xyz"][:, :2].mean(0) - Tensor([0.5, 0.5]) < 0.25).all()

    panel.realistic_validation = True
    mu._xy = torch.zeros_like(mu.xy)
    hits = panel.get_hits(mu)
    assert hits["reco_xyz"].isinf().sum() == 0

    panel.eval()
    hits = panel.get_hits(mu)
    assert hits["reco_xyz"].isinf().sum() == 2 * len(mu)
    mu = MuonBatch(mg(100), 1)
    hits = panel.get_hits(mu)
    mask = hits["reco_xyz"][:, :2].isinf().prod(1) - 1 < 0
    # Reco hits can't leave panel
    assert (Tensor([0.25, 0.25]) <= hits["reco_xyz"][mask][:, :2]).all()
    assert (hits["reco_xyz"][mask][:, :2] <= Tensor([0.75, 0.75])).all()

    # get_cost
    cost = panel.get_cost()
    assert cost == Tensor([0.5 * 0.5])
    assert (torch.autograd.grad(cost, panel.xy_span, retain_graph=True, allow_unused=True)[0] > 0).all()

    # clamp_params
    panel.clamp_params((0, 0, 0.8), (1, 1, 1))
    assert panel.z == Tensor([0.9])
    assert (panel.xy == Tensor([0.5, 0.5])).all()
    assert (panel.xy_span == Tensor([0.5, 0.5])).all()

    panel = partial_panel(res=10, eff=0.5, init_xyz=[2.0, -2.0, 2.0], init_xy_span=[0.0, 20.0])
    panel.clamp_params((0, 0, 0), (1, 1, 1))
    assert (panel.xy == Tensor([1, 0])).all()
    assert panel.z - 1 < 0
    assert (panel.z - 1).abs() < 5e-3
    assert (panel.xy_span == Tensor([5e-2, 10])).all()

    # Budget assignment
    panel = partial_panel(res=10, eff=0.5, init_xyz=[2.0, -2.0, 2.0], init_xy_span=[0.8, 0.5], m2_cost=10)
    panel.assign_budget(None)
    assert panel.budget_scale == 1
    panel.assign_budget(Tensor([64]))
    assert panel.budget_scale == Tensor([4])
    assert (panel.xy_span == Tensor([0.8, 0.5])).all()
    assert (panel.get_scaled_xy_span() == Tensor([3.2, 2.0])).all()
