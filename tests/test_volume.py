import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.volume.layer import Layer
from tomopt.volume import PassiveLayer, VoxelDetectorLayer, Volume, PanelDetectorLayer, DetectorPanel
from tomopt.muon import MuonBatch, MuonGenerator
from tomopt.core import X0
from tomopt.utils import jacobian


LW = Tensor([1, 1])
SZ = 0.1
N = 1000
Z = 1


@pytest.fixture
def batch():
    mg = MuonGenerator(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))
    return MuonBatch(mg(N), init_z=1)


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["lead"]
    if z < 0.5:
        rad_length[...] = X0["beryllium"]
    return rad_length


def test_layer(batch):
    l = Layer(lw=LW, z=1, size=SZ)
    batch.x = 0.5
    batch.y = 0.7
    assert torch.all(l.mu_abs2idx(batch)[0] == Tensor([5, 7]))


def test_passive_layer_methods():
    pl = PassiveLayer(lw=LW, z=Z, size=SZ)
    assert pl.rad_length is None

    pl.load_rad_length(arb_rad_length)
    assert torch.all(pl.rad_length == arb_rad_length(z=Z, lw=LW, size=SZ))


def test_passive_layer_forwards(batch):
    # Normal scattering
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=Z, size=SZ)
    start = batch.copy()
    pl(batch)
    assert torch.abs(batch.z - Tensor([Z - SZ])) < 1e-5
    assert torch.all(batch.dtheta(start) > 0)
    assert torch.all(batch.xy != start.xy)

    # X0 affects scattering
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=0, size=SZ)
    batch2 = start.copy()
    pl(batch2)
    assert batch2.dtheta(start).mean() < batch.dtheta(start).mean()

    # Small scattering
    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=Z, size=1e-4)
    batch = start.copy()
    pl(batch, 1)
    assert torch.abs(batch.z - Tensor([Z])) <= 1e-3
    assert torch.all(batch.dtheta(start) < 1e-2)
    assert torch.all(torch.abs(batch.xy - start.xy) < 1e-3)


@pytest.mark.parametrize("n", [(1), (2), (5)])
def test_passive_layer_scattering(mocker, batch, n):  # noqa: F811
    for m in ["propagate", "get_xy_mask"]:
        mocker.patch.object(MuonBatch, m)
    mock_getters = {}
    for m in ["theta_x", "theta_y", "x", "y", "mom"]:
        mock_getters[m] = mocker.PropertyMock(return_value=getattr(batch, m))
        mocker.patch.object(MuonBatch, m, mock_getters[m])

    pl = PassiveLayer(rad_length_func=arb_rad_length, lw=LW, size=SZ, z=Z)
    pl(batch, n)
    assert batch.propagate.call_count == n
    assert batch.propagate.called_with(SZ / n)
    assert batch.get_xy_mask.call_count == n
    assert mock_getters["mom"].call_count == n
    for m in ["x", "y"]:
        assert mock_getters[m].call_count == 2 * n
    for m in ["theta_x", "theta_y"]:
        assert mock_getters[m].call_count == 4 * n


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def test_voxel_detector_layer(batch):
    dl = VoxelDetectorLayer(pos="above", init_eff=1, init_res=1e3, lw=LW, z=Z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
    assert dl.resolution.mean() == Tensor([1e3])
    assert dl.efficiency.mean() == Tensor([1])

    start = batch.copy()
    dl(batch)
    assert torch.abs(batch.z - Tensor([Z - SZ])) < 1e-5
    assert torch.all(batch.dtheta(start) == 0)  # Detector layers don't scatter
    assert torch.all(batch.xy != start.xy)

    hits = batch.get_hits((0, 0), LW)
    assert len(hits) == 1
    assert hits["above"]["reco_xy"].shape == torch.Size([batch.get_xy_mask((0, 0), LW).sum(), 1, 2])
    assert hits["above"]["gen_xy"].shape == torch.Size([batch.get_xy_mask((0, 0), LW).sum(), 1, 2])
    assert torch.abs(hits["above"]["z"][0, 0, 0] - Z + (SZ / 2)) < 1e-5  # Hits located at z-centre of layer

    # every reco hit (x,y) is function of resolution
    grad = jacobian(hits["above"]["reco_xy"][:, 0], dl.resolution).sum((-1, -2))
    assert (grad == grad).sum() == 2 * len(grad)
    assert ((grad == grad) * (grad != 0)).sum() > 0

    # Conform detector
    dl = VoxelDetectorLayer(pos="above", init_eff=-1, init_res=1e14, lw=LW, z=Z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
    dl.conform_detector()
    assert (dl.resolution == 1e7).all()
    assert (dl.efficiency == 1e-7).all()
    dl = VoxelDetectorLayer(pos="above", init_eff=10.0, init_res=-10.0, lw=LW, z=Z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
    dl.conform_detector()
    assert (dl.resolution == 1).all()
    assert (dl.efficiency == 1).all()


def area_cost(a: Tensor) -> Tensor:
    return F.relu(a)


def test_panel_detector_layer(batch):
    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.5, 0.5], area_cost_func=area_cost)],
    )
    for p in dl.panels:
        assert p.resolution == Tensor([1e3])
        assert p.efficiency == Tensor([1])

    start = batch.copy()
    dl(batch)
    assert torch.abs(batch.z - Tensor([Z - (2 * SZ)])) < 1e-5
    assert torch.all(batch.dtheta(start) == 0)  # Detector layers don't scatter
    assert torch.all(batch.xy != start.xy)

    hits = batch.get_hits()
    assert len(hits) == 1
    assert hits["above"]["reco_xy"].shape == torch.Size([len(batch), 1, 2])
    assert hits["above"]["gen_xy"].shape == torch.Size([len(batch), 1, 2])

    # every reco hit (x,y) is function of panel position and size
    for v in [dl.panels[0].xy, dl.panels[0].xy_span]:
        grad = jacobian(hits["above"]["reco_xy"][:, 0], v).sum((-1))
        assert (grad == grad).sum() == 2 * len(grad)
        assert ((grad == grad) * (grad != 0)).sum() > 0

    dl = PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.5, 5.0], area_cost_func=area_cost),
            DetectorPanel(res=1e3, eff=1, init_xyz=[3.0, 0.5, 2.0], init_xy_span=[0.5, 0.5], area_cost_func=area_cost),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, -0.5, -0.3], init_xy_span=[0.5, 0.5], area_cost_func=area_cost),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.4], init_xy_span=[0.0, 0.5], area_cost_func=area_cost),
        ],
    )

    # z-ordering
    zorder = dl.get_panel_zorder()
    assert (zorder == np.array([1, 0, 3, 2])).all()
    for i, p in enumerate(dl.yield_zordered_panels()):
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
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.1, 0.2], area_cost_func=area_cost),
            DetectorPanel(res=1e3, eff=1, init_xyz=[3.0, 0.5, 2.0], init_xy_span=[0.3, 0.4], area_cost_func=area_cost),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, -0.5, -0.3], init_xy_span=[0.5, 0.6], area_cost_func=area_cost),
            DetectorPanel(res=1e3, eff=1, init_xyz=[0.5, 0.5, 0.4], init_xy_span=[0.7, 0.8], area_cost_func=area_cost),
        ],
    )
    assert dl.get_cost().detach().cpu().numpy() == np.sum([area_cost(p.xy_span.prod()).detach().cpu().numpy() for p in dl.panels])


def get_voxel_layers():
    layers = []
    init_eff = 0.5
    init_res = 1000
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


def test_volume_methods():
    layers = get_voxel_layers()
    volume = Volume(layers=layers)
    assert volume.get_detectors()[-1] == layers[-1]
    assert volume.get_passives()[-1] == layers[-3]
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
    assert volume.get_cost() == 41392.6758

    cube = volume.get_rad_cube()
    assert cube.shape == torch.Size([6] + list((LW / SZ).long()))
    assert torch.all(cube[0] == arb_rad_length(z=SZ, lw=LW, size=SZ))  # cube reversed to match lookup_passive_xyz_coords: layer zero = bottom layer
    assert torch.all(cube[-1] == arb_rad_length(z=Z, lw=LW, size=SZ))

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

    def arb_rad_length2(*, z: float, lw: Tensor, size: float) -> float:
        rad_length = torch.ones(list((lw / size).long())) * X0["aluminium"]
        if z < 0.5:
            rad_length[...] = X0["lead"]
        return rad_length

    volume.load_rad_length(arb_rad_length2)
    cube = volume.get_rad_cube()
    assert cube.shape == torch.Size([6] + list((LW / SZ).long()))
    assert torch.all(cube[0] == arb_rad_length2(z=SZ, lw=LW, size=SZ))  # cube reversed to match lookup_passive_xyz_coords: layer zero = bottom layer
    assert torch.all(cube[-1] == arb_rad_length2(z=Z, lw=LW, size=SZ))

    edges = volume.edges
    assert edges.shape == torch.Size((600, 3))
    assert (edges[0] == Tensor([0, 0, 2 * SZ])).all()
    assert (edges[-1] == Tensor([LW[0] - SZ, LW[1] - SZ, 7 * SZ])).all()
    centres = volume.centres
    assert centres.shape == torch.Size((600, 3))
    assert (centres[0] == Tensor([0, 0, 2 * SZ]) + (SZ / 2)).all()
    assert (centres[-1] == Tensor([LW[0] - SZ, LW[1] - SZ, 7 * SZ]) + (SZ / 2)).all()


def test_volume_forward_voxel(batch):
    layers = get_voxel_layers()
    volume = Volume(layers=layers)
    start = batch.copy()
    volume(batch)

    assert torch.abs(batch.z) <= 1e-5  # Muons traverse whole volume
    mask = batch.get_xy_mask((0, 0), LW)
    assert mask.sum() > N / 2  # At least half the muons stay inside the volume
    assert torch.all(batch.dtheta(start)[mask] > 0)  # All masked muons scatter

    hits = batch.get_hits((0, 0), LW)
    assert "above" in hits and "below" in hits
    assert hits["above"]["reco_xy"].shape[1] == 2
    assert hits["below"]["reco_xy"].shape[1] == 2
    assert hits["above"]["gen_xy"].shape[1] == 2
    assert hits["below"]["gen_xy"].shape[1] == 2
    assert torch.abs(hits["below"]["z"][0, 1, 0] - (SZ / 2)) < 1e-5  # Last Hit located at z-centre of last layer

    for i, l in enumerate(volume.get_detectors()):
        grad = jacobian(hits["above" if l.z > 0.5 else "below"]["reco_xy"][:, i % 2], l.resolution).sum((-1, -2))
        assert (grad == grad).sum() == 2 * len(grad)
        assert ((grad == grad) * (grad != 0)).sum() > 0  # every reco hit (x,y) is function of resolution


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


def test_volume_forward_panel(batch):
    layers = get_panel_layers(n_panels=4)
    volume = Volume(layers=layers)
    start = batch.copy()
    volume(batch)

    assert torch.abs(batch.z) <= 1e-5  # Muons traverse whole volume
    mask = batch.get_xy_mask((0, 0), LW)
    assert mask.sum() > N / 2  # At least half the muons stay inside the volume
    assert torch.all(batch.dtheta(start)[mask] > 0)  # All masked muons scatter

    hits = batch.get_hits()
    assert "above" in hits and "below" in hits
    assert hits["above"]["reco_xy"].shape[1] == 4
    assert hits["below"]["reco_xy"].shape[1] == 4
    assert hits["above"]["gen_xy"].shape[1] == 4
    assert hits["below"]["gen_xy"].shape[1] == 4

    # every reco hit (x,y) is function of panel position and size
    for i, l in enumerate(volume.get_detectors()):
        for j, p in enumerate(l.yield_zordered_panels()):
            for v in [p.xy, p.xy_span]:
                grad = jacobian(hits["above" if l.z > 0.5 else "below"]["reco_xy"][:, j], v).sum((-1))
                assert (grad == grad).sum() == 2 * len(grad)
                assert ((grad == grad) * (grad != 0)).sum() > 0


def test_detector_panel_properties():
    panel = DetectorPanel(res=1, eff=0.5, init_xyz=[0.5, 0.4, 0.9], init_xy_span=[0.3, 0.5], area_cost_func=area_cost, realistic_validation=False)
    assert panel.area_cost_func == area_cost
    assert panel.resolution == Tensor([1])
    assert panel.efficiency == Tensor([0.5])
    assert (panel.xy == Tensor([0.5, 0.4])).all()
    assert panel.z == Tensor([0.9])
    assert (panel.xy_span == Tensor([0.3, 0.5])).all()
    assert panel.x == Tensor([0.5])
    assert panel.y == Tensor([0.4])


def test_detector_panel_methods():
    panel = DetectorPanel(res=10, eff=0.5, init_xyz=[0.0, 0.01, 0.9], init_xy_span=[0.5, 0.51], area_cost_func=area_cost)

    # get_xy_mask
    mask = panel.get_xy_mask(Tensor([[0, 0], [0.1, 0.1], [0.25, 0.25], [0.5, 0.5], [1, 1], [0.1, 1], [1, 0.1], [-1, -1]]))
    assert (mask.int() == Tensor([1, 1, 0, 0, 0, 0, 0, 0])).all()

    # get_gauss
    with pytest.raises(ValueError):
        DetectorPanel(res=1, eff=0.5, init_xyz=[np.NaN, 0.0, 0.9], init_xy_span=[0.5, 0.5], area_cost_func=area_cost).get_gauss()
    with pytest.raises(ValueError):
        DetectorPanel(res=1, eff=0.5, init_xyz=[0.0, 0.0, 0.9], init_xy_span=[0.5, np.NaN], area_cost_func=area_cost).get_gauss()
    gauss = panel.get_gauss()
    assert (gauss.loc == Tensor([0.0, 0.01])).all()
    assert (gauss.scale == Tensor([0.5, 0.51])).all()

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
    panel = DetectorPanel(res=10, eff=0.5, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.5, 0.5], area_cost_func=area_cost)
    mg = MuonGenerator(x_range=(0, LW[0].item()), y_range=(0, LW[1].item()))
    mu = MuonBatch(mg(100), 1)
    mu.xy = torch.ones_like(mu.xy) / 2
    hits = panel.get_hits(mu)
    assert (hits["gen_xy"] == mu.xy).all()
    assert (hits["z"].mean() - 0.9).abs() < 1e-5
    assert (hits["reco_xy"].mean(0) - Tensor([0.5, 0.5]) < 0.25).all()

    panel.realistic_validation = True
    mu.xy = torch.zeros_like(mu.xy)
    hits = panel.get_hits(mu)
    assert hits["reco_xy"].isinf().sum() == 0

    panel.eval()
    hits = panel.get_hits(mu)
    assert hits["reco_xy"].isinf().sum() == 2 * len(mu)
    mu = MuonBatch(mg(100), 1)
    hits = panel.get_hits(mu)
    mask = hits["reco_xy"].isinf().prod(1) - 1 < 0
    # Reco hits can't leave panel
    assert (Tensor([0.25, 0.25]) <= hits["reco_xy"][mask]).all()
    assert (hits["reco_xy"][mask] <= Tensor([0.75, 0.75])).all()

    # get_cost
    cost = panel.get_cost()
    assert cost == area_cost(Tensor([0.5 * 0.5]))
    assert (torch.autograd.grad(cost, panel.xy_span, retain_graph=True, allow_unused=True)[0] > 0).all()

    # clamp_params
    panel.clamp_params((0, 0, 0.8), (1, 1, 1))
    assert panel.z == Tensor([0.9])
    assert (panel.xy == Tensor([0.5, 0.5])).all()
    assert (panel.xy_span == Tensor([0.5, 0.5])).all()

    panel = DetectorPanel(res=10, eff=0.5, init_xyz=[2.0, -2.0, 2.0], init_xy_span=[0.0, 20.0], area_cost_func=area_cost)
    panel.clamp_params((0, 0, 0), (1, 1, 1))
    assert (panel.xy == Tensor([1, 0])).all()
    assert panel.z - 1 < 0
    assert (panel.z - 1).abs() < 5e-3
    assert (panel.xy_span == Tensor([5e-2, 10])).all()
