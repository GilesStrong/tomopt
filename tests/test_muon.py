import numpy as np
import pytest
import torch
from pytest_mock import mocker  # noqa F401
from torch import Tensor, nn

from tomopt.core import X0
from tomopt.muon import MuonBatch, MuonGenerator2015, MuonGenerator2016
from tomopt.volume import DetectorPanel, PanelDetectorLayer, PassiveLayer, Volume

N = 3
LW = Tensor([1, 1])
SZ = 0.1
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["aluminium"]
    if z >= 0.5:
        rad_length[3:7, 3:7] = X0["lead"]
    return rad_length


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
        layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))
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


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize("generator", [MuonGenerator2016, MuonGenerator2015])
def test_muon_generator(generator):
    n_muons = 10000
    gen = generator((0, 1.0), (0, 1.0), fixed_mom=None, energy_range=(0.5, 500))
    data = gen.generate_set(n_muons)
    assert data.shape == (n_muons, 5)
    # x pos
    assert (data[:, 0] >= 0).all() and (data[:, 0] <= 1).all()
    assert data[:, 0].mean() - 0.5 < 1e-1
    # y pos
    assert (data[:, 1] >= 0).all() and (data[:, 1] <= 1).all()
    assert data[:, 1].mean() - 0.5 < 1e-1
    # p
    assert (data[:, 2] >= np.sqrt((0.5**2) - gen._muon_mass2)).all() and (data[:, 2] <= np.sqrt((500**2) - gen._muon_mass2)).all()
    assert data[:, 2].mean() - 2 <= 1e-1
    # theta
    assert (data[:, 3] >= 0).all() and (data[:, 3] <= np.pi / 2).all()
    # phi
    assert (data[:, 4] >= 0).all() and (data[:, 4] <= 2 * np.pi).all()
    assert data[:, 4].mean() - np.pi < 1e-1


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize("generator", [MuonGenerator2016, MuonGenerator2015])
def test_muon_generator_from_volume(generator):
    volume = Volume(get_panel_layers())
    mg = generator.from_volume(volume)
    assert mg.x_range[0] < LW[0] and mg.x_range[1] > LW[1]

    n = 10000
    mu = MuonBatch(mg(n), volume.h)
    mu.propagate_dz(volume.h - volume.get_passive_z_range()[1])
    m1 = mu.get_xy_mask(
        (
            0,
            0,
        ),
        LW.numpy().tolist(),
    )
    mu.propagate_dz(volume.get_passive_z_range()[1] - volume.get_passive_z_range()[0])
    m2 = mu.get_xy_mask(
        (
            0,
            0,
        ),
        LW.numpy().tolist(),
    )
    assert (m1 + m2).sum() / n > 0.5


def test_muon_batch_properties():
    batch = MuonBatch(Tensor([range(5) for _ in range(N)]), init_z=1)
    # shape
    assert len(batch) == N
    assert batch.muons.shape == torch.Size([N, 6])

    # Getters
    assert batch.x[0] == Tensor([0])
    assert batch.y[0] == Tensor([1])
    assert torch.all(batch.xy[0] == Tensor([0, 1]))
    assert batch.mom[0] == Tensor([2])
    assert batch.reco_mom[0] == Tensor([2])
    assert batch.theta[0] == Tensor([3])
    assert batch.phi[0] == Tensor([4])
    assert batch.z[0] == Tensor([1])

    # Setters
    new_coords = Tensor([range(5, 10) for _ in range(N)])
    with pytest.raises(AttributeError):
        batch.x = new_coords[:, 0]
    with pytest.raises(AttributeError):
        batch.y = new_coords[:, 1]
    with pytest.raises(AttributeError):
        batch.xy = new_coords[:, :1]
    with pytest.raises(AttributeError):
        batch.z = 0.2
    with pytest.raises(NotImplementedError):
        batch.mom = new_coords[:, 2]
    with pytest.raises(NotImplementedError):
        batch.reco_mom = new_coords[:, 2]
    with pytest.raises(AttributeError):
        batch.theta = new_coords[:, 3]
    with pytest.raises(AttributeError):
        batch.phi = new_coords[:, 4]
    with pytest.raises(AttributeError):
        batch.theta_x = new_coords[:, 3]
    with pytest.raises(AttributeError):
        batch.theta_y = new_coords[:, 3]


def test_muon_batch_methods():
    # copy & propagate_dz
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    start = batch.copy()
    batch.propagate_dz(0.1)
    assert (batch.z == Tensor([0.9])).all()
    assert (start.z == Tensor([1.0])).all()
    assert torch.all(start.xy != batch.xy)
    assert torch.all(batch.x >= start.x)
    assert torch.all(batch.y <= start.y)

    batch = MuonBatch(Tensor([[0, 0, 0, np.pi / 4, 3 * np.pi / 4]]), init_z=0)
    batch.propagate_dz(1)
    assert ((batch.xyz - Tensor([-np.sqrt(2) / 2, np.sqrt(2) / 2, -1])).abs() < 1e-5).all()

    # copy & propagate_d
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    start = batch.copy()
    batch.propagate_d(0.1)
    assert (batch.z < start.z).all()
    assert torch.all(start.xy != batch.xy)
    assert torch.all(batch.x >= start.x)
    assert torch.all(batch.y <= start.y)

    batch = MuonBatch(Tensor([[0, 0, 0, np.arccos(1 / np.sqrt(3)), 3 * np.pi / 4]]), init_z=0)
    d = np.sqrt(3)
    batch.propagate_d(d)
    assert (batch.xyz.square().sum().sqrt() - d).abs() < 1e-5
    assert ((batch.xyz - Tensor([[-1, 1, -1]])).abs() < 1e-5).all()

    # propagation consistency
    batch = MuonBatch(Tensor([[1, 0, 0, 0, 6], [1, 0, 0, 0, 6], [1, 0, 0, 0, 6]]), init_z=1)
    batch2 = batch.copy()
    batch.propagate_d(0.1)
    batch2.propagate_dz(0.1)
    assert torch.all((batch.xyz - batch2.xyz).abs() < 1e-5)

    # copy & propagate_d
    batch = start.copy()
    batch.propagate_d(0.1)
    assert (batch.z < start.z).all()
    assert torch.all(start.xy != batch.xy)

    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    # snapshot
    batch.snapshot_xyz()
    assert len(batch.xyz_hist) == 1
    assert np.all(batch.xyz_hist[0] == batch.xyz.detach().cpu().clone().numpy())

    # mask
    lw = Tensor([1, 1])
    assert batch.get_xy_mask((0, 0), lw).sum() == 0
    batch._x = torch.zeros(3) + 0.5
    batch._y = torch.zeros(3) + 0.5
    assert batch.get_xy_mask((0, 0), lw).sum() == 3

    # hits
    above_hits = {"xy": Tensor([[0, 0], [0, 0], [0, 0]]), "z": Tensor([[1], [1], [1]])}
    batch.append_hits(above_hits, "above")
    batch.append_hits(above_hits, "above")
    below_hits = {"xy": Tensor([[0, 0], [0, 0], [0, 0]]), "z": Tensor([[1], [1], [1]])}
    batch.append_hits(below_hits, "below")
    batch.append_hits(below_hits, "below")
    hits = batch.get_hits()
    assert len(hits) == 2
    assert len(hits["above"]["xy"]) == 3
    assert torch.all(hits["above"]["xy"][:, 0] == above_hits["xy"])

    # deltas
    assert torch.all(batch.dtheta(start.theta) == 0)
    batch._theta = batch.theta + 2
    assert batch.dtheta(start.theta)[0] == Tensor([2])

    # Angle calculations
    tx, ty = torch.pi * (torch.rand(100) - 0.5), torch.pi * (torch.rand(100) - 0.5)
    phi = MuonBatch.phi_from_theta_xy(tx, ty)
    assert (phi < 2 * torch.pi).all() and (phi > 0).all() and (phi.max() > torch.pi) and (phi.min() < torch.pi)
    theta = MuonBatch.theta_from_theta_xy(tx, ty)
    assert (theta < torch.pi / 2).all() and (theta > 0).all() and (theta.max() > torch.pi / 4) and (theta.min() < torch.pi / 4)

    # Remove upward muons
    batch._theta = Tensor([2, 1, 1])
    batch.remove_upwards_muons()
    assert len(batch) == 2
    assert (batch.theta < torch.pi / 2).all()
    hits = batch.get_hits()
    assert len(hits["above"]["xy"]) == 2


def test_muon_batch_scatter_dxyz():
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    # copy & propagate
    start = batch.copy()

    batch.scatter_dxyz()
    assert (batch.xyz == start.xyz).all()

    batch.scatter_dxyz(dx_vol=Tensor([1, -1]))
    assert (batch.x != start.x).all()
    assert (batch.y == start.y).all()
    assert (batch.z == start.z).all()

    batch = start.copy()
    batch.scatter_dxyz(dy_vol=Tensor([1, -1]))
    assert (batch.y != start.y).all()
    assert (batch.x == start.x).all()
    assert (batch.z == start.z).all()

    batch = start.copy()
    batch.scatter_dxyz(dx_vol=Tensor([1, -1]), dy_vol=Tensor([1, -1]), dz_vol=Tensor([1, -1]))
    assert (batch.xyz != start.xyz).all()

    batch = start.copy()
    batch.scatter_dxyz(dx_vol=Tensor([1]), dy_vol=Tensor([1]), dz_vol=Tensor([1]), mask=Tensor([1, 0]).bool())
    assert batch.x[0] != start.x[0]
    assert batch.x[1] == start.x[1]
    assert batch.y[0] != start.y[0]
    assert batch.y[1] == start.y[1]
    assert batch.z[0] != start.z[0]
    assert batch.z[1] == start.z[1]

    assert (batch.theta == start.theta).all()
    assert (batch.phi == start.phi).all()
    assert len(batch) == len(start)


def test_muon_batch_scatter_dtheta_dphi():
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    # copy & propagate
    start = batch.copy()

    batch.scatter_dtheta_dphi()
    assert (batch.theta == start.theta).all()
    assert (batch.phi == start.phi).all()

    batch.scatter_dtheta_dphi(dtheta_vol=Tensor([1, -0.9, -2]))
    assert len(batch) == len(start) - 1  # upwards muon removed
    assert (batch.theta >= 0).all() and (batch.theta < torch.pi / 2).all()
    assert batch.theta[0] != start.theta[1]
    assert batch.theta[1] == start.theta[2]
    assert batch.phi[0] == start.phi[1]
    assert batch.phi[1] == (start.phi[2] + torch.pi) % (2 * torch.pi)

    batch = start.copy()
    batch.scatter_dtheta_dphi(dphi_vol=Tensor([1, -1, 3]))
    assert (batch.phi != start.phi).all()
    assert (batch.theta == start.theta).all()

    batch = start.copy()
    batch.scatter_dtheta_dphi(dtheta_vol=Tensor([0.1, -0.9, -0.2]), dphi_vol=Tensor([1, -1, 3]))
    assert (batch.theta != start.theta).all()
    assert (batch.phi != start.phi).all()

    batch = start.copy()
    batch.scatter_dtheta_dphi(dtheta_vol=Tensor([-0.5]), dphi_vol=Tensor([1]), mask=Tensor([1, 0, 0]).bool())
    assert batch.theta[0] != start.theta[0]
    assert batch.theta[1] == start.theta[1]
    assert batch.phi[0] != start.phi[0]
    assert batch.phi[1] == start.phi[1]

    assert (batch.xy == start.xy).all()


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_muon_batch_angle_consistency():
    volume = Volume(get_panel_layers())
    mg = MuonGenerator2016.from_volume(volume)
    n = 10000
    mu_orig = MuonBatch(mg(n), volume.h)

    # Realistic angles
    tx, ty = mu_orig.theta_x, mu_orig.theta_y
    t, p = mu_orig.theta_from_theta_xy(tx, ty), mu_orig.phi_from_theta_xy(tx, ty)
    assert (t - mu_orig.theta).sum().abs() < 1e-3
    assert (p - mu_orig.phi).sum().abs() < 1e-3
    tx_n, ty_n = mu_orig.theta_x_from_theta_phi(t, p), mu_orig.theta_y_from_theta_phi(t, p)
    assert (tx - tx_n).sum().abs() < 1e-3
    assert (ty - ty_n).sum().abs() < 1e-3

    # All angles
    mu_orig._theta = torch.pi * torch.rand(n) / 2
    mu_orig._phi = 2 * torch.pi * torch.rand(n)

    tx, ty = mu_orig.theta_x, mu_orig.theta_y
    t, p = mu_orig.theta_from_theta_xy(tx, ty), mu_orig.phi_from_theta_xy(tx, ty)
    assert (t - mu_orig.theta).sum().abs() < 1e-3
    assert (p - mu_orig.phi).sum().abs() < 1e-3
    tx_n, ty_n = mu_orig.theta_x_from_theta_phi(t, p), mu_orig.theta_y_from_theta_phi(t, p)
    assert (tx - tx_n).sum().abs() < 1e-3
    assert (ty - ty_n).sum().abs() < 1e-3
