import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tomopt.muon import MuonGenerator, MuonBatch
from tomopt.volume import PassiveLayer, Volume, PanelDetectorLayer, DetectorPanel
from tomopt.core import X0

N = 3
LW = Tensor([1, 1])
SZ = 0.1
Z = 1


def area_cost(a: Tensor) -> Tensor:
    return F.relu(a)


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
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0], area_cost_func=area_cost)
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
                    res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0], area_cost_func=area_cost
                )
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_muon_generator():
    n_muons = 10000
    gen = MuonGenerator((0, 1.0), (0, 1.0), fixed_mom=None, energy_range=(0.5, 500))
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
def test_muon_generator_from_volume():
    volume = Volume(get_panel_layers())
    mg = MuonGenerator.from_volume(volume)
    assert mg.x_range[0] < LW[0] and mg.x_range[1] > LW[1]

    n = 10000
    mu = MuonBatch(mg(n), volume.h)
    mu.propagate(volume.h - volume.get_passive_z_range()[1])
    m1 = mu.get_xy_mask(
        (
            0,
            0,
        ),
        LW.numpy().tolist(),
    )
    mu.propagate(volume.get_passive_z_range()[1] - volume.get_passive_z_range()[0])
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
    assert batch.muons.shape == torch.Size([N, 5])

    # Getters
    assert batch.x[0] == Tensor([0])
    assert batch.y[0] == Tensor([1])
    assert torch.all(batch.xy[0] == Tensor([0, 1]))
    assert batch.mom[0] == Tensor([2])
    assert batch.reco_mom[0] == Tensor([2])
    assert batch.theta[0] == Tensor([3])
    assert batch.phi[0] == Tensor([4])
    assert batch.z == Tensor([1])

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
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    # copy & propagate
    start = batch.copy()
    batch.propagate(0.1)
    assert batch.z == Tensor([0.9])
    assert start.z == Tensor([1.0])
    assert torch.all(start.xy != batch.xy)
    assert torch.all(batch.x >= start.x)
    assert torch.all(batch.y <= start.y)

    # snapshot
    batch.snapshot_xyz()
    assert len(batch.xy_hist) == 1
    assert np.all(batch.xy_hist[list(batch.xy_hist.keys())[0]] == batch.xy.detach().cpu().clone().numpy())

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
    assert torch.all(batch.dtheta(start) == 0)
    batch._theta = batch.theta + 2
    assert batch.dtheta(start)[0] == Tensor([2])

    # Angle calculations
    tx, ty = 2 * torch.pi * (torch.rand(100) - 0.5), 2 * torch.pi * (torch.rand(100) - 0.5)
    phi = MuonBatch.phi_from_theta_xy(tx, ty)
    assert (phi < 2 * torch.pi).all() and (phi > 0).all() and (phi.max() > torch.pi) and (phi.min() < torch.pi)
    theta = MuonBatch.theta_from_theta_xy(tx, ty)
    assert (theta < torch.pi).all() and (theta > 0).all() and (theta.max() > torch.pi / 2) and (theta.min() < torch.pi / 2)

    # Remove upward muons
    batch._theta = Tensor([2, 1, 1])
    batch.remove_upwards_muons()
    assert len(batch) == 2
    assert (batch.theta < torch.pi / 2).all()
    hits = batch.get_hits()
    assert len(hits["above"]["xy"]) == 2


def test_muon_batch_scatter_dxy():
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 6], [1, 0, 0, 1, 6]]), init_z=1)
    # copy & propagate
    start = batch.copy()

    batch.scatter_dxy()
    assert (batch.xy == start.xy).all()

    batch.scatter_dxy(dx=Tensor([1, -1]))
    assert (batch.x != start.x).all()
    assert (batch.y == start.y).all()

    batch = start.copy()
    batch.scatter_dxy(dy=Tensor([1, -1]))
    assert (batch.y != start.y).all()
    assert (batch.x == start.x).all()

    batch = start.copy()
    batch.scatter_dxy(dx=Tensor([1, -1]), dy=Tensor([1, -1]))
    assert (batch.xy != start.xy).all()

    batch = start.copy()
    batch.scatter_dxy(dx=Tensor([1]), dy=Tensor([1]), mask=Tensor([1, 0]).bool())
    assert batch.x[0] != start.x[0]
    assert batch.x[1] == start.x[1]
    assert batch.y[0] != start.y[0]
    assert batch.y[1] == start.y[1]

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

    batch.scatter_dtheta_dphi(dtheta=Tensor([1, -0.9, -2]))
    assert len(batch) == len(start) - 1  # upwards muon removed
    assert (batch.theta >= 0).all() and (batch.theta < torch.pi / 2).all()
    assert batch.theta[0] != start.theta[1]
    assert batch.theta[1] == start.theta[2]
    assert batch.phi[0] == start.phi[1]
    assert batch.phi[1] == (start.phi[2] + torch.pi) % (2 * torch.pi)

    batch = start.copy()
    batch.scatter_dtheta_dphi(dphi=Tensor([1, -1, 3]))
    assert (batch.phi != start.phi).all()
    assert (batch.theta == start.theta).all()

    batch = start.copy()
    batch.scatter_dtheta_dphi(dtheta=Tensor([0.1, -0.9, -0.2]), dphi=Tensor([1, -1, 3]))
    assert (batch.theta != start.theta).all()
    assert (batch.phi != start.phi).all()

    batch = start.copy()
    batch.scatter_dtheta_dphi(dtheta=Tensor([-0.5]), dphi=Tensor([1]), mask=Tensor([1, 0, 0]).bool())
    assert batch.theta[0] != start.theta[0]
    assert batch.theta[1] == start.theta[1]
    assert batch.phi[0] != start.phi[0]
    assert batch.phi[1] == start.phi[1]

    assert (batch.xy == start.xy).all()


def test_muon_batch_scatter_dtheta_xy():
    batch = MuonBatch(Tensor([[1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0]]), init_z=1)
    # copy & propagate
    start = batch.copy()

    batch.scatter_dtheta_xy()
    assert (batch.theta - start.theta).abs().sum() < 1e-5
    assert (batch.phi - start.phi).abs().sum() < 1e-5

    batch.scatter_dtheta_xy(dtheta_x=Tensor([3, -0.9, -2]))
    print(batch.theta)
    assert len(batch) == len(start) - 1  # upwards muon removed
    assert (batch.theta >= 0).all() and (batch.theta < torch.pi / 2).all()
    assert (batch.theta[0] - start.theta[1]).abs().sum() > 1e-2
    assert (batch.theta[1] - start.theta[2]).abs().sum() < 1e-5
    assert (batch.phi[0] - start.phi[1]).abs().sum() < 1e-5
    assert (batch.phi[1] - (start.phi[2] + torch.pi) % (2 * torch.pi)).abs().sum() < 1e-5

    batch = start.copy()
    batch.scatter_dtheta_xy(dtheta_y=Tensor([0.1, -0.9, -1]))
    assert (batch.phi != start.phi).all()
    assert (batch.theta >= start.theta).all()

    batch = start.copy()
    batch.scatter_dtheta_xy(dtheta_x=Tensor([0.1, -0.9, -0.2]), dtheta_y=Tensor([0.1, -1, 0.1]))
    assert (batch.theta != start.theta).all()
    assert (batch.phi != start.phi).all()

    batch = start.copy()
    batch.scatter_dtheta_xy(dtheta_x=Tensor([-0.5]), dtheta_y=Tensor([1]), mask=Tensor([1, 0, 0]).bool())
    assert batch.theta[0] != start.theta[0]
    assert batch.theta[1] == start.theta[1]
    assert batch.phi[0] != start.phi[0]
    assert batch.phi[1] == start.phi[1]

    assert (batch.xy == start.xy).all()


def test_muon_batch_angle_consistency():
    volume = Volume(get_panel_layers())
    mg = MuonGenerator.from_volume(volume)
    n = 10000
    mu_orig = MuonBatch(mg(n), volume.h)

    # mu_orig._theta = torch.pi*torch.rand(n)
    # mu_orig._phi = 2*torch.pi*torch.rand(n)

    # Angle conversion
    tx, ty = mu_orig.theta_x, mu_orig.theta_y
    t, p = mu_orig.theta_from_theta_xy(tx, ty), mu_orig.phi_from_theta_xy(tx, ty)
    assert (t - mu_orig.theta).sum().abs() < 1e-3
    assert (p - mu_orig.phi).sum().abs() < 1e-3
    tx_n, ty_n = mu_orig.theta_x_from_theta_phi(t, p), mu_orig.theta_y_from_theta_phi(t, p)
    assert (tx - tx_n).sum().abs() < 1e-3
    assert (ty - ty_n).sum().abs() < 1e-3

    # Scattering dtheta dphi -> dtheta_xy
    dtheta, dphi = torch.pi * torch.rand(n), 2 * torch.pi * torch.rand(n)
    dtheta_x, dtheta_y = mu_orig.theta_x_from_theta_phi(dtheta, dphi), mu_orig.theta_y_from_theta_phi(dtheta, dphi)
    print(dtheta.min(), dtheta.max())
    print(dphi.min(), dphi.max())
    print(dtheta_x.min(), dtheta_x.max())
    print(dtheta_y.min(), dtheta_y.max())
    assert dtheta_x.mean().abs() < 1e-2
    assert dtheta_y.mean().abs() < 1e2
    mu_dtp = mu_orig.copy()
    mu_dtxy = mu_orig.copy()
    mu_dtp.scatter_dtheta_dphi(dtheta=dtheta, dphi=dphi)
    mu_dtxy.scatter_dtheta_xy(dtheta_x=dtheta_x, dtheta_y=dtheta_y)
    assert len(mu_dtxy) == len(mu_dtp)
    assert (mu_dtxy.theta - mu_dtp.theta).sum().abs() < 1e-5
    assert (mu_dtxy.phi - mu_dtp.phi).sum().abs() < 1e-5
    assert (mu_dtxy.theta_x - mu_dtp.theta_x).sum().abs() < 1e-5
    assert (mu_dtxy.theta_y - mu_dtp.theta_y).sum().abs() < 1e-5

    # Scattering dtheta_xy -> dtheta dphi
    dtheta_x, dtheta_y = 2 * torch.pi * (torch.rand(n) - 0.5), 2 * torch.pi * (torch.rand(n) - 0.5)
    dtheta, dphi = mu_orig.theta_from_theta_xy(dtheta_x, dtheta_y), mu_orig.phi_from_theta_xy(dtheta_x, dtheta_y)
    print(dtheta.min(), dtheta.max())
    print(dphi.min(), dphi.max())
    print(dtheta_x.min(), dtheta_x.max())
    print(dtheta_y.min(), dtheta_y.max())
    assert (dphi.mean() - torch.pi).abs() < 1e-2
    mu_dtp = mu_orig.copy()
    mu_dtxy = mu_orig.copy()
    mu_dtp.scatter_dtheta_dphi(dtheta=dtheta, dphi=dphi)
    mu_dtxy.scatter_dtheta_xy(dtheta_x=dtheta_x, dtheta_y=dtheta_y)
    assert len(mu_dtxy) == len(mu_dtp)
    assert (mu_dtxy.theta - mu_dtp.theta).sum().abs() < 1e-5
    assert (mu_dtxy.phi - mu_dtp.phi).sum().abs() < 1e-5
    assert (mu_dtxy.theta_x - mu_dtp.theta_x).sum().abs() < 1e-5
    assert (mu_dtxy.theta_y - mu_dtp.theta_y).sum().abs() < 1e-5
