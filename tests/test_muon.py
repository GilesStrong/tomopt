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


@pytest.fixture
def batch():
    batch = Tensor([range(5) for _ in range(N)])
    return MuonBatch(batch, init_z=1)


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
    # theta x
    assert (data[:, 3] >= -np.pi / 2).all() and (data[:, 3] <= np.pi / 2).all()
    assert data[:, 3].mean() - 0.5 < 1e-1
    # theta y
    assert (data[:, 4] >= -np.pi / 2).all() and (data[:, 4] <= np.pi / 2).all()
    assert data[:, 4].mean() - 0.5 < 1e-1


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_muon_generator_from_volume():
    volume = Volume(get_panel_layers())
    mg = MuonGenerator.from_volume(volume)
    print(mg.x_range, mg.y_range)
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


def test_muon_batch_properties(batch):
    # shape
    assert len(batch) == N
    assert batch.muons.shape == torch.Size([N, 5])

    # Getters
    assert batch.x[0] == Tensor([0])
    assert batch.y[0] == Tensor([1])
    assert torch.all(batch.xy[0] == Tensor([0, 1]))
    assert batch.mom[0] == Tensor([2])
    assert batch.theta_x[0] == Tensor([3])
    assert batch.theta_y[0] == Tensor([4])
    assert batch.theta[0] == Tensor([5])
    assert batch.z == Tensor([1])

    # Setters
    new_coords = Tensor([range(5, 10) for _ in range(N)])
    batch.x = new_coords[:, 0]
    assert torch.all(batch.x == new_coords[:, 0])
    batch.y = new_coords[:, 1]
    assert torch.all(batch.y == new_coords[:, 1])
    batch.mom = new_coords[:, 2]
    assert torch.all(batch.mom == new_coords[:, 2])
    batch.theta_x = new_coords[:, 3]
    assert torch.all(batch.theta_x == new_coords[:, 3])
    batch.theta_y = new_coords[:, 4]
    assert torch.all(batch.theta_y == new_coords[:, 4])
    assert torch.all(batch.xy == new_coords[:, :2])
    assert torch.all(batch.theta == new_coords[:, 3:].pow(2).sum(1).sqrt())


def test_muon_batch_methods(batch):
    # copy & propagate
    start = batch.copy()
    batch.propagate(0.1)
    assert batch.z == Tensor([0.9])
    assert start.z == Tensor([1.0])
    assert torch.all(start.xy != batch.xy)
    assert torch.all(batch.x == start.x + (0.1 * torch.tan(start.theta_x)))
    assert torch.all(batch.y == start.y + (0.1 * torch.tan(start.theta_y)))

    # snapshot
    batch.snapshot_xyz()
    assert len(batch.xy_hist) == 1
    assert np.all(batch.xy_hist[list(batch.xy_hist.keys())[0]] == batch.xy.detach().cpu().clone().numpy())

    # mask
    lw = Tensor([1, 1])
    assert batch.get_xy_mask((0, 0), lw).sum() == 0
    batch.x = torch.zeros(N) + 0.5
    batch.y = torch.zeros(N) + 0.5
    assert batch.get_xy_mask((0, 0), lw).sum() == N

    # hits
    above_hits = {"xy": Tensor([[0, 0], [0, 0], [0, 0]]), "z": Tensor([[1], [1], [1]])}
    batch.append_hits(above_hits, "above")
    batch.append_hits(above_hits, "above")
    below_hits = {"xy": Tensor([[0, 0], [0, 0], [0, 0]]), "z": Tensor([[1], [1], [1]])}
    batch.append_hits(below_hits, "below")
    batch.append_hits(below_hits, "below")
    hits = batch.get_hits()
    assert len(hits) == 2
    assert torch.all(hits["above"]["xy"][:, 0] == above_hits["xy"])

    # deltas
    assert torch.all(batch.dtheta(start) == 0)
    batch.theta_x = batch.theta_x + 2
    batch.theta_y = batch.theta_y + 3
    assert batch.dtheta(start)[0] == Tensor([np.sqrt(13)])
