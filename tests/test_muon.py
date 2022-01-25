from unittest import mock
import pytest
import numpy as np

import torch
import tomopt
from torch import Tensor

from tomopt.muon import MuonGenerator, MuonBatch


N = 3


@pytest.fixture
def batch():
    batch = Tensor([range(5) for _ in range(N)])
    return MuonBatch(batch, init_z=1)


mocker = mock


def test_muon_generator(mocker):
    n_muons = int(1e4)
    gen = MuonGenerator(1.0, 1.0, True)
    mocker.patch("tomopt.muon.generation.np.random.uniform", return_value=np.ones(n_muons))
    set = gen.generate_set(n_muons)
    momenta = 2.7619 * np.ones(n_muons) if gen._sample_momentum is True else 5.0 * np.ones(n_muons)
    theta_x = 0.7553 * np.ones(n_muons)
    theta_y = -0.6554 * np.ones(n_muons)
    compare = torch.Tensor(np.stack([np.ones(n_muons), np.ones(n_muons), momenta, theta_x, theta_y], axis=1))
    print(torch.sum(torch.round(set)) == torch.sum(torch.round(compare)))
    assert set.shape == (n_muons, 5)
    assert torch.sum(torch.round(set)) == torch.sum(torch.round(compare))


def test_muon_dataset():
    np.random.seed(42)
    gen = MuonGenerator(1.0, 1.0, True)
    set = gen.generate_set(1000)
    assert (set[:, 2] < 0).any
    assert (set[:, 0] > gen._dimensions[0]).any or (set[:, 1] > gen._dimensions[1]).any
    assert (np.abs(set[:, 3]) > np.pi).any or (np.abs(set[:, 4]) > np.pi).any


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
