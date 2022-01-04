from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tomopt.optimisation import VoxelX0Loss, VoxelClassLoss, VolumeClassLoss
from tomopt.optimisation.loss.loss import AbsDetectorLoss
from tomopt.volume import Volume

SHP = (6, 10, 10)


class MockLayer(nn.Module):
    device = torch.device("cpu")


def test_abs_detector_loss(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    mocker.patch("tomopt.optimisation.loss.loss.Volume.get_cost", return_value=cost)
    pred = torch.ones(SHP, requires_grad=True) / 2
    volume = Volume(nn.ModuleList([MockLayer()]))

    class DetectorLoss(AbsDetectorLoss):
        def _get_inference_loss(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
            return Tensor([3])

    # No target budget
    loss_func = DetectorLoss(target_budget=None)
    assert loss_func.sub_losses == {}
    assert loss_func._get_budget_coef(Tensor([0])) == Tensor([0])

    assert loss_func.cost_coef is None
    loss_func._compute_cost_coef(Tensor([2]))
    assert loss_func.cost_coef == Tensor([2])
    assert loss_func._get_cost_loss(volume) == Tensor([0])

    val = loss_func(pred, None, volume)
    assert val == Tensor([3])
    assert loss_func.sub_losses["error"] == 3
    assert loss_func.sub_losses["cost"] == Tensor([0])

    # With target budget
    loss_func = DetectorLoss(target_budget=cost)
    val = loss_func(pred, None, volume)
    assert val == Tensor([6])
    assert loss_func.sub_losses["error"] == 3
    assert loss_func.sub_losses["cost"] == Tensor([3])


def test_voxel_X0_loss(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones(SHP)
    volume = Volume(nn.ModuleList([MockLayer()]))
    mocker.patch.object(volume, "get_rad_cube", return_value=true)
    mocker.patch.object(volume, "get_cost", return_value=cost)
    pred = torch.ones(SHP, requires_grad=True) / 2

    loss_func = VoxelX0Loss(target_budget=None, cost_coef=0, debug=True)

    # Loss goes to zero
    loss_val = loss_func(true, torch.ones_like(pred), volume)
    assert loss_val == 0

    # Decreasing variance improves loss
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean()

    new_loss_val = loss_func(pred, 10 * torch.ones_like(pred), volume)
    assert new_loss_val < loss_val

    # Include cost
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=1, debug=True, steep_budget=True)
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean() + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() == grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget

    with torch.no_grad():
        cost /= cost
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=1, debug=True, steep_budget=False)
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean() + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget


def test_voxel_class_loss(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones(SHP)
    x02id = {1: 1}
    volume = Volume(nn.ModuleList([MockLayer()]))
    mocker.patch.object(volume, "get_rad_cube", return_value=true)
    mocker.patch.object(volume, "get_cost", return_value=cost)
    pred = F.log_softmax(torch.ones((1, 2, np.prod(SHP)), requires_grad=True) / 2, dim=1)

    loss_func = VoxelClassLoss(target_budget=None, cost_coef=0, debug=True, x02id=x02id)

    # Loss goes to zero
    correct = 10 * torch.ones((1, 2, np.prod(SHP)))
    correct[:, 0] = -10
    correct = F.log_softmax(correct, dim=1)
    loss_val = loss_func(correct, 1, volume)
    assert loss_val <= 1e-5

    # Decreasing variance improves loss
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true.flatten()[None].long())

    new_loss_val = loss_func(pred, 10, volume)
    assert new_loss_val < loss_val

    # Include cost
    loss_func = VoxelClassLoss(target_budget=1, cost_coef=1, debug=True, steep_budget=True, x02id=x02id)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true.flatten()[None].long()) + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() == grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget

    with torch.no_grad():
        cost /= cost
    loss_func = VoxelClassLoss(target_budget=1, cost_coef=1, debug=True, steep_budget=False, x02id=x02id)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true.flatten()[None].long()) + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget


def test_volume_class_loss_multi(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones((1)).long()
    volume = Volume(nn.ModuleList([MockLayer()]))
    x02id = {1: 1}
    volume._target = true
    mocker.patch.object(volume, "get_cost", return_value=cost)
    pred = F.log_softmax(torch.ones((1, 2), requires_grad=True) / 2, dim=1)

    loss_func = VolumeClassLoss(target_budget=None, cost_coef=0, debug=True, x02id=x02id)

    # Loss goes to zero
    correct = 10 * torch.ones((1, 2))
    correct[:, 0] = -10
    correct = F.log_softmax(correct, dim=1)
    loss_val = loss_func(correct, 1, volume)
    assert loss_val <= 1e-5

    # Decreasing variance improves loss
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true.long())

    new_loss_val = loss_func(pred, 10, volume)
    assert new_loss_val < loss_val

    # Include cost
    loss_func = VolumeClassLoss(target_budget=1, cost_coef=1, debug=True, steep_budget=True, x02id=x02id)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true.long()) + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() == grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget

    with torch.no_grad():
        cost /= cost
    loss_func = VolumeClassLoss(target_budget=1, cost_coef=1, debug=True, steep_budget=False, x02id=x02id)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true.long()) + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, 1, volume)
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget
