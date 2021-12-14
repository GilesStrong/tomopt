from pytest_mock import mocker  # noqa F401

import torch
from torch import nn

from tomopt.optimisation import VoxelX0Loss
from tomopt.volume import Volume

SHP = (6, 10, 10)


class MockLayer(nn.Module):
    device = torch.device("cpu")


def test_detector_loss(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones(SHP)
    mocker.patch("tomopt.optimisation.loss.loss.Volume.get_rad_cube", return_value=true)
    mocker.patch("tomopt.optimisation.loss.loss.Volume.get_cost", return_value=cost)
    pred = torch.ones(SHP, requires_grad=True) / 2

    loss_func = VoxelX0Loss(target_budget=None, cost_coef=0, debug=True)
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean()

    # Decreasing variance improves loss
    new_loss_val = loss_func(pred, 10 * torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert new_loss_val < loss_val

    loss_func = VoxelX0Loss(target_budget=1, cost_coef=1, debug=True, steep_budget=True)
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean() + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() == grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget

    with torch.no_grad():
        cost /= cost
    loss_func = VoxelX0Loss(target_budget=1, cost_coef=1, debug=True, steep_budget=False)
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean() + cost
    assert loss_val > 0

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert (grad_at_budget := torch.autograd.grad(loss_val, cost)[0].abs().sum()) > 0

    with torch.no_grad():
        cost += 1
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget
    with torch.no_grad():
        cost /= 4
    loss_val = loss_func(pred, torch.ones_like(pred), Volume(nn.ModuleList([MockLayer()])))
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() < grad_at_budget
