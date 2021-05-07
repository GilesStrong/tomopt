from pytest_mock import mocker  # noqa F401

import torch
from torch import nn

from tomopt.loss import DetectorLoss
from tomopt.volume import Volume

SHP = (6, 10, 10)


def test_detector_loss(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones(SHP)
    mocker.patch("tomopt.loss.loss.Volume.get_rad_cube", return_value=true)
    mocker.patch("tomopt.loss.loss.Volume.get_cost", return_value=cost)
    pred = torch.ones(SHP, requires_grad=True) / 2

    loss_func = DetectorLoss(0)
    loss_val = loss_func(pred, Volume(nn.ModuleList([])))
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean()

    loss_func = DetectorLoss(1)
    loss_val = loss_func(pred, Volume(nn.ModuleList([])))
    assert loss_val.shape == torch.Size([1])
    assert loss_val == (pred - true).pow(2).mean() + cost

    assert torch.autograd.grad(loss_val, pred)[0].abs().sum() > 0
    assert torch.autograd.grad(loss_val, cost)[0].abs().sum() > 0
