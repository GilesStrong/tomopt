from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tomopt.optimisation import VoxelX0Loss, VoxelClassLoss, VolumeClassLoss, VolumeIntClassLoss, integer_class_loss
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
    assert loss_val == F.nll_loss(pred, true)

    new_loss_val = loss_func(pred, 10, volume)
    assert new_loss_val < loss_val

    # Include cost
    loss_func = VolumeClassLoss(target_budget=1, cost_coef=1, debug=True, steep_budget=True, x02id=x02id)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.nll_loss(pred, true) + cost
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
    assert loss_val == F.nll_loss(pred, true) + cost
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


def test_volume_class_loss_binary(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones((1))
    volume = Volume(nn.ModuleList([MockLayer()]))
    x02id = {1: 1}
    volume._target = true
    mocker.patch.object(volume, "get_cost", return_value=cost)
    pred = torch.ones((1, 1), requires_grad=True) / 2

    loss_func = VolumeClassLoss(target_budget=None, cost_coef=0, debug=True, x02id=x02id)

    # Loss goes to zero
    correct = torch.ones((1, 1))
    loss_val = loss_func(correct, 1, volume)
    assert loss_val == 0

    # Decreasing variance improves loss
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.binary_cross_entropy(pred, true[:, None])

    new_loss_val = loss_func(pred, 10, volume)
    assert new_loss_val < loss_val

    # Include cost
    loss_func = VolumeClassLoss(target_budget=1, cost_coef=1, debug=True, steep_budget=True, x02id=x02id)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
    assert loss_val == F.binary_cross_entropy(pred, true[:, None]) + cost
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
    assert loss_val == F.binary_cross_entropy(pred, true[:, None]) + cost
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


def test_volume_int_class_loss(mocker):  # noqa F811
    cost = torch.ones((1), requires_grad=True)
    true = torch.ones((1)).long()
    volume = Volume(nn.ModuleList([MockLayer()]))
    volume._target = true
    mocker.patch.object(volume, "get_cost", return_value=cost)
    pred = F.softmax(torch.ones((1, 2), requires_grad=True) / 2, dim=1)

    loss_func = VolumeIntClassLoss(targ2int=lambda x, v: x.long(), pred_int_start=2, use_mse=False, target_budget=None, cost_coef=0, debug=True)

    # Decreasing variance improves loss
    loss_val = loss_func(pred, torch.ones_like(pred), volume)
    assert loss_val.shape == torch.Size([1])
    new_loss_val = loss_func(pred, 10, volume)
    assert new_loss_val < loss_val

    # Include cost
    loss_func = VolumeIntClassLoss(targ2int=lambda x, v: x.long(), pred_int_start=2, use_mse=False, target_budget=1, cost_coef=1, debug=True, steep_budget=True)
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
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
    loss_func = VolumeIntClassLoss(
        targ2int=lambda x, v: x.long(), pred_int_start=2, use_mse=False, target_budget=1, cost_coef=1, debug=True, steep_budget=False
    )
    loss_val = loss_func(pred, 1, volume)
    assert loss_val.shape == torch.Size([1])
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

    # Behaviour
    loss_func = VolumeIntClassLoss(targ2int=lambda x, v: x.long(), pred_int_start=2, use_mse=False, target_budget=None, cost_coef=0, debug=True)
    # Good prediction = low loss
    volume._target = Tensor([[0]]).long()
    good_loss = loss_func(F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), 1, volume)
    # Close prediction = higher loss
    close_loss = loss_func(F.softmax(Tensor([[1, 10, 5, 3]]), dim=1), 1, volume)
    assert close_loss > good_loss
    # Far prediction = highest loss
    far_loss = loss_func(F.softmax(Tensor([[1, 3, 5, 10]]), dim=1), 1, volume)
    assert far_loss > close_loss

    # MSE affects loss
    loss_func.use_mse = True
    mse_far_loss = loss_func(F.softmax(Tensor([[1, 3, 5, 10]]), dim=1), 1, volume)
    assert mse_far_loss > far_loss

    # pred int start works
    loss_func.use_mse = False
    volume._target = Tensor([[1]]).long()
    zero_loss = loss_func(F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), 1, volume)
    loss_func.pred_int_start = 1
    one_loss = loss_func(F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), 1, volume)
    assert one_loss < zero_loss

    # targ2int works
    volume._target = Tensor([1.7])
    assert loss_func(F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), 1, volume) == one_loss


def test_integer_class_loss():
    # Behaviour
    # Good prediction = low loss
    good_loss = integer_class_loss(int_probs=F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), target_int=0, pred_start_int=0, use_mse=False)
    # Close prediction = higher loss
    close_loss = integer_class_loss(int_probs=F.softmax(Tensor([[1, 10, 5, 3]]), dim=1), target_int=0, pred_start_int=0, use_mse=False)
    assert close_loss > good_loss
    # Far prediction = highest loss
    far_loss = integer_class_loss(int_probs=F.softmax(Tensor([[1, 3, 5, 10]]), dim=1), target_int=0, pred_start_int=0, use_mse=False)
    assert far_loss > close_loss

    # MSE affects loss
    mse_far_loss = integer_class_loss(int_probs=F.softmax(Tensor([[1, 3, 5, 10]]), dim=1), target_int=0, pred_start_int=0, use_mse=True)
    assert mse_far_loss > far_loss

    # pred int start works
    zero_loss = integer_class_loss(int_probs=F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), target_int=1, pred_start_int=0, use_mse=False)
    one_loss = integer_class_loss(int_probs=F.softmax(Tensor([[10, 0, 0, 0]]), dim=1), target_int=1, pred_start_int=1, use_mse=False)
    assert one_loss < zero_loss

    # reductions
    ur_loss = integer_class_loss(
        int_probs=F.softmax(Tensor([[10, 0, 0, 0], [10, 0, 0, 0]]), dim=1), target_int=Tensor([[0], [1]]), pred_start_int=0, use_mse=False, reduction="none"
    )
    assert ur_loss.shape == torch.Size([2, 1])
    s_loss = integer_class_loss(
        int_probs=F.softmax(Tensor([[10, 0, 0, 0], [10, 0, 0, 0]]), dim=1), target_int=Tensor([[0], [1]]), pred_start_int=0, use_mse=False, reduction="sum"
    )
    assert s_loss.shape == torch.Size([])
    assert ur_loss.sum() == s_loss
    m_loss = integer_class_loss(
        int_probs=F.softmax(Tensor([[10, 0, 0, 0], [10, 0, 0, 0]]), dim=1), target_int=Tensor([[0], [1]]), pred_start_int=0, use_mse=False, reduction="mean"
    )
    assert m_loss.shape == torch.Size([])
    assert ur_loss.mean() == m_loss

    # weights
    loss = integer_class_loss(
        int_probs=F.softmax(Tensor([[10, 0, 0, 0], [10, 0, 0, 0]]), dim=1),
        target_int=Tensor([[0], [0]]),
        weight=Tensor([[1], [10]]),
        pred_start_int=0,
        use_mse=False,
        reduction="none",
    )
    assert loss.shape == torch.Size([2, 1])
    assert ((loss[0] * 10) - loss[1]).abs() < 1e-5

    # diff check
    x = torch.tensor([[10.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss = integer_class_loss(int_probs=F.softmax(x, dim=1), target_int=0, pred_start_int=0, use_mse=False)
    assert torch.autograd.grad(loss, x)[0].abs().sum() > 0
