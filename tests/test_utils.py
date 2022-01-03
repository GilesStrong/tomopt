import torch
from torch import Tensor
import numpy as np
import pytest

from tomopt.utils import jacobian, class_to_x0preds


def test_jacobian():
    def y1(x):
        return x ** 2

    x = torch.randn((10), requires_grad=True)
    y = y1(x)
    assert torch.all(jacobian(y, x).sum(1) == 2 * x)

    def y2(x):
        return (x[:, 0] ** 2) + x[:, 1]

    x = torch.randn((10, 2), requires_grad=True)
    y = y2(x)
    dy = torch.stack([2 * x[:, 0], torch.ones(10)], dim=1)
    assert torch.all(jacobian(y, x).sum(1) == dy)

    def non_vmap_jacobian(y: Tensor, x: Tensor):
        jac = []
        flat_y = y.reshape(-1)
        for grad_y in torch.eye(len(flat_y)).unbind():
            (grad_x,) = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True)
            jac.append(grad_x.reshape(x.shape))
        return torch.stack(jac).reshape(y.shape + x.shape)

    assert torch.all(jacobian(y, x) == non_vmap_jacobian(y, x))


def test_class_to_x0preds():
    id2x0 = {0: -1.0, 1: -2.0}
    arr = np.array([0, 1, 1])
    tarr = class_to_x0preds(arr, id2x0)
    assert tarr[0] == -1
    assert (tarr[1:] == -2).all()
    assert (arr == np.array([0, 1, 1])).all()

    with pytest.raises(KeyError):
        class_to_x0preds(np.array([0, 1, 1, 2]), id2x0)
