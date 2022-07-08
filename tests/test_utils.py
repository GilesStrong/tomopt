import torch
from torch import Tensor
import numpy as np
import pytest

from tomopt.utils import jacobian, class_to_x0preds, x0targs_to_classtargs, x0_from_mixture


def test_jacobian():
    def y1(x):
        return x**2

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
    id2x0 = {0: -1.5, 1: -2.0}
    arr = np.array([0, 1, 1])
    tarr = class_to_x0preds(arr, id2x0)
    assert tarr[0] == -1.5
    assert (tarr[1:] == -2.0).all()
    assert (arr == np.array([0, 1, 1])).all()

    with pytest.raises(KeyError):
        class_to_x0preds(np.array([0, 1, 1, 2]), id2x0)


def test_x0targs_to_classtargs():
    id2x0 = {-1.5: 0, -2.0: 1}
    arr = np.array([-1.5, -2.0, -3.0])
    tarr = x0targs_to_classtargs(arr, id2x0)
    assert tarr[0] == 0
    assert (tarr[1:] == 1).all()  # -3 gets mapped to -2 ID to account for float precision
    assert (arr == np.array([-1.5, -2.0, -3.0])).all()


def test_x0_from_mixture():
    props = x0_from_mixture([43.25 / 1.33, 42.7 / 3.52], [1.33, 3.52], [1, 3])
    assert np.abs(props["X0"] - 17.179) < 1e-3
    assert np.abs(props["density"] - 2.493) < 1e-3
