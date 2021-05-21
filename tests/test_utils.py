import torch
from torch import Tensor

from tomopt.utils import jacobian


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
