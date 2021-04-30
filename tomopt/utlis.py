import torch
from torch import Tensor

__all__ = ["jacobian"]


def jacobian(y: Tensor, x: Tensor, create_graph=False, allow_unused=True):
    r"""Compute full jacobian matrix for single tensor. Call twice for hessian.
    Copied from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7 credits: Adam Paszke
    TODO: Fix this to work batch-wise (maybe https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa)"""
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0  # may need to use value of flat_y?
        (grad_x,) = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph, allow_unused=allow_unused)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + x.shape)
