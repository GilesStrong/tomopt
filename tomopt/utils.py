from distutils.version import LooseVersion

import torch
from torch import Tensor
from torch._vmap_internals import _vmap as vmap

__all__ = ["jacobian"]


if (ver := LooseVersion(torch.__version__).vstring.split("+")[0]) not in ["1.8.1"]:
    raise ImportError(
        f"jacobian function relies on PyTorch vmap, which is experimental and has only been tested for use in this repo using torch==1.8.1, \
          your current version is {ver}, please install 1.8.1 or test in your version and update this error message."
    )


def jacobian(y: Tensor, x: Tensor, create_graph: bool = False, allow_unused: bool = True) -> Tensor:
    if len(y) == 0:
        return None
    flat_y = y.reshape(-1)

    def get_vjp(v: Tensor) -> Tensor:
        return torch.autograd.grad(flat_y, x, v, retain_graph=True, create_graph=create_graph, allow_unused=allow_unused)[0].reshape(x.shape)

    return vmap(get_vjp)(torch.eye(len(flat_y))).reshape(y.shape + x.shape)
