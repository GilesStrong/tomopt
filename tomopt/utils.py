from distutils.version import LooseVersion
import numpy as np
from typing import Dict, Union, List

import torch
from torch import Tensor
from torch._vmap_internals import _vmap as vmap

__all__ = ["jacobian", "class_to_x0preds", "x0targs_to_classtargs", "x0_from_mixture"]


if (ver := LooseVersion(torch.__version__)) < LooseVersion("1.10.0"):
    raise ImportError(
        f"jacobian function relies on PyTorch vmap, which is experimental and has only been tested for use in this repo using torch 1.8.1 & 1.10.0\
          your current version is {ver}, please install 1.10.0 or test in your version and update this error message."
    )


def jacobian(y: Tensor, x: Tensor, create_graph: bool = False, allow_unused: bool = True) -> Tensor:
    if len(y) == 0:
        return None
    flat_y = y.reshape(-1)

    def get_vjp(v: Tensor) -> Tensor:
        return torch.autograd.grad(flat_y, x, v, retain_graph=True, create_graph=create_graph, allow_unused=allow_unused)[0].reshape(x.shape)

    return vmap(get_vjp)(torch.eye(len(flat_y), device=y.device)).reshape(y.shape + x.shape)


def class_to_x0preds(array: np.ndarray, id2x0: Dict[int, float]) -> np.ndarray:
    x0array = np.zeros_like(array, dtype="float32")
    for i in np.unique(array):
        x0array[array == i] = id2x0[i]
    return x0array


def x0targs_to_classtargs(array: np.ndarray, x02id: Dict[float, int]) -> np.ndarray:
    x0array = np.zeros_like(array)
    for i in np.unique(array):
        x0array[array == i] = x02id[min(x02id, key=lambda x: abs(x - i))]
    return x0array


def x0_from_mixture(x0s: Union[np.ndarray, List[float]], densities: Union[np.ndarray, List[float]], fracs: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    if not isinstance(x0s, np.ndarray):
        x0s = np.array(x0s)
    if not isinstance(densities, np.ndarray):
        densities = np.array(densities)
    if not isinstance(fracs, np.ndarray):
        fracs = np.array(fracs)
    fracs = fracs / fracs.sum()

    x0rho = 1 / (fracs / (x0s * densities)).sum()
    rho = 1 / (fracs / densities).sum()
    x0 = x0rho / rho
    return {"X0": x0, "density": rho}
