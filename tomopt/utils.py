from typing import Dict, List, Optional, Union

import numpy as np
import torch
from functorch import vmap
from torch import Tensor

r"""
Common utility functions
"""

__all__ = ["jacobian", "class_to_x0preds", "x0targs_to_classtargs", "x0_from_mixture"]


def jacobian(y: Tensor, x: Tensor, create_graph: bool = False, allow_unused: bool = True) -> Tensor:
    r"""
    Computes the Jacobian (dy/dx) of y with respect to variables x. x and y can have multiple elements.
    If y has multiple elements then computation is vectorised via vmap.

    Arguments:
        y: tensor to be differentiated
        x: dependent variables
        create_graph: If True, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: False.
        allow_unused: If False, specifying inputs that were not
            used when computing outputs (and therefore their grad is always

    Returns:
        dy/dx tensor of shape y.shape+x.shape
    """

    if len(y) == 0:
        return None
    flat_y = y.reshape(-1)

    def get_vjp(v: Tensor) -> Tensor:
        return torch.autograd.grad(flat_y, x, v, retain_graph=True, create_graph=create_graph, allow_unused=allow_unused)[0].reshape(x.shape)

    return vmap(get_vjp)(torch.eye(len(flat_y), device=y.device)).reshape(y.shape + x.shape)


def class_to_x0preds(array: np.ndarray, id2x0: Dict[int, float]) -> np.ndarray:
    r"""
    Converts array of classes to X0 predictions using the map defined in id2x0

    Arguments:
        array: array of integer class IDs
        id2x0: map of class IDs to X0 float values

    Returns:
        New array of X0 values
    """

    x0array = np.zeros_like(array, dtype="float32")
    for i in np.unique(array):
        x0array[array == i] = id2x0[i]
    return x0array


def x0targs_to_classtargs(array: np.ndarray, x02id: Dict[float, int]) -> np.ndarray:
    r"""
    Converts array of float X0 targets to integer class IDs using the map defined in x02id.

    .. warning::
        To account for floating point precision, X0 values are mapped to the integer class IDs which are closest to X0 keys defined in the map.
        This means that the method cannot detect missing X0 values from x02id;
        X0s will always be mapped to a class ID, even if the material isn't defined in the map

    .. warning::
        The input array is modified in-place

    Arguments:
        array: array of X0 values
        x02id: map of X0 float values to class IDs

    Returns:
        Array of integer class IDs
    """

    x0array = np.zeros_like(array)
    for i in np.unique(array):
        x0array[array == i] = x02id[min(x02id, key=lambda x: abs(x - i))]
    return x0array


def x0_from_mixture(
    x0s: Union[np.ndarray, List[float]],
    densities: Union[np.ndarray, List[float]],
    weight_fracs: Optional[Union[np.ndarray, List[float]]] = None,
    volume_fracs: Optional[Union[np.ndarray, List[float]]] = None,
) -> Dict[str, float]:
    r"""
    Computes the X0 of a mixture of (non-chemically bonded) materials,
    Based on https://cds.cern.ch/record/1279627/files/PH-EP-Tech-Note-2010-013.pdf

    Arguments:
        x0s: X0 values of the materials in the mixture in metres
        densities: densities of the materials in the mixture kg/m^3
        weight_fracs: the relative amounts of each material by weight
        volume_fracs: the relative amounts of each material by volume

    Returns:
        {"X0":The X0 of the defined mixture in metres, "density": The density in kg/m^3 of the defined mixture}
    """

    if weight_fracs is None and volume_fracs is None:
        raise ValueError("Must pass the fractional composition by either weight or volume")
    if weight_fracs is not None and volume_fracs is not None:
        raise ValueError("Cannot pass both weight and volume fractions")
    if not isinstance(x0s, np.ndarray):
        x0s = np.array(x0s)
    if not isinstance(densities, np.ndarray):
        densities = np.array(densities)

    if weight_fracs is None:
        if not isinstance(volume_fracs, np.ndarray):
            volume_fracs = np.array(volume_fracs)
        weight_fracs = densities * volume_fracs

    if not isinstance(weight_fracs, np.ndarray):
        weight_fracs = np.array(weight_fracs)
    weight_fracs = weight_fracs / weight_fracs.sum()

    x0rho = 1 / (weight_fracs / (x0s * densities)).sum()
    rho = 1 / (weight_fracs / densities).sum()
    x0 = x0rho / rho
    return {"X0": x0, "density": rho}
