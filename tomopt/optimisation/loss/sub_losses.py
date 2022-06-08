from typing import Optional

import torch
from torch import Tensor

__all__ = ["integer_class_loss"]


def integer_class_loss(
    int_probs: Tensor,
    target_int: Tensor,
    pred_start_int: int,
    use_mse: bool,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    ints = torch.arange(pred_start_int, pred_start_int + int_probs.size(1))
    diffs = target_int - ints
    if use_mse:
        diffs = diffs**2
    else:
        diffs = diffs.abs()
    diffs = torch.softmax(diffs, dim=-1)
    loss = diffs * int_probs
    loss = loss.sum(-1)

    if weight is not None:
        loss = loss * weight

    if reduction == "mean":
        return loss.mean(-1)
    elif reduction == "sum":
        return loss.sum(-1)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction {reduction}. Please use ['mean', 'sum', 'none'].")
