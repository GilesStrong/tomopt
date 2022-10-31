from typing import Optional

import torch
from torch import Tensor

r"""
Provides functions to compute sub-loss components
"""

__all__ = ["integer_class_loss"]


def integer_class_loss(
    int_probs: Tensor,
    target_int: Tensor,
    pred_start_int: int,
    use_mse: bool,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""
    Loss for classifying integers, when regression is not applicable.
    It assumed that the the integers really are quantifiably comparable, and not categorical codes of classes.

    Like multiclass-classification, predictions are a probabilities for each possible integer,
    but the ICL aims to penalise close predictions less than far-off ones:
    For a target of 3 and a close prediction of `softmax([1,3,10,5,5,3,1])` and a far-off prediction of `softmax([10,3,1,5,5,3,1])`,
    the categorical cross-entropy produces the same loss for both predictions (5.0154) despite the close prediction having a higher probability near the target.

    ICL instead computes the absolute error, or squared error, between each of the possible integers and the true target.
    These errors are then normalised, weighted by the predicted probabilities, and summed.
    I.e. integers close to the target have a lower error, and these are given greater weight in the sum if they have a higher probability.

    For the example, the ICL produces a loss of 1.0007 for the close prediction, and 8.8773 for the far-off one.

    Arguments:
        int_probs: (*,integers) tensor of predicted probabilities
        target_int: (*) tensor of target integers
        pred_start_int: the integer that the zeroth probability in predictions corresponds to
        use_mse: whether to compute errors as absolute or squared
        weight: Optional (*) tensor of multiplicative weights for the unreduced ICLs
        reduction: 'mean' return the average ICL, 'sum' sum the ICLs, 'none', return the individual ICLs
    """

    ints = torch.arange(pred_start_int, pred_start_int + int_probs.size(-1))
    diffs = target_int - ints
    if use_mse:
        diffs = diffs**2
    else:
        diffs = diffs.abs()
    diffs = diffs / diffs.sum(-1, keepdim=True)
    loss = diffs * int_probs
    loss = loss.sum(-1, keepdim=True)

    if weight is not None:
        loss = loss * weight

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction {reduction}. Please use ['mean', 'sum', 'none'].")
