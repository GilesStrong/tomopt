import torch
from torch import Tensor

__all__ = ["integer_class_loss"]


def integer_class_loss(int_probs: Tensor, target_int: Tensor, pred_start_int: int) -> Tensor:
    ints = torch.arange(pred_start_int, pred_start_int + int_probs.size(1))
    diffs = torch.softmax((target_int - ints).abs(), dim=-1)
    loss = diffs * int_probs
    return loss.sum(-1).mean()
