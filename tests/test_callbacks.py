import pytest
import numpy as np

import torch
from torch import Tensor

from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks import NoMoreNaNs, PredHandler
from tomopt.optimisation.wrapper.volume_wrapper import FitParams


def check_callback_base(cb: Callback) -> bool:
    assert cb.wrapper is None
    with pytest.raises(AttributeError):
        cb.on_pred_begin()
        cb.on_train_begin()
    cb.set_wrapper(1)
    assert cb.wrapper == 1
    return True


class MockWrapper:
    pass


class MockVolume:
    pass


class MockLayer:
    pass


def test_no_more_nans():
    cb = NoMoreNaNs()
    assert check_callback_base(cb)

    l = MockLayer()
    l.resolution = torch.rand(10, 10)
    l.efficiency = torch.rand(10, 10)
    l.resolution.grad = l.resolution.data
    l.efficiency.grad = l.efficiency.data
    l.resolution.grad[:5, :5] = Tensor([np.nan])
    l.efficiency.grad[:5, :5] = Tensor([np.nan])
    assert l.resolution.grad.sum() != l.resolution.grad.sum()
    assert l.efficiency.grad.sum() != l.efficiency.grad.sum()

    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.get_detectors = lambda: [l]
    cb.set_wrapper(vw)
    cb.on_backwards_end()
    assert l.resolution.grad.sum() == l.resolution.grad.sum()
    assert l.efficiency.grad.sum() == l.efficiency.grad.sum()


def test_pred_handler():
    cb = PredHandler()
    assert check_callback_base(cb)

    cb.on_pred_begin()
    assert isinstance(cb.preds, list)
    assert len(cb.preds) == 0

    vw = MockWrapper()
    vw.fit_params = FitParams(state="train", pred=Tensor([1]))
    cb.set_wrapper(vw)

    cb.on_x0_pred_end()
    assert len(cb.preds) == 0
    vw.fit_params.state = "valid"
    cb.on_x0_pred_end()
    assert len(cb.preds) == 0
    vw.fit_params.state = "test"
    cb.on_x0_pred_end()
    assert len(cb.preds) == 1
    vw.fit_params.pred = vw.fit_params.pred + 1
    cb.on_x0_pred_end()
    assert len(cb.preds) == 2
    assert cb.preds[0][0] == 1
    assert cb.preds[1][0] == 2

    preds = cb.get_preds()
    assert preds[0][0] == 1
    assert preds[1][0] == 2
