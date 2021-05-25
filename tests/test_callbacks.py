import pytest
import numpy as np
from pytest_mock import mocker  # noqa F401
import math

import torch
from torch import Tensor

from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks.eval_metric import EvalMetric
from tomopt.optimisation.callbacks import NoMoreNaNs, PredHandler, MetricLogger
from tomopt.optimisation.loss import DetectorLoss
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
    vw.volume = MockVolume()
    vw.volume.get_rad_cube = lambda: Tensor([3])

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
    assert preds[0][0][0] == 1
    assert preds[1][0][0] == 2
    assert preds[0][1][0] == 3
    assert preds[1][1][0] == 3


def test_metric_logger(mocker):  # noqa F811
    logger = MetricLogger()
    vw = MockWrapper()
    vw.loss_func = DetectorLoss(1)
    vw.fit_params = FitParams(trn_passives=range(10), passive_bs=2, metric_cbs=[EvalMetric(name="test", main_metric=True, lower_metric_better=True)])
    logger.set_wrapper(vw)
    mocker.spy(logger, "_reset")

    logger.on_train_begin()
    assert logger._reset.call_count == 1
    assert logger.best_loss == math.inf
    assert len(logger.loss_vals["Training"]) == 0
    assert len(logger.loss_vals["Validation"]) == 0
    assert logger.n_trn_batches == 5
    assert logger.lock_to_metric is True
    assert logger.main_metric_idx == 0
    assert len(logger.metric_vals) == 1
    assert len(logger.metric_vals[0]) == 0

    logger.on_epoch_begin()
    assert logger.tmp_loss == 0
    assert logger.batch_cnt == 0
    assert logger.volume_cnt == 0
    assert len(logger.tmp_sub_losses.keys()) == 0

    for state in ["train", "valid"]:
        vw.fit_params.state = state
        for i in range(5):  # one epoch
            vw.fit_params.mean_loss = Tensor([i])
            logger.on_backwards_end()
        assert logger.loss_vals["Training"] == (trn_losses := list(range(5)))

    for state in ["valid", "train"]:
        vw.fit_params.state = state
        for i in range(10):  # one epoch
            vw.loss_func.sub_losses = {"error": Tensor([i]), "cost": Tensor([-i])}
            logger.on_volume_end()
        assert logger.tmp_sub_losses["error"] == np.sum(range(10))
        assert logger.tmp_sub_losses["cost"] == -np.sum(range(10))
        assert logger.volume_cnt == 10

    for state in ["valid", "train"]:
        vw.fit_params.state = state
        for i in range(5):  # one epoch
            vw.loss_func.mean_loss = Tensor([2 * i])
            logger.on_volume_batch_end()
        assert logger.tmp_loss == (val_loss := np.sum(2 * np.arange(5)))
        assert logger.batch_cnt == 5

    vw.fit_params.metric_cbs[0].metric = 3
    vw.fit_params.state = "train"
    logger.on_epoch_end()
    assert logger.val_epoch_results is None
    vw.fit_params.state = "valid"
    logger.on_epoch_end()
    assert logger.loss_vals["Validation"] == [(val_loss := val_loss / 5)]
    assert len(logger.sub_losses.keys()) == 2
    assert len(logger.sub_losses["error"]) == 1
    assert logger.sub_losses["error"][0] + logger.sub_losses["cost"][0] - 1 < 1e-5
    assert len(logger.metric_vals[0]) == 1
    assert logger.metric_vals[0][0] == 3
    assert logger.best_loss == val_loss
    assert logger.val_epoch_results == (val_loss, 3)

    history = logger.get_loss_history()
    assert history[0]["Training"] == trn_losses
    assert history[0]["Validation"] == [val_loss]
    assert history[1]["test"] == [3]

    logger.loss_vals["Validation"] = [9, 8, 7, 6, 5, 9]
    logger.metric_vals = [[10, 3, 5, 6, 7, 5]]
    results = logger.get_results(loaded_best=True)
    assert results["loss"] == 8
    assert results["test"] == 3
    results = logger.get_results(loaded_best=False)
    assert results["loss"] == 9
    assert results["test"] == 5
    logger.lock_to_metric = False
    results = logger.get_results(loaded_best=True)
    assert results["loss"] == 5
    assert results["test"] == 7
