import pytest
import numpy as np
from pytest_mock import mocker  # noqa F401
import math
import pandas as pd

import torch
from torch import Tensor
import torch.nn.functional as F

from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks.eval_metric import EvalMetric
from tomopt.optimisation.callbacks import NoMoreNaNs, PredHandler, MetricLogger, VoxelMetricLogger, PanelMetricLogger
from tomopt.optimisation.callbacks.diagnostic_callbacks import ScatterRecord, HitRecord
from tomopt.optimisation.loss import DetectorLoss
from tomopt.optimisation.wrapper.volume_wrapper import FitParams
from tomopt.volume import VoxelDetectorLayer, PanelDetectorLayer, DetectorPanel

LW = Tensor([1, 1])
SZ = 0.1
Z = 1


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def area_cost(x: Tensor) -> Tensor:
    return F.relu(x / 1000) ** 2


def check_callback_base(cb: Callback) -> bool:
    assert cb.wrapper is None
    with pytest.raises(AttributeError):
        cb.on_pred_begin()
    with pytest.raises(AttributeError):
        cb.on_train_begin()
    cb.set_wrapper(1)
    assert cb.wrapper == 1
    return True


def get_voxel_detector() -> VoxelDetectorLayer:
    return VoxelDetectorLayer("above", init_res=1, init_eff=1, lw=LW, z=1, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)


def get_panel_detector() -> VoxelDetectorLayer:
    return PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[DetectorPanel(res=1, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[0.5, 0.5], area_cost_func=area_cost)],
    )


class MockWrapper:
    pass


class MockVolume:
    pass


class MockLayer:
    pass


class MockScatterBatch:
    def __init__(self, n) -> None:
        self.n = n

    def get_scatter_mask(self):
        return torch.ones(self.n) > 0


def test_no_more_nans_voxel():
    cb = NoMoreNaNs()
    assert check_callback_base(cb)

    l = get_voxel_detector()
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


def test_no_more_nans_panel():
    cb = NoMoreNaNs()
    assert check_callback_base(cb)

    l = get_panel_detector()
    p = l.panels[0]
    p.xy.grad = p.xy.data
    p.z.grad = p.z.data
    p.xy_span.grad = p.xy_span.data
    p.xy.grad[:1] = Tensor([np.nan])
    p.z.grad[:1] = Tensor([np.nan])
    p.xy_span.grad[:1] = Tensor([np.nan])
    assert p.xy.grad.sum() != p.xy.grad.sum()
    assert p.z.grad.sum() != p.z.grad.sum()
    assert p.xy_span.grad.sum() != p.xy_span.grad.sum()

    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.get_detectors = lambda: [l]
    cb.set_wrapper(vw)
    cb.on_backwards_end()
    assert p.xy.grad.sum() == p.xy.grad.sum()
    assert p.z.grad.sum() == p.z.grad.sum()
    assert p.xy_span.grad.sum() == p.xy_span.grad.sum()


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


@pytest.mark.parametrize("detector", ["none", "voxel", "panel"])
def test_metric_logger(detector, mocker):  # noqa F811
    vw = MockWrapper()
    vw.volume = MockVolume()
    if detector == "none":
        logger = MetricLogger()
    if detector == "voxel":
        logger = VoxelMetricLogger()
        det = get_voxel_detector()
        vw.get_detectors = lambda: [det]
    if detector == "panel":
        logger = PanelMetricLogger()
        det = get_panel_detector()
        vw.get_detectors = lambda: [det]
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

    logger.on_train_end()
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


def test_scatter_record():
    sr = ScatterRecord()
    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.h = 1
    vw.volume.get_passive_z_range = lambda: (0.2, 0.8)
    vw.volume.get_passives = lambda: range(6)
    locs = torch.rand(10, 3)
    vw.fit_params = FitParams(sb=MockScatterBatch(5))
    sr.set_wrapper(vw)

    vw.fit_params.sb.location = locs[:5]
    sr.on_scatter_end()
    assert len(sr.record) == 1
    assert len(sr.record[0]) == 5
    vw.fit_params.sb.location = locs[5:]
    sr.on_scatter_end()
    assert len(sr.record) == 2
    assert len(sr.record[1]) == 5
    assert torch.all(sr.get_record() == locs)

    sr._reset()
    assert len(sr.record) == 0

    sr.record = [Tensor([[0.2, 0.2, 0.71], [0.2, 0.2, 0.21]])]
    df = sr.get_record(True)
    assert isinstance(df, pd.DataFrame)
    assert df.values.shape == (2, 4)
    assert np.all(df.z.values == np.array([0.71, 0.21], dtype="float32"))
    assert np.all(df.layer.values == np.array([0, 5]))


def test_hit_record():
    hr = HitRecord()
    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.h = 1
    xa0 = torch.rand(10, 3)
    xa1 = torch.rand(10, 3)
    xb0 = torch.rand(10, 3)
    xb1 = torch.rand(10, 3)
    vw.fit_params = FitParams(sb=MockScatterBatch(5))
    hr.set_wrapper(vw)

    vw.fit_params.sb.above_hits = [xa0[:5], xa1[:5]]
    vw.fit_params.sb.below_hits = [xb0[:5], xb1[:5]]
    hr.on_scatter_end()
    vw.fit_params.sb.above_hits = [xa0[5:], xa1[5:]]
    vw.fit_params.sb.below_hits = [xb0[5:], xb1[5:]]
    hr.on_scatter_end()

    assert len(hr.record) == 2
    assert hr.record[1].shape == torch.Size([5, 4, 3])
    assert torch.all(hr.get_record() == torch.stack([xa0, xa1, xb0, xb1], dim=1))

    hr.record = [Tensor([[0.0, 0.0, 0.95], [0.1, 0.1, 0.85], [0.2, 0.2, 0.15], [0.3, 0.3, 0.05]])]
    df = hr.get_record(True)
    assert isinstance(df, pd.DataFrame)
    assert df.values.shape == (4, 4)
    assert np.all(df[["x", "y", "z"]].values == hr.record[0].numpy().reshape(-1, 3))
    assert np.all(df.layer.values == np.array([0, 1, 2, 3]))
