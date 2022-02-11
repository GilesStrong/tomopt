import pytest
import numpy as np
from pytest_mock import mocker  # noqa F401
import math
import pandas as pd
from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F

from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks.eval_metric import EvalMetric
from tomopt.optimisation.callbacks import (
    NoMoreNaNs,
    PredHandler,
    MetricLogger,
    VoxelMetricLogger,
    PanelMetricLogger,
    ScatterRecord,
    HitRecord,
    CostCoefWarmup,
    PanelOptConfig,
)
from tomopt.optimisation.loss import VoxelX0Loss
from tomopt.optimisation.wrapper.volume_wrapper import AbsVolumeWrapper, FitParams, PanelVolumeWrapper
from tomopt.volume import VoxelDetectorLayer, PanelDetectorLayer, DetectorPanel
from tomopt.volume.volume import Volume

LW = Tensor([1, 1])
SZ = 0.1
Z = 1


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def area_cost(a: Tensor) -> Tensor:
    return F.relu(a)


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
    device = torch.device("cpu")


class MockLayer:
    pass


class MockOpt:
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
    vw.loss_func = VoxelX0Loss(target_budget=1, cost_coef=1)
    vw.fit_params = FitParams(pred=10, trn_passives=range(10), passive_bs=2, metric_cbs=[EvalMetric(name="test", main_metric=True, lower_metric_better=True)])
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
    vw.volume.h = Tensor([1])
    vw.volume.get_passive_z_range = lambda: Tensor([0.2, 0.8])
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
    vw.volume.h = Tensor([1])
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


def test_cost_coef_warmup():
    class VW(AbsVolumeWrapper):
        def _build_opt(self, **kwargs) -> None:
            pass

    loss = VoxelX0Loss(target_budget=0.8)
    vol = MockVolume()
    vol.parameters = []
    vw = VW(volume=vol, partial_opts={}, loss_func=loss, partial_scatter_inferer=None, partial_volume_inferer=None)
    vw.fit_params = FitParams(pred=10)
    opt = MockOpt()
    opt.param_groups = [{"lr": 1e2}]
    vw.opts = {"mock_opt": opt}
    ccw = CostCoefWarmup(5)
    ccw.set_wrapper(vw)

    ccw.on_train_begin()
    assert opt.param_groups[0]["lr"] == 0.0
    for e in range(6):
        for s in range(2):  # Training & validation
            vw.fit_params.state = "train" if s == 0 else "valid"
            ccw.on_epoch_begin()
            for v in range(3):
                print(e, s, v)
                loss.sub_losses["error"] = Tensor([((-1) ** s) * (2 ** e) * (3 ** v)])  # Unique value per epoch per volume
                ccw.on_volume_end()
            if e < 5:
                if s == 0:
                    assert ccw.v_sum.item() == ((2 ** e)) + ((2 ** e) * 3) + ((2 ** e) * 9)
                    assert ccw.volume_cnt == 3
                else:
                    assert ccw.v_sum.item() == 0.0
                    assert ccw.volume_cnt == 0
            else:  # Tracking stopped
                assert ccw.v_sum.item() == 0.0
                assert ccw.volume_cnt == 0
            ccw.on_epoch_end()
            if e < 5:
                assert np.abs(ccw.e_sum.item() - np.sum([((2 ** i)) + ((2 ** i) * 3) + ((2 ** i) * 9) for i in range(0, e + 1)]) / 3) < 1e-4
                assert ccw.epoch_cnt == e + 1
            else:  # warm-up finished
                assert ccw.tracking is False
                assert opt.param_groups[0]["lr"] == 1e2
                assert np.abs(ccw.e_sum.item() - (sum := np.sum([((2 ** i)) + ((2 ** i) * 3) + ((2 ** i) * 9) for i in range(0, 5)]) / 3)) < 1e-4
                assert ccw.epoch_cnt == 5
                assert np.abs(loss.cost_coef.item() - sum / 5) < 1e-4


def test_panel_opt_config():
    volume = MockVolume()
    volume.parameters = []
    panel_det = get_panel_detector()
    volume.get_detectors = lambda: [panel_det]
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(torch.optim.SGD, lr=5e4),
        z_pos_opt=partial(torch.optim.SGD, lr=5e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e4),
        loss_func=VoxelX0Loss(target_budget=0),
    )
    vw.fit_params = FitParams()
    xy_pos_rate = 0.1
    z_pos_rate = 0.05
    xy_span_rate = 0.2
    xy_pos_mult = 1
    z_pos_mult = -3
    xy_span_mult = 2
    poc = PanelOptConfig(n_warmup=2, xy_pos_rate=xy_pos_rate, z_pos_rate=z_pos_rate, xy_span_rate=xy_span_rate)
    poc.set_wrapper(vw)

    poc.on_train_begin()
    for o in ["xy_pos_opt", "z_pos_opt", "xy_span_opt"]:
        assert vw.get_opt_lr(o) == 0.0
    for e in range(3):
        if e < 2:
            assert poc.tracking is True
        for s in range(2):  # Training & validation
            vw.fit_params.state = "train" if s == 0 else "valid"
            for p in panel_det.panels:
                p.xy.grad = xy_pos_mult * torch.ones(2) * e * (1 + s)
                p.z.grad = z_pos_mult * torch.ones(1) * e * (1 + s)
                p.xy_span.grad = xy_span_mult * torch.ones(2) * e * (1 + s)
            poc.on_backwards_end()
            if e < 2:
                assert len(poc.stats["xy_pos_opt"]) == e + 1
                assert len(poc.stats["z_pos_opt"]) == e + 1
                assert len(poc.stats["xy_span_opt"]) == e + 1
                assert poc.stats["xy_pos_opt"][-1].mean() == xy_pos_mult * e
                assert poc.stats["z_pos_opt"][-1].mean() == z_pos_mult * e
                assert poc.stats["xy_span_opt"][-1].mean() == xy_span_mult * e
            poc.on_epoch_end()
            if e >= 1:
                assert poc.tracking is False
                assert vw.get_opt_lr("xy_pos_opt") == xy_pos_rate / (xy_pos_mult / 2)
                assert vw.get_opt_lr("z_pos_opt") == np.abs(z_pos_rate / (z_pos_mult / 2))
                assert vw.get_opt_lr("xy_span_opt") == xy_span_rate / (xy_span_mult / 2)
