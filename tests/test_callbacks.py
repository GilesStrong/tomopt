import pytest
import numpy as np
from pytest_mock import mocker  # noqa F401
import math
import pandas as pd
from functools import partial
from glob import glob
from fastcore.all import Path
import matplotlib.pyplot as plt
import h5py
import os

import torch
from torch import Tensor
import torch.nn.functional as F

from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks.eval_metric import EvalMetric
from tomopt.optimisation.callbacks import (
    NoMoreNaNs,
    PredHandler,
    MetricLogger,
    PanelMetricLogger,
    ScatterRecord,
    HitRecord,
    CostCoefWarmup,
    PanelOptConfig,
    MuonResampler,
    HeatMapGif,
    VolumeTargetPredHandler,
    Save2HDF5PredHandler,
    WarmupCallback,
    PostWarmupCallback,
    SigmoidPanelSmoothnessSchedule,
)
from tomopt.optimisation.loss import VoxelX0Loss
from tomopt.optimisation.wrapper.volume_wrapper import AbsVolumeWrapper, FitParams, PanelVolumeWrapper
from tomopt.volume import PanelDetectorLayer, DetectorPanel, DetectorHeatMap, SigmoidDetectorPanel
from tomopt.muon import MuonBatch, MuonGenerator2016

LW = Tensor([1, 1])
SZ = 0.1
Z = 1
PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def check_callback_base(cb: Callback) -> bool:
    assert cb.wrapper is None
    with pytest.raises(AttributeError):
        cb.on_pred_begin()
    with pytest.raises(AttributeError):
        cb.on_train_begin()
    cb.set_wrapper(1)
    assert cb.wrapper == 1
    return True


def get_panel_detector() -> PanelDetectorLayer:
    return PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[DetectorPanel(res=1, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[1.0, 1.0])],
    )


def get_sigmoid_panel_detector(smooth: Tensor = Tensor([0.5])) -> PanelDetectorLayer:
    return PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[SigmoidDetectorPanel(smooth=smooth, res=1, eff=1, init_xyz=[0.5, 0.5, 0.9], init_xy_span=[1.0, 1.0])],
    )


class MockWrapper:
    pass


class MockVolume:
    device = torch.device("cpu")
    lw = LW
    h = Tensor([Z])
    passive_size = SZ
    budget_weights = torch.zeros(3, requires_grad=True)

    def get_passive_z_range(self) -> Tensor:
        return Tensor([0.2, 0.8])


class MockLayer:
    pass


class MockOpt:
    pass


class MockScatterBatch:
    def __init__(self, n) -> None:
        self.n = n

    def get_scatter_mask(self):
        return torch.ones(self.n) > 0


def test_no_more_nans_panel():
    cb = NoMoreNaNs()
    assert check_callback_base(cb)

    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.budget_weights.grad = Tensor([np.nan, np.nan, np.nan])
    vw.volume.get_detectors = lambda: [l]

    l = get_panel_detector()
    p = l.panels[0]
    p.xy.grad = p.xy.data
    p.z.grad = p.z.data
    p.xy_span.grad = p.xy_span.data
    p.xy.grad[:1] = Tensor([np.nan])
    p.z.grad[:1] = Tensor([np.nan])
    p.xy_span.grad[:1] = Tensor([np.nan])
    assert p.xy.grad.sum().isnan()
    assert p.z.grad.sum().isnan()
    assert p.xy_span.grad.sum().isnan()
    assert vw.volume.budget_weights.grad.sum().isnan()

    cb.set_wrapper(vw)
    cb.on_backwards_end()
    assert not p.xy.grad.sum().isnan()
    assert not p.z.grad.sum().isnan()
    assert not p.xy_span.grad.sum().isnan()
    assert not vw.volume.budget_weights.grad.sum().isnan()


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


def test_x0_class_pred_handler():
    cb = VolumeTargetPredHandler({0.5: 0, 1.5: 1, 3.0: 2})
    assert check_callback_base(cb)

    cb.on_pred_begin()
    assert isinstance(cb.preds, list)
    assert len(cb.preds) == 0

    vw = MockWrapper()
    vw.fit_params = FitParams(state="train", pred=Tensor([1]))
    cb.set_wrapper(vw)
    vw.volume = MockVolume()
    vw.volume.target = Tensor([0.5, 0.5, 3.0])

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
    assert (preds[0][1][:2] == 0).all()
    assert preds[0][1][2] == 2


def test_float_pred_handler():
    cb = VolumeTargetPredHandler()
    assert check_callback_base(cb)

    cb.on_pred_begin()
    assert isinstance(cb.preds, list)
    assert len(cb.preds) == 0

    vw = MockWrapper()
    vw.fit_params = FitParams(state="train", pred=Tensor([1]))
    cb.set_wrapper(vw)
    vw.volume = MockVolume()
    vw.volume.target = Tensor([0.5])

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
    assert preds[0][1] == 0.5


@pytest.mark.parametrize("detector", ["none", "panel", "sigmoid_panel"])
def test_metric_logger(detector, mocker):  # noqa F811
    vw = MockWrapper()
    vw.volume = MockVolume()
    if detector == "none":
        logger = MetricLogger()
    else:
        if detector == "panel":
            det = get_panel_detector()
        elif detector == "sigmoid_panel":
            det = get_sigmoid_panel_detector()
        else:
            raise ValueError(f"Detector {detector} not recognised")
        logger = PanelMetricLogger()
        vw.volume.get_detectors = lambda: [det]
        vw.get_detector = lambda: vw.volume.get_detectors()

    vw.loss_func = VoxelX0Loss(target_budget=1, cost_coef=1)
    vw.fit_params = FitParams(
        pred=10,
        trn_passives=range(10),
        passive_bs=2,
        state="train",
        metric_cbs=[EvalMetric(name="test", main_metric=True, lower_metric_better=True)],
        cb_savepath=Path(PKG_DIR),
    )
    logger.set_wrapper(vw)
    mocker.spy(logger, "_reset")
    mocker.spy(logger, "_snapshot_monitor")
    assert logger.gif_filename == "optimisation_history.gif"
    logger.fig = plt.figure(figsize=(5, 5), constrained_layout=True)  # Hack to make a figure

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

    logger.show_plots = True
    logger.on_epoch_begin()
    logger.show_plots = False
    assert logger.tmp_loss == 0
    assert logger.batch_cnt == 0
    assert logger.volume_cnt == 0
    assert len(logger.tmp_sub_losses.keys()) == 0
    assert logger._snapshot_monitor.call_count == 1
    assert len(logger._buffer_files) == 1
    assert logger._buffer_files[-1] == Path(PKG_DIR / "temp_monitor_0.png")
    assert logger._buffer_files[-1].exists()

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
    assert len(logger._buffer_files) == 1
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
    assert len(logger._buffer_files) == 1

    logger.show_plots = True
    logger.on_train_end()
    logger.show_plots = False
    history = logger.get_loss_history()
    assert history[0]["Training"] == trn_losses
    assert history[0]["Validation"] == [val_loss]
    assert history[1]["test"] == [3]
    assert len(logger._buffer_files) == 2
    assert logger._buffer_files[-1] == Path(PKG_DIR / "temp_monitor_1.png")
    for f in logger._buffer_files:
        assert not f.exists()
    assert Path(PKG_DIR / "optimisation_history.gif").exists()

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
    assert check_callback_base(sr)
    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.h = Tensor([1])
    vw.volume.get_passive_z_range = lambda: Tensor([0.2, 0.8])
    vw.volume.get_passives = lambda: range(6)
    locs = torch.rand(10, 3)
    vw.fit_params = FitParams(sb=MockScatterBatch(5))
    sr.set_wrapper(vw)

    vw.fit_params.sb.poca_xyz = locs[:5]
    sr.on_scatter_end()
    assert len(sr.record) == 1
    assert len(sr.record[0]) == 5
    vw.fit_params.sb.poca_xyz = locs[5:]
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
    assert check_callback_base(hr)
    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.h = Tensor([1])
    hits = torch.rand(10, 4, 3)
    vw.fit_params = FitParams(sb=MockScatterBatch(5))
    hr.set_wrapper(vw)

    vw.fit_params.sb._reco_hits = hits[:5]
    hr.on_scatter_end()
    vw.fit_params.sb._reco_hits = hits[5:]
    hr.on_scatter_end()

    assert len(hr.record) == 2
    assert hr.record[1].shape == torch.Size([5, 4, 3])
    print(hr.get_record().shape, hits.shape)
    assert torch.all(hr.get_record() == hits)

    hr.record = [Tensor([[0.0, 0.0, 0.95], [0.1, 0.1, 0.85], [0.2, 0.2, 0.15], [0.3, 0.3, 0.05]])]
    df = hr.get_record(True)
    assert isinstance(df, pd.DataFrame)
    assert df.values.shape == (4, 4)
    assert np.all(df[["x", "y", "z"]].values == hr.record[0].numpy().reshape(-1, 3))
    assert np.all(df.layer.values == np.array([0, 1, 2, 3]))


def test_warmup_callback():
    vw = MockWrapper()
    wc1 = WarmupCallback(1)
    assert check_callback_base(wc1)
    wc2 = WarmupCallback(2)
    vw.fit_params = FitParams(warmup_cbs=[wc1, wc2], state="train")
    wc1.set_wrapper(vw)
    wc2.set_wrapper(vw)

    # CBs initialise ok
    assert wc1.n_warmup == 1
    assert wc2.n_warmup == 2
    wc1.on_train_begin()
    wc2.on_train_begin()
    assert wc1.epoch_cnt == 0
    assert wc1.warmup_active
    assert not wc2.warmup_active
    assert vw.fit_params.skip_opt_step

    # wc1 begins warmup first
    wc1.on_epoch_begin()
    wc2.on_epoch_begin()
    assert wc1.warmup_active
    assert not wc2.warmup_active

    # wc1 only updates on training epochs and wc2 doesn't update
    vw.fit_params.state = "valid"
    wc1.on_epoch_end()
    assert wc1.epoch_cnt == 0
    vw.fit_params.state = "train"
    wc2.on_epoch_end()
    assert wc2.epoch_cnt == 0

    # wc1 competes warmup
    wc1.on_epoch_end()
    assert wc1.epoch_cnt == 1
    assert wc1.warmup_active

    # wc1 deactivates and wc2 begins warmup
    wc1.on_epoch_begin()
    assert not wc1.warmup_active
    assert wc2.warmup_active
    wc2.on_epoch_end()
    assert wc2.epoch_cnt == 1
    wc1.on_epoch_begin()
    assert wc2.warmup_active
    wc2.on_epoch_end()
    assert wc2.epoch_cnt == 2
    assert vw.fit_params.skip_opt_step
    assert wc2.warmup_active

    # All warmups complete, fitting begins
    wc1.on_epoch_begin()
    assert not wc2.warmup_active
    assert not vw.fit_params.skip_opt_step


@pytest.mark.parametrize("n_warmups", [0, 1, 2])
def test_post_warmup_callback(n_warmups, mocker):  # noqa F811
    vw = MockWrapper()
    pwcb = PostWarmupCallback()
    assert check_callback_base(pwcb)
    wcbs = [WarmupCallback(i + 1) for i in range(n_warmups)]
    cbs = wcbs + [pwcb]

    vw.fit_params = FitParams(cbs=cbs, warmup_cbs=wcbs, state="train")
    for cb in vw.fit_params.cbs:
        cb.set_wrapper(vw)
    mocker.spy(pwcb, "_activate")

    for cb in vw.fit_params.cbs:
        cb.on_train_begin()
    assert pwcb.active is False
    n_warmup_epochs = np.arange(1, n_warmups + 1).sum()
    print(f"{n_warmup_epochs=}")
    for epoch in range(1, n_warmup_epochs + 3):
        vw.fit_params.epoch = epoch
        for c in vw.fit_params.cbs:
            c.on_epoch_begin()
        print(f"{epoch=}")
        if n_warmup_epochs > 0 and epoch < n_warmup_epochs + 1:
            for c in vw.fit_params.warmup_cbs:
                print(c.warmup_active)
            assert pwcb.active is False
            assert pwcb._activate.call_count == 0
        elif n_warmup_epochs == 0 or epoch >= n_warmup_epochs + 1:
            if n_warmups > 0:
                assert wcbs[-1].warmup_active is False
            assert pwcb.active
            assert pwcb._activate.call_count == 1
        for c in vw.fit_params.cbs:
            c.on_epoch_end()


def test_cost_coef_warmup():
    class VW(AbsVolumeWrapper):
        def _build_opt(self, **kwargs) -> None:
            pass

    loss = VoxelX0Loss(target_budget=0.8)
    vol = MockVolume()
    vol.parameters = []
    vw = VW(volume=vol, partial_opts={}, loss_func=loss, partial_scatter_inferrer=None, partial_volume_inferrer=None)
    vw.fit_params = FitParams(pred=10)
    ccw = CostCoefWarmup(5)
    assert check_callback_base(ccw)
    ccw.set_wrapper(vw)
    vw.fit_params.warmup_cbs = [ccw]

    ccw.on_train_begin()
    for e in range(6):
        for s in range(2):  # Training & validation
            vw.fit_params.state = "train" if s == 0 else "valid"
            ccw.on_epoch_begin()
            for v in range(3):
                print(e, s, v)
                loss.sub_losses["error"] = Tensor([((-1) ** s) * (2**e) * (3**v)])  # Unique value per epoch per volume
                ccw.on_volume_end()
            if e < 5:
                assert np.sum(ccw.errors) == np.sum([((2**i)) + ((2**i) * 3) + ((2**i) * 9) for i in range(e + 1)])
            else:  # Tracking stopped
                assert np.sum(ccw.errors) == np.sum([((2**i)) + ((2**i) * 3) + ((2**i) * 9) for i in range(5)])
            ccw.on_epoch_end()
            if e < 5:
                assert np.abs(np.sum(ccw.errors) - np.sum([((2**i)) + ((2**i) * 3) + ((2**i) * 9) for i in range(e + 1)])) < 1e-4
                assert ccw.epoch_cnt == e + 1
            else:  # warm-up finished
                assert ccw.warmup_active is False
                assert np.abs(np.sum(ccw.errors) - np.sum([((2**i)) + ((2**i) * 3) + ((2**i) * 9) for i in range(5)])) < 1e-4
                assert ccw.epoch_cnt == 5
                assert np.abs(loss.cost_coef - np.median(ccw.errors)) < 1e-4


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
    assert check_callback_base(poc)
    poc.set_wrapper(vw)
    vw.fit_params.warmup_cbs = [poc]

    poc.on_train_begin()
    for e in range(3):
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
                assert poc.stats["xy_pos_opt"][-1].mean() == np.abs(xy_pos_mult * e)
                assert poc.stats["z_pos_opt"][-1].mean() == np.abs(z_pos_mult * e)
                assert poc.stats["xy_span_opt"][-1].mean() == np.abs(xy_span_mult * e)


def test_muon_resampler_callback():
    # Check checker
    volume = MockVolume()
    l = MockLayer()
    l.z = 0.5
    volume.get_passives = lambda: [l]
    mu = MuonBatch(Tensor([[0.5, 0.5, 5, 0, 0], [-0.4, -0.4, 5, np.pi / 4, np.pi / 4], [-1, 1, 5, 0, 0]]), volume.h)
    assert (MuonResampler.check_mu_batch(mu, volume) == Tensor([1, 0, 0]).bool()).all()

    # Check resampler
    gen = MuonGenerator2016.from_volume(volume)
    mus = gen(1000)
    while MuonResampler.check_mu_batch(MuonBatch(mus, volume.h), volume).sum() == 1000:
        mus = gen(1000)
    mus = MuonResampler.resample(mus, volume=volume, gen=gen)
    mu = MuonBatch(mus, volume.h)
    assert MuonResampler.check_mu_batch(mu, volume).sum() == 1000
    assert (mu.z == volume.h).all()

    # Check callback
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

    mus = gen(1000)
    while MuonResampler.check_mu_batch(MuonBatch(mus, volume.h), volume).sum() == 1000:
        mus = gen(1000)
    vw.fit_params = FitParams(mu=MuonBatch(mus, volume.h))
    assert (vw.fit_params.mu.z == volume.h).all()
    vw.mu_generator = gen

    mr = MuonResampler()
    assert check_callback_base(mr)
    mr.set_wrapper(vw)
    mr.on_mu_batch_begin()
    assert (vw.fit_params.mu.z == volume.h).all()
    assert mr.check_mu_batch(vw.fit_params.mu, volume).sum() == 1000


def get_heatmap_detector() -> PanelDetectorLayer:
    return PanelDetectorLayer(
        pos="above",
        lw=LW,
        z=1,
        size=2 * SZ,
        panels=[
            DetectorHeatMap(
                res=1.0,
                eff=1.0,
                init_xyz=[0.5, 0.5, 0.9],
                init_xy_span=[-0.5, 0.5],
            )
        ],
    )


def test_no_more_nans_heatmap():
    cb = NoMoreNaNs()
    assert check_callback_base(cb)

    vw = MockWrapper()
    vw.volume = MockVolume()
    vw.volume.budget_weights.grad = Tensor([np.nan, np.nan, np.nan])
    vw.volume.get_detectors = lambda: [l]

    l = get_heatmap_detector()
    p = l.panels[0]
    p.mu.grad = p.mu.data
    p.z.grad = p.z.data
    p.sig.grad = p.sig.data
    p.norm.grad = p.norm.data
    p.mu.grad[:1] = Tensor([np.nan])
    p.z.grad[:1] = Tensor([np.nan])
    p.sig.grad[:1] = Tensor([np.nan])
    p.norm.grad[:1] = Tensor([np.nan])
    assert p.mu.grad.sum().isnan()
    assert p.z.grad.sum().isnan()
    assert p.sig.grad.sum().isnan()
    assert p.norm.grad.sum().isnan()
    assert vw.volume.budget_weights.grad.sum().isnan()

    cb.set_wrapper(vw)
    cb.on_backwards_end()
    assert not p.mu.grad.sum().isnan()
    assert not p.z.grad.sum().isnan()
    assert not p.sig.grad.sum().isnan()
    assert not p.norm.grad.sum().isnan()
    assert not vw.volume.budget_weights.grad.sum().isnan()


def test_heat_map_gif():
    cb = HeatMapGif(PKG_DIR / "heatmap.gif")
    assert check_callback_base(cb)

    l = get_heatmap_detector()
    vw = MockWrapper()
    vw.fit_params = FitParams(state="valid", cb_savepath=Path(PKG_DIR))
    vw.volume = MockVolume()
    vw.volume.get_detectors = lambda: [l]
    cb.set_wrapper(vw)
    cb.on_train_begin()

    assert len(cb._buffer_files) == 0
    cb.on_epoch_begin()
    assert len(cb._buffer_files) == 0
    vw.fit_params.state = "train"
    cb.on_epoch_begin()
    cb.on_epoch_begin()
    assert len(cb._buffer_files) == 2
    assert len(glob(str(PKG_DIR / "temp_heatmap_*.png"))) == 2

    cb.on_train_end()
    assert len(cb._buffer_files) == 3
    assert len(glob(str(PKG_DIR / "temp_heatmap_*.png"))) == 0
    assert len(glob(str(PKG_DIR / "heatmap.gif"))) == 1


def test_save_2_hdf5_pred_handler():
    try:
        out_path = Path(PKG_DIR / "test_pred_save.h5")
        if out_path.exists():
            out_path.unlink()
        cb = Save2HDF5PredHandler(out_path, use_volume_target=False)
        assert check_callback_base(cb)

        vw = MockWrapper()
        vw.fit_params = FitParams(state="train", pred=Tensor([1]))
        cb.set_wrapper(vw)
        vw.volume = MockVolume()
        vw.volume.get_rad_cube = lambda: Tensor([3, 4])

        assert not out_path.exists()
        cb.on_pred_begin()
        vw.fit_params.state = "test"
        cb.on_x0_pred_end()
        assert len(cb.preds) == 0
        assert out_path.exists()
        vw.fit_params.pred = vw.fit_params.pred + 1
        cb.on_x0_pred_end()

        with h5py.File(out_path) as h5:
            preds, targs = h5["preds"][()], h5["targs"][()]
        assert (preds == np.array([[1], [2]])).all()
        assert (targs == np.array([[3, 4], [3, 4]])).all()

    finally:
        out_path.unlink()


@pytest.mark.parametrize("n_warmups", [0, 1, 2])
def test_sigmoid_panel_smoothness_schedule(n_warmups):
    cb = SigmoidPanelSmoothnessSchedule((1, 0.01))
    assert check_callback_base(cb)

    vw = MockWrapper()
    wcbs = [WarmupCallback(1) for _ in range(n_warmups)]
    cbs = wcbs + [cb]
    vw.fit_params = FitParams(state="train", warmup_cbs=wcbs, cbs=cbs, n_epochs=3 + n_warmups)
    cb.set_wrapper(vw)
    vw.volume = MockVolume()
    det = get_sigmoid_panel_detector()
    panel = det.panels[0]
    vw.volume.get_detectors = lambda: [det]

    for c in vw.fit_params.cbs:
        c.set_wrapper(vw)
    assert panel.smooth == 0.5
    for c in vw.fit_params.cbs:
        c.on_train_begin()
    assert (panel.smooth - Tensor([1.0])).abs() < 1e-5
    for i, smooth in enumerate(Tensor([1.0 for _ in range(n_warmups)] + [1.0, 0.1, 0.01])):
        vw.fit_params.epoch = i + 1
        for c in vw.fit_params.cbs:
            c.on_epoch_begin()
        assert (panel.smooth - smooth).abs() < 1e-5
        for c in vw.fit_params.cbs:
            c.on_epoch_end()
