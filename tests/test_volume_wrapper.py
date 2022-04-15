from functools import partial
from pathlib import Path
import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.volume import Volume, PassiveLayer, VoxelDetectorLayer, PanelDetectorLayer, DetectorPanel, DetectorHeatMap
from tomopt.optimisation.wrapper.volume_wrapper import VoxelVolumeWrapper, FitParams, PanelVolumeWrapper, HeatMapVolumeWrapper
from tomopt.optimisation.callbacks.callback import Callback
from tomopt.optimisation.callbacks.cyclic_callbacks import CyclicCallback
from tomopt.optimisation.callbacks.monitors import MetricLogger
from tomopt.optimisation.callbacks.diagnostic_callbacks import ScatterRecord
from tomopt.optimisation.data.passives import PassiveYielder
from tomopt.optimisation.loss import VoxelX0Loss
from tomopt.optimisation.callbacks.grad_callbacks import NoMoreNaNs
from tomopt.optimisation.callbacks.eval_metric import EvalMetric
from tomopt.optimisation.callbacks.data_callbacks import MuonResampler
from tomopt.muon.generation import MuonGenerator2016
from tomopt.optimisation.callbacks.pred_callbacks import PredHandler

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
    if z >= 0.4 and z <= 0.5:
        rad_length[5:, 5:] = X0["lead"]
    return rad_length


def get_voxel_layers(init_res: float = 1e3):
    def eff_cost(x: Tensor) -> Tensor:
        return torch.expm1(3 * F.relu(x))

    def res_cost(x: Tensor) -> Tensor:
        return F.relu(x / 100) ** 2

    layers = []
    init_eff = 0.5
    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(
                VoxelDetectorLayer(pos=pos, init_eff=init_eff, init_res=init_res, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
            )
        else:
            pos = "below"
            layers.append(PassiveLayer(lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


def get_panel_layers(init_res: float = 1e3) -> nn.ModuleList:
    def area_cost(a: Tensor) -> Tensor:
        return F.relu(a)

    layers = []
    init_eff = 0.9
    n_panels = 4
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0], area_cost_func=area_cost)
                for i in range(n_panels)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                DetectorPanel(
                    res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)], init_xy_span=[1.0, 1.0], area_cost_func=area_cost
                )
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


def test_voxel_volume_wrapper_methods():
    volume = Volume(get_voxel_layers())
    vw = VoxelVolumeWrapper(
        volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15)
    )

    # _build_opt
    for l, r, e in zip(volume.get_detectors(), vw.opts["res_opt"].param_groups[0]["params"], vw.opts["eff_opt"].param_groups[0]["params"]):
        assert torch.all(l.resolution == r)
        assert torch.all(l.efficiency == e)

    # get_detectors
    for i, j, k in zip(volume.get_detectors(), vw.get_detectors(), [volume.layers[0], volume.layers[1], volume.layers[-2], volume.layers[-1]]):
        assert i == j == k

    # get_param_count
    assert vw.get_param_count() == 4 * 2 * 10 * 10

    # save
    def zero_params(v, vw):
        for l in v.get_detectors():
            nn.init.zeros_(l.resolution)
            nn.init.zeros_(l.efficiency)
        assert l.resolution.sum() == 0
        assert l.efficiency.sum() == 0
        vw.set_opt_lr(0, "res_opt")
        vw.set_opt_lr(0, "eff_opt")

    path = Path("tests/test_voxel_save.pt")
    vw.save("tests/test_voxel_save.pt")
    assert path.exists()
    zero_params(volume, vw)

    vw.load(path)
    for l in volume.get_detectors():
        assert l.resolution.sum() != 0
        assert l.efficiency.sum() != 0
    vw.get_opt_lr("res_opt") == 0
    vw.get_opt_lr("eff_opt") == 0

    # from_save
    zero_params(volume, vw)
    vw = VoxelVolumeWrapper.from_save(
        path, volume=volume, res_opt=partial(optim.SGD, lr=0), eff_opt=partial(optim.Adam, lr=0), loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15)
    )
    for l in volume.get_detectors():
        assert l.resolution.sum() != 0
        assert l.efficiency.sum() != 0
    vw.get_opt_lr("res_opt") != 0
    vw.get_opt_lr("eff_opt") != 0


def test_panel_volume_wrapper_methods():
    volume = Volume(get_panel_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=2e0, momentum=0.95),
        z_pos_opt=partial(optim.Adam, lr=2e-2),
        xy_span_opt=partial(optim.Adam, lr=2e-0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )

    # _build_opt
    for p, xy, z, s in zip(
        [p for l in volume.get_detectors() for p in l.panels],
        vw.opts["xy_pos_opt"].param_groups[0]["params"],
        vw.opts["z_pos_opt"].param_groups[0]["params"],
        vw.opts["xy_span_opt"].param_groups[0]["params"],
    ):
        assert torch.all(p.xy == xy)
        assert torch.all(p.z == z)
        assert torch.all(p.xy_span == s)

    # get_detectors
    for i, j, k in zip(volume.get_detectors(), vw.get_detectors(), (volume.layers[0], volume.layers[-1])):
        assert i == j == k

    # get_param_count
    assert vw.get_param_count() == 4 * 2 * 5

    # save
    def zero_params(v, vw):
        for l in v.get_detectors():
            for p in l.panels:
                nn.init.zeros_(p.xy)
                nn.init.zeros_(p.z)
                nn.init.zeros_(p.xy_span)
                assert p.xy.sum() == 0
                assert p.z.sum() == 0
                assert p.xy_span.sum() == 0
        vw.set_opt_lr(0, "xy_pos_opt")
        vw.set_opt_lr(0, "z_pos_opt")
        vw.set_opt_lr(0, "xy_span_opt")

    path = Path("tests/test_panel_save.pt")
    vw.save("tests/test_panel_save.pt")
    assert path.exists()
    zero_params(volume, vw)

    vw.load(path)
    for l in volume.get_detectors():
        for p in l.panels:
            assert p.xy.sum() != 0
            assert p.z.sum() != 0
            assert p.xy_span.sum() != 0
    vw.get_opt_lr("xy_pos_opt") != 0
    vw.get_opt_lr("z_pos_opt") != 0
    vw.get_opt_lr("xy_span_opt") != 0

    # from_save
    zero_params(volume, vw)
    vw = PanelVolumeWrapper.from_save(
        path,
        volume=volume,
        xy_pos_opt=partial(optim.SGD, lr=0),
        z_pos_opt=partial(optim.Adam, lr=0),
        xy_span_opt=partial(optim.Adam, lr=0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )
    for l in volume.get_detectors():
        for p in l.panels:
            assert p.xy.sum() != 0
            assert p.z.sum() != 0
            assert p.xy_span.sum() != 0
    vw.get_opt_lr("xy_pos_opt") != 0
    vw.get_opt_lr("z_pos_opt") != 0
    vw.get_opt_lr("xy_span_opt") != 0


def test_volume_wrapper_parameters():
    volume = Volume(get_voxel_layers())
    vw = VoxelVolumeWrapper(
        volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15)
    )

    assert vw.get_opt_lr("res_opt") == 2e1
    assert vw.get_opt_lr("eff_opt") == 2e-5
    assert vw.get_opt_mom("res_opt") == 0.95
    assert vw.get_opt_mom("eff_opt") == 0.9

    vw.set_opt_lr(2, "res_opt")
    vw.set_opt_lr(3, "eff_opt")
    vw.set_opt_mom(0.8, "res_opt")
    vw.set_opt_mom(0.7, "eff_opt")

    assert vw.get_opt_lr("res_opt") == 2
    assert vw.get_opt_lr("eff_opt") == 3
    assert vw.get_opt_mom("res_opt") == 0.8
    assert vw.get_opt_mom("eff_opt") == 0.7


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize("state", ["train", "valid", "test"])
def test_volume_wrapper_scan_volume(state, mocker):  # noqa F811
    volume = Volume(get_panel_layers())
    volume.load_rad_length(arb_rad_length)
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=2e0, momentum=0.95),
        z_pos_opt=partial(optim.Adam, lr=2e-2),
        xy_span_opt=partial(optim.Adam, lr=2e-0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )
    cb = Callback()
    cb.set_wrapper(vw)
    vw.fit_params = FitParams(n_mu_per_volume=100, mu_bs=10, cbs=[cb], state=state)
    mocker.spy(vw, "mu_generator")
    mocker.spy(vw, "loss_func")
    mocker.spy(volume, "forward")
    mocker.spy(cb, "on_mu_batch_begin")
    mocker.spy(cb, "on_scatter_end")
    mocker.spy(cb, "on_mu_batch_end")
    mocker.spy(cb, "on_x0_pred_begin")
    mocker.spy(cb, "on_x0_pred_end")

    vw._scan_volume()

    assert vw.mu_generator.call_count == 10
    vw.mu_generator.assert_called_with(10)
    assert volume.forward.call_count == 10
    assert cb.on_mu_batch_begin.call_count == 10
    assert cb.on_scatter_end.call_count == 10
    assert cb.on_mu_batch_end.call_count == 10
    assert cb.on_x0_pred_begin.call_count == 1
    assert cb.on_x0_pred_end.call_count == 1
    assert vw.fit_params.pred.shape == torch.Size((6, 10, 10))
    assert vw.fit_params.inv_weight.shape == torch.Size((6, 10, 10))

    if state == "test":
        assert vw.loss_func.call_count == 0
    else:
        assert vw.loss_func.call_count == 1
        assert (loss1 := vw.fit_params.loss_val) is not None
        vw._scan_volume()
        assert loss1 < vw.fit_params.loss_val


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_volume_wrapper_scan_volume_mu_batch(mocker):  # noqa F811
    volume = Volume(get_panel_layers())
    volume.load_rad_length(arb_rad_length)
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=2e0, momentum=0.95),
        z_pos_opt=partial(optim.Adam, lr=2e-2),
        xy_span_opt=partial(optim.Adam, lr=2e-0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )
    vw.fit_params = FitParams(n_mu_per_volume=100, mu_bs=100, state="train")
    gen = MuonGenerator2016.from_volume(volume)
    mu = MuonResampler.resample(gen(100), volume=volume, gen=gen)

    # Fix scattering
    mocker.patch("tomopt.volume.layer.torch.randn", lambda n, device: torch.ones(n, device=device))
    mocker.patch("tomopt.volume.layer.torch.rand", lambda n, device: 0.35 * torch.ones(n, device=device))

    class mu_batch_yielder:
        def __init__(self, mu: Tensor) -> None:
            self.mu = mu.clone()
            self.i = 0

        def __call__(self, n: int) -> Tensor:
            b = self.mu[self.i : self.i + n]
            self.i += n
            return b

    sr = ScatterRecord()
    sr.set_wrapper(vw)
    vw.fit_params.cbs = [sr]
    vw.mu_generator = mu_batch_yielder(mu)
    vw._scan_volume()
    pred_1b, weight_1b = vw.fit_params.pred.detach().clone(), vw.fit_params.inv_weight.detach().clone()
    scatters_1b = sr.get_record()
    loss_1b = vw.fit_params.loss_val.detach().clone()

    vw.fit_params.loss_val = None
    vw.fit_params.pred = None
    vw.fit_params.inv_weight = None

    sr._reset()
    vw.mu_generator = mu_batch_yielder(mu)
    vw.fit_params.mu_bs = 10
    vw._scan_volume()
    scatters_10b = sr.get_record()
    loss_10b = vw.fit_params.loss_val.detach().clone()

    assert scatters_1b.shape == scatters_10b.shape
    assert torch.all(scatters_1b == scatters_10b)

    pred_10b, weight_10b = vw.fit_params.pred.detach().clone(), vw.fit_params.inv_weight.detach().clone()
    diff = torch.abs((weight_1b - weight_10b) / weight_1b)
    mask = diff > 1e-7
    print("diff", diff[mask])
    print("1bp", pred_1b[mask])
    print("10bp", pred_10b[mask])
    print("1bw", weight_1b[mask])
    print("10bw", weight_10b[mask])
    print("preds", (pred_1b - pred_10b).mean(), pred_1b.mean(), pred_10b.mean())
    print("weights", (weight_1b - weight_10b).mean(), weight_1b.mean(), weight_10b.mean())
    print("loss", (loss_1b - loss_10b).mean(), loss_1b.mean(), loss_10b.mean())
    assert torch.abs((pred_1b - pred_10b) / pred_1b).sum() < 1e-4
    assert torch.abs((weight_1b - weight_10b) / weight_1b).sum() < 1e-4
    assert torch.abs((loss_1b - loss_10b) / loss_1b).sum() < 1e-4


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize("state", ["train", "valid", "test"])
def test_volume_wrapper_scan_volumes(state, mocker):  # noqa F811
    volume = Volume(get_panel_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=0),
        z_pos_opt=partial(optim.SGD, lr=0),
        xy_span_opt=partial(optim.SGD, lr=0),
        loss_func=VoxelX0Loss(target_budget=None),
    )
    cb = Callback()
    cb.set_wrapper(vw)
    vw.fit_params = FitParams(n_mu_per_volume=100, mu_bs=10, cbs=[cb], state=state, passive_bs=2)
    py = PassiveYielder([arb_rad_length, arb_rad_length, arb_rad_length, arb_rad_length, arb_rad_length])
    mocker.spy(vw, "_scan_volume")
    mocker.spy(cb, "on_volume_begin")
    mocker.spy(cb, "on_volume_end")
    mocker.spy(cb, "on_volume_batch_begin")
    mocker.spy(cb, "on_volume_batch_end")
    mocker.spy(cb, "on_backwards_begin")
    mocker.spy(cb, "on_backwards_end")
    mocker.spy(vw.opts["xy_pos_opt"], "zero_grad")
    mocker.spy(vw.opts["z_pos_opt"], "zero_grad")
    mocker.spy(vw.opts["xy_span_opt"], "zero_grad")
    mocker.spy(vw.opts["xy_pos_opt"], "step")
    mocker.spy(vw.opts["z_pos_opt"], "step")
    mocker.spy(vw.opts["xy_span_opt"], "step")

    mocker.patch.object(vw, "loss_func", return_value=torch.tensor(3.0, requires_grad=True))

    vw._scan_volumes(py)

    if state == "test":
        assert vw._scan_volume.call_count == 5
        assert cb.on_volume_begin.call_count == 5
        assert cb.on_volume_end.call_count == 5
        assert cb.on_volume_batch_begin.call_count == 0
        assert cb.on_volume_batch_end.call_count == 0
        assert vw.fit_params.loss_val is None
    else:
        assert vw._scan_volume.call_count == 4
        assert cb.on_volume_begin.call_count == 4
        assert cb.on_volume_end.call_count == 4
        assert cb.on_volume_batch_begin.call_count == 2
        assert cb.on_volume_batch_end.call_count == 2
        assert vw.fit_params.loss_val == 6
        assert vw.fit_params.mean_loss == 3

    if state == "train":
        assert cb.on_backwards_begin.call_count == 2
        assert cb.on_backwards_end.call_count == 2
        assert vw.opts["xy_pos_opt"].zero_grad.call_count == 2
        assert vw.opts["z_pos_opt"].zero_grad.call_count == 2
        assert vw.opts["xy_span_opt"].zero_grad.call_count == 2
        assert vw.opts["xy_pos_opt"].step.call_count == 2
        assert vw.opts["z_pos_opt"].zero_grad.call_count == 2
        assert vw.opts["xy_span_opt"].step.call_count == 2
    else:
        assert cb.on_backwards_begin.call_count == 0
        assert cb.on_backwards_end.call_count == 0
        assert vw.opts["xy_pos_opt"].zero_grad.call_count == 0
        assert vw.opts["z_pos_opt"].zero_grad.call_count == 0
        assert vw.opts["xy_span_opt"].zero_grad.call_count == 0
        assert vw.opts["xy_pos_opt"].step.call_count == 0
        assert vw.opts["z_pos_opt"].step.call_count == 0
        assert vw.opts["xy_span_opt"].step.call_count == 0


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_volume_wrapper_fit_epoch(mocker):  # noqa F811
    volume = Volume(get_panel_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=0),
        z_pos_opt=partial(optim.SGD, lr=0),
        xy_span_opt=partial(optim.SGD, lr=0),
        loss_func=VoxelX0Loss(target_budget=None),
    )
    cb = NoMoreNaNs()
    cb.set_wrapper(vw)
    trn_py = PassiveYielder([arb_rad_length, arb_rad_length, arb_rad_length])
    val_py = PassiveYielder([arb_rad_length, arb_rad_length])
    vw.fit_params = FitParams(n_mu_per_volume=100, mu_bs=100, cbs=[cb], trn_passives=trn_py, val_passives=val_py, passive_bs=1)
    mocker.spy(cb, "on_epoch_begin")
    mocker.spy(cb, "on_epoch_end")
    mocker.spy(cb, "on_volume_begin")
    mocker.spy(cb, "on_backwards_begin")
    mocker.spy(cb, "on_backwards_end")
    mocker.spy(vw, "_scan_volumes")
    mocker.spy(volume, "train")
    mocker.spy(volume, "eval")

    vw._fit_epoch()

    assert cb.on_epoch_begin.call_count == 2
    assert cb.on_epoch_end.call_count == 2
    assert cb.on_volume_begin.call_count == 5
    assert cb.on_backwards_begin.call_count == 3
    assert cb.on_backwards_end.call_count == 3
    assert vw._scan_volumes.call_count == 2
    assert volume.train.call_count == 2  # eval calls train(False)
    assert volume.eval.call_count == 1


def test_volume_wrapper_sort_cbs():
    cbs = [Callback(), CyclicCallback(), MetricLogger(), EvalMetric(False)]
    cyclic_cbs, metric_log, metric_cbs = VoxelVolumeWrapper._sort_cbs(cbs)
    assert len(cyclic_cbs) == 1
    assert len(metric_cbs) == 1
    assert cyclic_cbs[0] == cbs[1]
    assert metric_log == cbs[2]
    assert metric_cbs[0] == cbs[3]


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_volume_wrapper_fit(mocker):  # noqa F811
    volume = Volume(get_panel_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=0),
        z_pos_opt=partial(optim.SGD, lr=0),
        xy_span_opt=partial(optim.SGD, lr=0),
        loss_func=VoxelX0Loss(target_budget=None),
    )
    trn_py = PassiveYielder([arb_rad_length, arb_rad_length, arb_rad_length])
    val_py = PassiveYielder([arb_rad_length, arb_rad_length])
    cb = NoMoreNaNs()
    mocker.spy(cb, "set_wrapper")
    mocker.spy(cb, "on_train_begin")
    mocker.spy(cb, "on_train_end")
    mocker.spy(vw, "_fit_epoch")

    vw.fit(n_epochs=2, n_mu_per_volume=100, mu_bs=100, passive_bs=1, trn_passives=trn_py, val_passives=val_py, cbs=[cb])

    assert cb.set_wrapper.call_count == 1
    assert cb.on_train_begin.call_count == 1
    assert cb.on_train_end.call_count == 1
    assert vw._fit_epoch.call_count == 2


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_volume_wrapper_predict(mocker):  # noqa F811
    volume = Volume(get_panel_layers())
    vw = PanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(optim.SGD, lr=2e0, momentum=0.95),
        z_pos_opt=partial(optim.Adam, lr=2e-2),
        xy_span_opt=partial(optim.Adam, lr=2e-0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )
    py = PassiveYielder([arb_rad_length, arb_rad_length, arb_rad_length])
    cbs = [Callback()]
    pred_cb = PredHandler()
    for c in [cbs[0], pred_cb]:
        mocker.spy(c, "set_wrapper")
        mocker.spy(c, "on_pred_begin")
        mocker.spy(c, "on_pred_end")
    mocker.spy(c, "get_preds")

    preds = vw.predict(py, n_mu_per_volume=100, mu_bs=100, pred_cb=pred_cb, cbs=cbs)

    for c in [cbs[0], pred_cb]:
        assert c.set_wrapper.call_count == 1
        assert c.on_pred_begin.call_count == 1
        assert c.on_pred_end.call_count == 1
    assert pred_cb.get_preds.call_count == 1
    assert len(cbs) == 1
    assert len(preds) == 3
    assert len(preds[0]) == 2
    assert preds[0][0].shape == (6, 10, 10)
    assert preds[0][1].shape == (6, 10, 10)
    assert preds[0][0].sum() > 0


def get_heatmap_layers(init_res: float = 1e3) -> nn.ModuleList:
    def area_cost(a: Tensor) -> Tensor:
        return F.relu(a)

    layers = []
    init_eff = 0.9
    n_panels = 4
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorHeatMap(
                    res=init_res,
                    eff=init_eff,
                    init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / n_panels)],
                    init_xy_span=[-0.5, 0.5],
                    area_cost_func=area_cost,
                )
                for i in range(n_panels)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                DetectorHeatMap(
                    res=init_res,
                    eff=init_eff,
                    init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / n_panels)],
                    init_xy_span=[-0.5, 0.5],
                    area_cost_func=area_cost,
                )
                for i in range(n_panels)
            ],
        )
    )

    return nn.ModuleList(layers)


def test_heatmap_volume_wrapper_methods():
    volume = Volume(get_heatmap_layers())
    vw = HeatMapVolumeWrapper(
        volume,
        mu_opt=partial(optim.SGD, lr=2e0, momentum=0.95),
        z_pos_opt=partial(optim.Adam, lr=2e-2),
        sig_opt=partial(optim.Adam, lr=2e-0),
        norm_opt=partial(optim.Adam, lr=2e-0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )

    # _build_opt
    for p, xy, z, s, n in zip(
        [p for l in volume.get_detectors() for p in l.panels],
        vw.opts["mu_opt"].param_groups[0]["params"],
        vw.opts["z_pos_opt"].param_groups[0]["params"],
        vw.opts["sig_opt"].param_groups[0]["params"],
        vw.opts["norm_opt"].param_groups[0]["params"],
    ):
        assert torch.all(p.mu == xy)
        assert torch.all(p.z == z)
        assert torch.all(p.sig == s)
        assert torch.all(p.norm == n)

    # get_detectors
    for i, j, k in zip(volume.get_detectors(), vw.get_detectors(), (volume.layers[0], volume.layers[-1])):
        assert i == j == k

    # get_param_count
    assert vw.get_param_count() == 4.0 * 2.0 * (30.0 * 4.0 + 1 + 1)

    # save
    def zero_params(v, vw):
        for l in v.get_detectors():
            for p in l.panels:
                nn.init.zeros_(p.mu)
                nn.init.zeros_(p.z)
                nn.init.zeros_(p.sig)
                nn.init.zeros_(p.norm)
                nn.init.zeros_(p.resolution)
                nn.init.zeros_(p.efficiency)
                nn.init.zeros_(p.xy_fix)
                nn.init.zeros_(p.xy_span_fix)

                assert p.mu.sum() == 0
                assert p.z.sum() == 0
                assert p.sig.sum() == 0
                assert p.norm.sum() == 0
                assert p.resolution.sum() == 0
                assert p.efficiency.sum() == 0
                assert p.xy_fix.sum() == 0
                assert p.xy_span_fix.abs().sum() == 0

        vw.set_opt_lr(0, "mu_opt")
        vw.set_opt_lr(0, "z_pos_opt")
        vw.set_opt_lr(0, "sig_opt")
        vw.set_opt_lr(0, "norm_opt")

    path = Path("tests/test_panel_save.pt")
    vw.save("tests/test_panel_save.pt")
    assert path.exists()
    zero_params(volume, vw)

    vw.load(path)
    for l in volume.get_detectors():
        for p in l.panels:
            assert p.mu.sum() != 0
            assert p.z.sum() != 0
            assert p.sig.sum() != 0
            assert p.norm.sum() != 0
            assert p.resolution.sum() != 0
            assert p.efficiency.sum() != 0
            assert p.xy_fix.sum() != 0
            assert p.xy_span_fix.abs().sum() != 0
    vw.get_opt_lr("mu_opt") != 0
    vw.get_opt_lr("z_pos_opt") != 0
    vw.get_opt_lr("sig_opt") != 0
    vw.get_opt_lr("norm_opt") != 0

    # from_save
    zero_params(volume, vw)
    vw = HeatMapVolumeWrapper.from_save(
        path,
        volume=volume,
        mu_opt=partial(optim.SGD, lr=0),
        z_pos_opt=partial(optim.Adam, lr=0),
        sig_opt=partial(optim.Adam, lr=0),
        norm_opt=partial(optim.Adam, lr=0),
        loss_func=VoxelX0Loss(target_budget=1, cost_coef=0.15),
    )
    for l in volume.get_detectors():
        for p in l.panels:
            assert p.mu.sum() != 0
            assert p.z.sum() != 0
            assert p.sig.sum() != 0
            assert p.norm.sum() != 0
            assert p.resolution.sum() != 0
            assert p.efficiency.sum() != 0
            assert p.xy_fix.sum() != 0
            assert p.xy_span_fix.abs().sum() != 0
    vw.get_opt_lr("mu_opt") != 0
    vw.get_opt_lr("z_pos_opt") != 0
    vw.get_opt_lr("sig_opt") != 0
    vw.get_opt_lr("norm_opt") != 0
