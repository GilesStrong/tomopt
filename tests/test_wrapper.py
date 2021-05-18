from functools import partial
from pathlib import Path
from tomopt.optimisation.data.passives import PassiveYielder
import pytest
from pytest_mock import mocker  # noqa F401
import numpy as np

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.volume import Volume, PassiveLayer, DetectorLayer
from tomopt.optimisation.wrapper.volume_wrapper import VolumeWrapper, FitParams
from tomopt.optimisation.callbacks.callback import Callback
from tomopt.loss.loss import DetectorLoss
from tomopt.muon.generation import generate_batch

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
    if z >= 0.4 and z <= 0.5:
        rad_length[5:, 5:] = X0["lead"]
    return rad_length


def get_layers(init_res: float = 1e4):
    def eff_cost(x: Tensor) -> Tensor:
        return torch.expm1(3 * F.relu(x))

    def res_cost(x: Tensor) -> Tensor:
        return F.relu(x / 100) ** 2

    layers = []
    init_eff = 0.5
    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(DetectorLayer(pos=pos, init_eff=init_eff, init_res=init_res, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost))
        else:
            pos = "below"
            layers.append(PassiveLayer(lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


def test_volume_wrapper_methods():
    volume = Volume(get_layers())
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))

    # _build_opt
    for l, r, e in zip(volume.get_detectors(), vw.res_opt.param_groups[0]["params"], vw.eff_opt.param_groups[0]["params"]):
        assert torch.all(l.resolution == r)
        assert torch.all(l.efficiency == e)

    # get_detectors
    for i, j in zip(volume.get_detectors(), vw.get_detectors()):
        assert i == j

    # get_param_count
    assert vw.get_param_count() == 4 * 2 * 10 * 10

    # save
    def zero_params(v, vw):
        for l in v.get_detectors():
            nn.init.zeros_(l.resolution)
            nn.init.zeros_(l.efficiency)
        assert l.resolution.sum() == 0
        assert l.efficiency.sum() == 0
        vw.res_lr = 0
        vw.eff_lr = 0

    p = Path("test_save.pt")
    vw.save("test_save.pt")
    assert p.exists()
    zero_params(volume, vw)

    vw.load(p)
    for l in volume.get_detectors():
        assert l.resolution.sum() != 0
        assert l.efficiency.sum() != 0
    vw.res_lr != 0
    vw.eff_lr != 0

    # from_save
    zero_params(volume, vw)
    vw = VolumeWrapper.from_save(p, volume=volume, res_opt=partial(optim.SGD, lr=0), eff_opt=partial(optim.Adam, lr=0), loss_func=DetectorLoss(0.15))
    for l in volume.get_detectors():
        assert l.resolution.sum() != 0
        assert l.efficiency.sum() != 0
    vw.res_lr != 0
    vw.eff_lr != 0


def test_volume_wrapper_parameters():
    volume = Volume(get_layers())
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))

    assert vw.eff_lr == 2e-5
    assert vw.res_lr == 2e1
    assert vw.eff_mom == 0.9
    assert vw.res_mom == 0.95

    vw.eff_lr = 2
    vw.res_lr = 2
    vw.eff_mom = 0.8
    vw.res_mom = 0.8

    assert vw.eff_lr == 2
    assert vw.res_lr == 2
    assert vw.eff_mom == 0.8
    assert vw.res_mom == 0.8


@pytest.mark.parametrize("state", ["train", "valid", "test"])
def test_volume_wrapper_scan_volume(state, mocker):  # noqa F811
    volume = Volume(get_layers())
    volume.load_rad_length(arb_rad_length)
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))
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
    assert len(vw.fit_params.weights) == 10
    assert len(vw.fit_params.wpreds) == 10
    assert vw.fit_params.weights[0].shape == torch.Size((6, 10, 10))
    assert vw.fit_params.wpreds[0].shape == torch.Size((6, 10, 10))
    assert vw.fit_params.pred.shape == torch.Size((6, 10, 10))
    assert vw.fit_params.weight.shape == torch.Size((6, 10, 10))

    if state == "test":
        assert vw.loss_func.call_count == 0
    else:
        assert vw.loss_func.call_count == 1
        assert (loss1 := vw.fit_params.loss_val) is not None
        vw._scan_volume()
        assert loss1 < vw.fit_params.loss_val


@pytest.mark.parametrize("state", ["train", "test"])
def test_volume_wrapper_scan_volumes(state, mocker):  # noqa F811
    volume = Volume(get_layers())
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))
    cb = Callback()
    cb.set_wrapper(vw)
    vw.fit_params = FitParams(n_mu_per_volume=100, mu_bs=10, cbs=[cb], state=state)
    py = PassiveYielder([arb_rad_length, arb_rad_length])
    mocker.spy(vw, "_scan_volume")
    mocker.spy(cb, "on_volume_begin")
    mocker.spy(cb, "on_volume_end")
    mocker.patch.object(vw, "loss_func", return_value=3)

    vw._scan_volumes(py)

    assert vw._scan_volume.call_count == 2
    assert cb.on_volume_begin.call_count == 2
    assert cb.on_volume_end.call_count == 2
    if state == "test":
        assert vw.fit_params.mean_loss is None
    else:
        assert vw.fit_params.loss_val == 6
        assert vw.fit_params.mean_loss == 3


def test_volume_wrapper_fit_epoch(mocker):  # noqa F811
    volume = Volume(get_layers())
    vw = VolumeWrapper(volume, res_opt=partial(optim.SGD, lr=2e1, momentum=0.95), eff_opt=partial(optim.Adam, lr=2e-5), loss_func=DetectorLoss(0.15))
    cb = Callback()
    cb.set_wrapper(vw)
    trn_py = PassiveYielder([arb_rad_length, arb_rad_length, arb_rad_length])
    val_py = PassiveYielder([arb_rad_length, arb_rad_length])
    vw.fit_params = FitParams(n_mu_per_volume=100, mu_bs=10, cbs=[cb], trn_passives=trn_py, val_passives=val_py)
    mocker.spy(cb, "on_epoch_begin")
    mocker.spy(cb, "on_epoch_end")
    mocker.spy(cb, "on_volume_begin")
    mocker.spy(cb, "on_backwards_begin")
    mocker.spy(cb, "on_backwards_end")
    mocker.spy(vw, "_scan_volumes")
    mocker.spy(vw.res_opt, "zero_grad")
    mocker.spy(vw.eff_opt, "zero_grad")
    mocker.spy(vw.res_opt, "step")
    mocker.spy(vw.eff_opt, "step")
    mocker.spy(volume, "train")
    mocker.spy(volume, "eval")

    vw._fit_epoch()

    assert cb.on_epoch_begin.call_count == 2
    assert cb.on_epoch_end.call_count == 2
    assert cb.on_volume_begin.call_count == 5
    assert cb.on_backwards_begin.call_count == 1
    assert cb.on_backwards_end.call_count == 1
    assert vw._scan_volumes.call_count == 2
    assert vw.res_opt.zero_grad.call_count == 1
    assert vw.eff_opt.zero_grad.call_count == 1
    assert vw.res_opt.step.call_count == 1
    assert vw.eff_opt.step.call_count == 1
    assert volume.train.call_count == 2  # eval calls train(False)
    assert volume.eval.call_count == 1
