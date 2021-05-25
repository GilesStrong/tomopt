from __future__ import annotations
from fastcore.all import Path
from fastprogress import progress_bar
from typing import Callable, Iterator, Optional, List, Any, Tuple
from fastprogress.fastprogress import ProgressBar
import numpy as np

import torch
from torch import nn, Tensor

from ..data import PassiveYielder
from ..callbacks import MetricLogger, PredHandler
from ..callbacks.callback import Callback
from ..callbacks.cyclic_callbacks import CyclicCallback
from ..callbacks.eval_metric import EvalMetric
from ...optimisation.loss import DetectorLoss
from ...volume import Volume, DetectorLayer
from ...core import X0
from ...muon import generate_batch, MuonBatch
from ...inference import ScatterBatch, X0Inferer

__all__ = ["VolumeWrapper"]

r"""
This FitParams and VolumeWrapper are modified versions of the FitParams in LUMIN (https://github.com/GilesStrong/lumin/blob/v0.7.2/lumin/nn/models/abs_model.py#L16)
and Model in LUMIN (https://github.com/GilesStrong/lumin/blob/master/lumin/nn/models/model.py#L32), distributed under the following lincence:
    Copyright 2018 onwards Giles Strong

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Usage is compatible with the AGPL licence underwhich TomOpt is distributed.
Stated changes: adaption of FitParams to pass type-checking, heavy adaptation of Model to be suitable for task specific training
"""


class FitParams:
    state: Optional[str] = None
    pred: Optional[Tensor] = None
    wpreds: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    weight: Optional[Tensor] = None
    n_mu_per_volume: Optional[int] = None
    mu_bs: Optional[int] = None
    mu: Optional[MuonBatch] = None
    cbs: Optional[List[Callback]] = None
    sb: Optional[ScatterBatch] = None
    loss_val: Optional[Tensor] = None
    volume_id: Optional[int] = None
    cb_savepath: Optional[Path] = None
    trn_passives: Optional[PassiveYielder] = None
    val_passives: Optional[PassiveYielder] = None
    tst_passives: Optional[PassiveYielder] = None
    passive_bs: Optional[int] = None
    mean_loss: Optional[Tensor] = None
    n_epochs: Optional[int] = None
    epoch_bar: Optional[ProgressBar] = None
    stop: Optional[bool] = None
    epoch: int = 0
    cyclic_cbs: Optional[List[CyclicCallback]] = None
    metric_log: Optional[MetricLogger] = None
    metric_cbs: Optional[List[EvalMetric]] = None
    use_default_pred: bool = True

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class VolumeWrapper:
    def __init__(
        self,
        volume: Volume,
        *,
        res_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
        eff_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
        loss_func: Optional[DetectorLoss],
        default_pred: Optional[float] = X0["beryllium"],
        mu_generator: Callable[[int], Tensor] = generate_batch,
    ):
        self.volume, self.loss_func, self.default_pred, self.mu_generator = volume, loss_func, default_pred, mu_generator
        self._build_opt(res_opt, eff_opt)
        self.parameters = self.volume.parameters

    def _build_opt(
        self, res_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], eff_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
    ) -> None:
        self.res_opt = res_opt(((l.resolution for l in self.volume.get_detectors())))
        self.eff_opt = eff_opt(((l.efficiency for l in self.volume.get_detectors())))

    def get_detectors(self) -> List[DetectorLayer]:
        return self.volume.get_detectors()

    def save(self, name: str) -> None:
        torch.save({"volume": self.volume.state_dict(), "res_opt": self.res_opt.state_dict(), "eff_opt": self.eff_opt.state_dict()}, str(name))

    def load(self, name: str) -> None:
        state = torch.load(name, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.volume.load_state_dict(state["volume"])
        self.res_opt.load_state_dict(state["res_opt"])
        self.eff_opt.load_state_dict(state["eff_opt"])

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        res_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
        eff_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
        loss_func: Optional[DetectorLoss],
        default_pred: Optional[float] = X0["beryllium"],
    ) -> VolumeWrapper:
        vw = cls(volume=volume, res_opt=res_opt, eff_opt=eff_opt, loss_func=loss_func, default_pred=default_pred)
        vw.load(name)
        return vw

    def get_param_count(self, trainable: bool = True) -> int:
        r"""
        Return number of parameters in detector.

        Arguments:
            trainable: if true (default) only count trainable parameters

        Returns:
            Number of (trainable) parameters in detector
        """

        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable)

    def _scan_volume(self) -> None:
        # Scan volume with muon batches
        self.fit_params.wpreds, self.fit_params.weights = [], []
        for _ in range(self.fit_params.n_mu_per_volume // self.fit_params.mu_bs):
            self.fit_params.mu = MuonBatch(self.mu_generator(self.fit_params.mu_bs), init_z=self.volume.h)
            for c in self.fit_params.cbs:
                c.on_mu_batch_begin()
            self.volume(self.fit_params.mu)
            self.fit_params.sb = ScatterBatch(self.fit_params.mu, self.volume)
            for c in self.fit_params.cbs:
                c.on_scatter_end()
            inferer = X0Inferer(self.fit_params.sb, self.default_pred)
            pred, wgt = inferer.pred_x0(inc_default=False)
            pred = torch.nan_to_num(pred)
            self.fit_params.wpreds.append(pred * wgt)
            self.fit_params.weights.append(wgt)
            for c in self.fit_params.cbs:
                c.on_mu_batch_end()

        # Predict volume based on all muon batches
        for c in self.fit_params.cbs:
            c.on_x0_pred_begin()
        wgt = torch.stack(self.fit_params.weights, dim=0).sum(0)
        pred = torch.stack(self.fit_params.wpreds, dim=0).sum(0) / wgt
        if self.fit_params.use_default_pred:
            pred, wgt = inferer.add_default_pred(pred, wgt)
        self.fit_params.weight = wgt
        self.fit_params.pred = pred

        for c in self.fit_params.cbs:
            c.on_x0_pred_end()

        # Compute loss for volume
        if self.fit_params.state != "test" and self.loss_func is not None:
            loss = self.loss_func(pred_x0=self.fit_params.pred, pred_weight=self.fit_params.weight, volume=self.volume)
            if self.fit_params.loss_val is None:
                self.fit_params.loss_val = loss
            else:
                self.fit_params.loss_val = self.fit_params.loss_val + loss

    def _scan_volumes(self, passives: PassiveYielder) -> None:
        for i, passive in enumerate(passives):
            self.fit_params.volume_id = i
            if self.fit_params.state != "test" and i % self.fit_params.passive_bs == 0:  # Volume batch start
                self.fit_params.loss_val = None
                for c in self.fit_params.cbs:
                    c.on_volume_batch_begin()

            self.volume.load_rad_length(passive)
            for c in self.fit_params.cbs:
                c.on_volume_begin()
            self._scan_volume()
            for c in self.fit_params.cbs:
                c.on_volume_end()

            if self.fit_params.state != "test" and (i + 1) % self.fit_params.passive_bs == 0:  # Volume batch end
                if self.fit_params.loss_val is not None:
                    self.fit_params.mean_loss = self.fit_params.loss_val / self.fit_params.passive_bs
                for c in self.fit_params.cbs:
                    c.on_volume_batch_end()

                if self.fit_params.state == "train":
                    # Compute update step
                    self.res_opt.zero_grad()
                    self.eff_opt.zero_grad()
                    for c in self.fit_params.cbs:
                        c.on_backwards_begin()
                    self.fit_params.mean_loss.backward()
                    for c in self.fit_params.cbs:
                        c.on_backwards_end()
                    self.res_opt.step()
                    self.eff_opt.step()

                if len(passives) - (i + 1) < self.fit_params.passive_bs:
                    break

    def _fit_epoch(self) -> None:
        def run_epoch(passives: PassiveYielder) -> None:
            for c in self.fit_params.cbs:
                c.on_epoch_begin()
            self._scan_volumes(passives)
            for c in self.fit_params.cbs:
                c.on_epoch_end()

        self.fit_params.epoch += 1
        # Training
        self.volume.train()
        self.fit_params.state = "train"
        run_epoch(self.fit_params.trn_passives)

        # Validation
        if self.fit_params.val_passives is not None:
            self.volume.eval()
            self.fit_params.state = "valid"
            run_epoch(self.fit_params.val_passives)

    @staticmethod
    def _sort_cbs(cbs: List[Callback]) -> Tuple[List[CyclicCallback], Optional[MetricLogger], Optional[List[EvalMetric]]]:
        cyclic_cbs, metric_log, metric_cbs = [], None, []
        for c in cbs:
            if isinstance(c, CyclicCallback):
                cyclic_cbs.append(c)  # CBs that might prevent a wrapper from stopping training due to a hyper-param cycle
            if isinstance(c, MetricLogger):
                metric_log = c  # CB that logs losses and eval_metrics
            if isinstance(c, EvalMetric):
                metric_cbs.append(c)  # CB that computes additional performance metrics
        return cyclic_cbs, metric_log, metric_cbs

    def fit(
        self,
        n_epochs: int,
        passive_bs: int,
        n_mu_per_volume: int,
        mu_bs: int,
        trn_passives: PassiveYielder,
        val_passives: Optional[PassiveYielder],
        cbs: Optional[List[Callback]] = None,
        cb_savepath: Path = Path("train_weights"),
    ) -> List[Callback]:

        if cbs is None:
            cbs = []
        cyclic_cbs, metric_log, metric_cbs = self._sort_cbs(cbs)

        self.fit_params = FitParams(
            cbs=cbs,
            cyclic_cbs=cyclic_cbs,
            metric_log=metric_log,
            metric_cbs=metric_cbs,
            stop=False,
            n_epochs=n_epochs,
            mu_bs=mu_bs,
            n_mu_per_volume=n_mu_per_volume,
            cb_savepath=Path(cb_savepath),
            trn_passives=trn_passives,
            val_passives=val_passives,
            passive_bs=passive_bs,
        )
        self.fit_params.cb_savepath.mkdir(parents=True, exist_ok=True)
        try:
            for c in self.fit_params.cbs:
                c.set_wrapper(self)
            for c in self.fit_params.cbs:
                c.on_train_begin()
            self.fit_params.epoch_bar = progress_bar(range(self.fit_params.n_epochs))
            for e in self.fit_params.epoch_bar:
                self._fit_epoch()
                if self.fit_params.stop:
                    break
            for c in self.fit_params.cbs:
                c.on_train_end()
        finally:
            self.fit_params = None
            torch.cuda.empty_cache()
        return cbs

    def predict(
        self,
        passives: PassiveYielder,
        n_mu_per_volume: int,
        mu_bs: int,
        use_default_pred: bool = False,
        pred_cb: PredHandler = PredHandler(),
        cbs: Optional[List[Callback]] = None,
        cb_savepath: Path = Path("train_weights"),
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if cbs is None:
            cbs = []
        cbs.append(pred_cb)
        passives.shuffle = False

        self.fit_params = FitParams(
            n_mu_per_volume=n_mu_per_volume,
            mu_bs=mu_bs,
            cbs=cbs,
            tst_passives=passives,
            state="test",
            cb_savepath=cb_savepath,
            use_default_pred=use_default_pred,
        )
        try:
            for c in self.fit_params.cbs:
                c.set_wrapper(self)
            self.volume.eval()
            for c in self.fit_params.cbs:
                c.on_pred_begin()
            self._scan_volumes(self.fit_params.tst_passives)
            for c in self.fit_params.cbs:
                c.on_pred_end()
        finally:
            self.fit_params = None
            cbs.pop()  # Remove pred_cb to avoid mutating cbs
            torch.cuda.empty_cache()
        return pred_cb.get_preds()

    @property
    def eff_lr(self) -> float:
        return self.eff_opt.param_groups[0]["lr"]

    @eff_lr.setter
    def eff_lr(self, lr: float) -> None:
        self.eff_opt.param_groups[0]["lr"] = lr

    @property
    def eff_mom(self) -> float:
        if "betas" in self.eff_opt.param_groups[0]:
            return self.eff_opt.param_groups[0]["betas"][0]
        else:
            return self.eff_opt.param_groups[0]["momentum"]

    @eff_mom.setter
    def eff_mom(self, mom: float) -> None:
        if "betas" in self.eff_opt.param_groups[0]:
            self.eff_opt.param_groups[0]["betas"] = (mom, self.eff_opt.param_groups[0]["betas"][1])
        else:
            self.eff_opt.param_groups[0]["momentum"] = mom

    @property
    def res_lr(self) -> float:
        return self.res_opt.param_groups[0]["lr"]

    @res_lr.setter
    def res_lr(self, lr: float) -> None:
        self.res_opt.param_groups[0]["lr"] = lr

    @property
    def res_mom(self) -> float:
        if "betas" in self.res_opt.param_groups[0]:
            return self.res_opt.param_groups[0]["betas"][0]
        else:
            return self.res_opt.param_groups[0]["momentum"]

    @res_mom.setter
    def res_mom(self, mom: float) -> None:
        if "betas" in self.res_opt.param_groups[0]:
            self.res_opt.param_groups[0]["betas"] = (mom, self.res_opt.param_groups[0]["betas"][1])
        elif "momentum" in self.res_opt.param_groups[0]:
            self.res_opt.param_groups[0]["momentum"] = mom
