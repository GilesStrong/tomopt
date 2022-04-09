from __future__ import annotations
from fastcore.all import Path
from typing import Optional, List, Any, Tuple, Union, Dict, Type
from fastprogress.fastprogress import ConsoleProgressBar, NBProgressBar, ProgressBar
from fastprogress import master_bar, progress_bar
import numpy as np
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ..data import PassiveYielder
from ..callbacks import MetricLogger, PredHandler
from ..callbacks.callback import Callback
from ..callbacks.cyclic_callbacks import CyclicCallback
from ..callbacks.eval_metric import EvalMetric
from ...optimisation.loss.loss import AbsDetectorLoss
from ...volume import Volume, VoxelDetectorLayer, PanelDetectorLayer
from ...volume.layer import AbsDetectorLayer
from ...core import PartialOpt, DEVICE
from ...muon import MuonGenerator2016, MuonBatch
from ...muon.generation import AbsMuonGenerator
from ...inference.scattering import AbsScatterBatch, VoxelScatterBatch, PanelScatterBatch
from ...inference.volume import AbsVolumeInferer, VoxelX0Inferer, PanelX0Inferer

__all__ = ["VoxelVolumeWrapper", "PanelVolumeWrapper", "HeatMapVolumeWrapper"]

r"""
This FitParams and AbsVolumeWrapper are modified versions of the FitParams in LUMIN (https://github.com/GilesStrong/lumin/blob/v0.7.2/lumin/nn/models/abs_model.py#L16)
and Model in LUMIN (https://github.com/GilesStrong/lumin/blob/master/lumin/nn/models/model.py#L32), distributed under the following licence:
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
    volume_inferer: Optional[AbsVolumeInferer] = None
    state: Optional[str] = None
    pred: Optional[Tensor] = None
    inv_weight: Optional[Tensor] = None
    n_mu_per_volume: Optional[int] = None
    mu_bs: Optional[int] = None
    mu: Optional[MuonBatch] = None
    cbs: Optional[List[Callback]] = None
    sb: Optional[AbsScatterBatch] = None
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
    passive_bar: Optional[Union[NBProgressBar, ConsoleProgressBar]] = None
    device: torch.device = DEVICE

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class AbsVolumeWrapper(metaclass=ABCMeta):
    opts: Dict[str, Optimizer]

    def __init__(
        self,
        volume: Volume,
        *,
        partial_opts: Dict[str, PartialOpt],
        loss_func: Optional[AbsDetectorLoss],
        partial_scatter_inferer: Type[AbsScatterBatch],
        partial_volume_inferer: Type[AbsVolumeInferer],
        mu_generator: Optional[AbsMuonGenerator] = None,
    ):
        self.volume, self.loss_func = volume, loss_func
        if mu_generator is None:
            mu_generator = MuonGenerator2016.from_volume(volume)
        self.mu_generator = mu_generator
        self.partial_scatter_inferer, self.partial_volume_inferer = partial_scatter_inferer, partial_volume_inferer
        self.device = self.volume.device
        self._build_opt(**partial_opts)
        self.parameters = self.volume.parameters

    @abstractmethod
    def _build_opt(self, **kwargs: PartialOpt) -> None:
        r"""
        self.opts = {'res_opt': res_opt((l.resolution for l in self.volume.get_detectors())),
                     'eff_opt': eff_opt((l.efficiency for l in self.volume.get_detectors()))}
        """
        pass

    def get_detectors(self) -> List[AbsDetectorLayer]:
        return self.volume.get_detectors()

    def save(self, name: str) -> None:
        torch.save({"volume": self.volume.state_dict(), **{k: v.state_dict() for k, v in self.opts.items()}}, str(name))

    def load(self, name: str) -> None:
        state = torch.load(name, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.volume.load_state_dict(state["volume"])
        for k, v in state.items():
            if "_opt" in k:
                self.opts[k].load_state_dict(v)

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
        self.fit_params.pred, self.fit_params.inv_weight = None, None
        if self.fit_params.state != "test":
            muon_bar = progress_bar(range(self.fit_params.n_mu_per_volume // self.fit_params.mu_bs), display=False, leave=False)
        else:
            muon_bar = progress_bar(range(self.fit_params.n_mu_per_volume // self.fit_params.mu_bs), parent=self.fit_params.passive_bar)
        self.fit_params.volume_inferer = self.partial_volume_inferer(volume=self.volume)
        for _ in muon_bar:
            self.fit_params.mu = MuonBatch(self.mu_generator(self.fit_params.mu_bs), init_z=self.volume.h, device=self.fit_params.device)
            for c in self.fit_params.cbs:
                c.on_mu_batch_begin()
            self.volume(self.fit_params.mu)
            self.fit_params.sb = self.partial_scatter_inferer(mu=self.fit_params.mu, volume=self.volume)
            for c in self.fit_params.cbs:
                c.on_scatter_end()
            self.fit_params.volume_inferer.add_scatters(self.fit_params.sb)
            for c in self.fit_params.cbs:
                c.on_mu_batch_end()

        # Predict volume based on all muon batches
        for c in self.fit_params.cbs:
            c.on_x0_pred_begin()
        self.fit_params.pred, self.fit_params.inv_weight = self.fit_params.volume_inferer.get_prediction()
        for c in self.fit_params.cbs:
            c.on_x0_pred_end()

        # Compute loss for volume
        if self.fit_params.state != "test" and self.loss_func is not None and self.fit_params.pred is not None:
            loss = self.loss_func(pred=self.fit_params.pred, inv_pred_weight=self.fit_params.inv_weight, volume=self.volume)
            if self.fit_params.loss_val is None:
                self.fit_params.loss_val = loss
            else:
                self.fit_params.loss_val = self.fit_params.loss_val + loss

    def _scan_volumes(self, passives: PassiveYielder) -> None:
        if self.fit_params.state == "test":
            self.fit_params.passive_bar = master_bar(passives)
        for i, (passive, target) in enumerate(self.fit_params.passive_bar if self.fit_params.state == "test" else passives):
            self.fit_params.volume_id = i
            if self.fit_params.state != "test" and i % self.fit_params.passive_bs == 0:  # Volume batch start
                self.fit_params.loss_val = None
                for c in self.fit_params.cbs:
                    c.on_volume_batch_begin()

            self.volume.load_rad_length(passive, target)
            for c in self.fit_params.cbs:
                c.on_volume_begin()
            self._scan_volume()
            for c in self.fit_params.cbs:
                c.on_volume_end()

            if self.fit_params.state != "test" and (i + 1) % self.fit_params.passive_bs == 0:  # Volume batch end
                if self.fit_params.loss_val is not None:
                    self.fit_params.mean_loss = self.fit_params.loss_val / self.fit_params.passive_bs
                else:
                    self.fit_params.mean_loss = None
                for c in self.fit_params.cbs:
                    c.on_volume_batch_end()

                if self.fit_params.state == "train":
                    # Compute update step
                    for o in self.opts.values():
                        o.zero_grad()
                    for c in self.fit_params.cbs:
                        c.on_backwards_begin()
                    if self.fit_params.mean_loss is not None:
                        self.fit_params.mean_loss.backward()
                    for c in self.fit_params.cbs:
                        c.on_backwards_end()
                    if self.fit_params.mean_loss is not None:
                        for o in self.opts.values():
                            o.step()
                    for d in self.volume.get_detectors():
                        d.conform_detector()

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
            device=self.device,
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
            device=self.device,
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

    def get_opt_lr(self, opt: str) -> float:
        return self.opts[opt].param_groups[0]["lr"]

    def set_opt_lr(self, lr: float, opt: str) -> None:
        self.opts[opt].param_groups[0]["lr"] = lr

    def get_opt_mom(self, opt: str) -> float:
        if "betas" in self.opts[opt].param_groups[0]:
            return self.opts[opt].param_groups[0]["betas"][0]
        else:
            return self.opts[opt].param_groups[0]["momentum"]

    def set_opt_mom(self, mom: float, opt: str) -> None:
        if "betas" in self.opts[opt].param_groups[0]:
            self.opts[opt].param_groups[0]["betas"] = (mom, self.opts[opt].param_groups[0]["betas"][1])
        else:
            self.opts[opt].param_groups[0]["momentum"] = mom


class VoxelVolumeWrapper(AbsVolumeWrapper):
    def __init__(
        self,
        volume: Volume,
        *,
        res_opt: PartialOpt,
        eff_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        mu_generator: Optional[AbsMuonGenerator] = None,
        partial_scatter_inferer: Type[AbsScatterBatch] = VoxelScatterBatch,
        partial_volume_inferer: Type[AbsVolumeInferer] = VoxelX0Inferer,
    ):
        super().__init__(
            volume=volume,
            partial_opts={"res_opt": res_opt, "eff_opt": eff_opt},
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferer=partial_scatter_inferer,
            partial_volume_inferer=partial_volume_inferer,
        )

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        all_dets = self.volume.get_detectors()
        dets: List[VoxelDetectorLayer] = []
        for l in all_dets:
            if isinstance(l, VoxelDetectorLayer):
                dets.append(l)
        self.opts = {
            "res_opt": kwargs["res_opt"]((l.resolution for l in dets)),
            "eff_opt": kwargs["eff_opt"]((l.efficiency for l in dets)),
        }

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        res_opt: PartialOpt,
        eff_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        vw = cls(volume=volume, res_opt=res_opt, eff_opt=eff_opt, loss_func=loss_func, mu_generator=mu_generator)
        vw.load(name)
        return vw


class PanelVolumeWrapper(AbsVolumeWrapper):
    def __init__(
        self,
        volume: Volume,
        *,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        mu_generator: Optional[AbsMuonGenerator] = None,
        partial_scatter_inferer: Type[AbsScatterBatch] = PanelScatterBatch,
        partial_volume_inferer: Type[AbsVolumeInferer] = PanelX0Inferer,
    ):
        super().__init__(
            volume=volume,
            partial_opts={"xy_pos_opt": xy_pos_opt, "z_pos_opt": z_pos_opt, "xy_span_opt": xy_span_opt},
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferer=partial_scatter_inferer,
            partial_volume_inferer=partial_volume_inferer,
        )

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        all_dets = self.volume.get_detectors()
        dets: List[PanelDetectorLayer] = []
        for d in all_dets:
            if isinstance(d, PanelDetectorLayer):
                dets.append(d)
        self.opts = {
            "xy_pos_opt": kwargs["xy_pos_opt"]((p.xy for l in dets for p in l.panels)),
            "z_pos_opt": kwargs["z_pos_opt"]((p.z for l in dets for p in l.panels)),
            "xy_span_opt": kwargs["xy_span_opt"]((p.xy_span for l in dets for p in l.panels)),
        }

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        vw = cls(
            volume=volume,
            xy_pos_opt=xy_pos_opt,
            z_pos_opt=z_pos_opt,
            xy_span_opt=xy_span_opt,
            loss_func=loss_func,
            mu_generator=mu_generator,
        )
        vw.load(name)
        return vw


class HeatMapVolumeWrapper(AbsVolumeWrapper):
    def __init__(
        self,
        volume: Volume,
        *,
        mu_opt: PartialOpt,
        norm_opt: PartialOpt,
        sig_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        mu_generator: Optional[AbsMuonGenerator] = None,
        partial_scatter_inferer: Type[AbsScatterBatch] = PanelScatterBatch,
        partial_volume_inferer: Type[AbsVolumeInferer] = PanelX0Inferer,
    ):
        super().__init__(
            volume=volume,
            partial_opts={"mu_opt": mu_opt, "norm_opt": norm_opt, "sig_opt": sig_opt},
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferer=partial_scatter_inferer,
            partial_volume_inferer=partial_volume_inferer,
        )

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        all_dets = self.volume.get_detectors()
        dets: List[PanelDetectorLayer] = []
        for d in all_dets:
            if isinstance(d, PanelDetectorLayer):
                dets.append(d)
        self.opts = {
            "mu_opt": kwargs["mu_opt"]((p.mu for l in dets for p in l.panels)),
            "norm_opt": kwargs["norm_opt"]((p.norm for l in dets for p in l.panels)),
            "sig_opt": kwargs["sig_opt"]((p.sig for l in dets for p in l.panels)),
        }

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        mu_opt: PartialOpt,
        norm_opt: PartialOpt,
        sig_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        vw = cls(
            volume=volume,
            mu_opt=mu_opt,
            norm_opt=norm_opt,
            sig_opt=sig_opt,
            loss_func=loss_func,
            mu_generator=mu_generator,
        )
        vw.load(name)
        return vw
