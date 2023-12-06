from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from fastcore.all import Path
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleProgressBar, NBProgressBar, ProgressBar
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ...core import DEVICE, PartialOpt
from ...inference import AbsVolumeInferrer, PanelX0Inferrer, ScatterBatch
from ...muon import AbsMuonGenerator, MuonBatch, MuonGenerator2016
from ...optimisation.loss.loss import AbsDetectorLoss
from ...volume import AbsDetectorLayer, PanelDetectorLayer, Volume
from ..callbacks import (
    Callback,
    CyclicCallback,
    EvalMetric,
    MetricLogger,
    PredHandler,
    WarmupCallback,
)
from ..data import PassiveYielder

__all__ = ["FitParams", "AbsVolumeWrapper", "PanelVolumeWrapper", "HeatMapVolumeWrapper"]

r"""
Provides wrapper classes for optimising detectors and other quality-of-life methods

FitParams and AbsVolumeWrapper are modified versions of the FitParams in LUMIN (https://github.com/GilesStrong/lumin/blob/v0.7.2/lumin/nn/models/abs_model.py#L16)
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

Usage is compatible with the AGPL licence under-which TomOpt is distributed.
Stated changes: adaption of FitParams to pass type-checking, heavy adaptation of Model to be suitable for task specific training
"""


class FitParams:
    r"""
    Data class used for storing all aspects of optimisation and prediction when working with
    :class:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper`

    Arguments:
        kwargs: objects to be stored
    """

    volume_inferrer: Optional[AbsVolumeInferrer] = None
    state: Optional[str] = None
    pred: Optional[Tensor] = None
    inv_weight: Optional[Tensor] = None
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
    warmup_cbs: Optional[List[WarmupCallback]] = None
    metric_log: Optional[MetricLogger] = None
    metric_cbs: Optional[List[EvalMetric]] = None
    passive_bar: Optional[Union[NBProgressBar, ConsoleProgressBar]] = None
    device: torch.device = DEVICE
    skip_opt_step: bool = False

    def __init__(self, **kwargs: Any) -> None:
        r"""
        Stores any keyword arguments as an attribute
        """

        self.__dict__.update(kwargs)


class AbsVolumeWrapper(metaclass=ABCMeta):
    r"""
    Abstract base class for optimisation volume wrappers.
    Inheriting classes will need to override :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._build_opt`
    according to the detector parameters that need to be optimised.

    Volume wrappers are designed to contain a :class:`~tomopt.volume.volume.Volume` and provide means of optimising the detectors it contains,
    via their :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.fit` method.

    Wrappers also provide for various quality-of-life methods, such as saving and loading detector configurations,
    and computing predictions with a fixed detector (:meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.predict`)

    Fitting of a detector proceeds as training and validation epochs, each of which contains multiple batches of passive volumes.
    For each volume in a batch, the loss is evaluated using multiple batches of muons.
    The whole loop is:

    1. for epoch in `n_epochs`:
        A. `loss` = 0
        B. for `p`, `passive` in enumerate(`trn_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
            c. Compute loss based on precision and cost, and add to `loss`
            d. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. Backpropagate `loss` and update detector parameters
                iii. `loss` = 0
            e. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break

        C. `val_loss` = 0
        D. for `p`, `passive` in enumerate(`val_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
                iii. Compute loss based on precision and cost, and add to `val_loss`
            c. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        E. `val_loss` = `val_loss`/`p`

    In implementation, the loop is broken up into several functions:
        - :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._fit_epoch` runs one full epoch of volumes and updates for both training and validation
        - :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volumes` runs over all training/validation volumes, updating parameters when necessary
        - :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volume` irradiates a single volume with muons multiple batches, and computes the loss for that volume

    The optimisation and prediction loops are supported by a stateful callback mechanism.
    The base callback is :class:`~tomopt.optimisation.callbacks.callback.Callback`, which can interject at various points in the loops.
    All aspects of the optimisation and prediction are stored in a :class:`~tomopt.optimisation.wrapper.volume_wrapper.FitParams` data class,
    since the callbacks are also stored there, and the callbacks have a reference to the wrapper, they are able to read/write to the `FitParams` and be
    aware of other callbacks that are running.

    Accounting for the interjection calls (`on_*_begin` & `on_*_end`), the full optimisation loop is:

    1. Associate callbacks with wrapper (`set_wrapper`)
    2. `on_train_begin`
    3. for epoch in `n_epochs`:
        A. `state` = "train"
        B. `on_epoch_begin`
        C. for `p`, `passive` in enumerate(`trn_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. load `passive` into passive volume
            c. `on_volume_begin`
            d. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            e. `on_x0_pred_begin`
            f. Compute overall x0 prediction
            g. `on_x0_pred_end`
            h. Compute loss based on precision and cost, and add to `loss`
            i. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
                iii. Zero parameter gradients
                iv. `on_backwards_begin`
                v. Backpropagate `loss` and compute parameter gradients
                vi. `on_backwards_end`
                vii. Update detector parameters
                viii. Ensure detector parameters are within physical boundaries (`AbsDetectorLayer.conform_detector`)
                viv. `loss` = 0
            j. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        D. `on_epoch_end`
        E. `state` = "valid"
        F. `on_epoch_begin`
        G. for `p`, `passive` in enumerate(`val_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. `on_volume_begin`
            c. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            d. `on_x0_pred_begin`
            e. Compute overall x0 prediction
            f. `on_x0_pred_end`
            g. Compute loss based on precision and cost, and add to `loss`
            h. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
            i. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        H. `on_epoch_end`
    4. `on_train_end`

    Arguments:
        volume: the volume containing the detectors to be optimised
        partial_opts: dictionary of uninitialised optimisers to be associated with the detector parameters, via `_build_opt`
        loss_func: Optional loss function (required if planning to optimise the detectors)
        partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
        partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
        mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
    """

    opts: Dict[str, Optimizer]

    def __init__(
        self,
        volume: Volume,
        *,
        partial_opts: Dict[str, PartialOpt],
        loss_func: Optional[AbsDetectorLoss] = None,
        partial_scatter_inferrer: Type[ScatterBatch],
        partial_volume_inferrer: Type[AbsVolumeInferrer],
        mu_generator: Optional[AbsMuonGenerator] = None,
    ):
        self.volume, self.loss_func = volume, loss_func
        if mu_generator is None:
            mu_generator = MuonGenerator2016.from_volume(volume)
        self.mu_generator = mu_generator
        self.partial_scatter_inferrer, self.partial_volume_inferrer = partial_scatter_inferrer, partial_volume_inferrer
        self.device = self.volume.device
        self._build_opt(**partial_opts)
        self.parameters = self.volume.parameters

    @abstractmethod
    def _build_opt(self, **kwargs: PartialOpt) -> None:
        r"""
        Inheriting classes should override this method to initialise the optimisers by associating them to the detector parameters. e.g.:
        self.opts = {'res_opt': res_opt((l.resolution for l in self.volume.get_detectors())),
                     'eff_opt': eff_opt((l.efficiency for l in self.volume.get_detectors()))}

        Arguments:
            kwargs: uninitialised optimisers passed as keyword arguments
        """

        pass

    def get_detectors(self) -> List[AbsDetectorLayer]:
        r"""
        Returns:
            A list of all :class:`~tomopt.volume.layer.AbsDetectorLayer` s in the volume, in the order of `layers` (normally decreasing z position)
        """

        return self.volume.get_detectors()

    def save(self, name: str) -> None:
        r"""
        Saves the volume and optimiser parameters to a file.

        Arguments:
            name: savename for the file
        """

        torch.save({"volume": self.volume.state_dict(), **{k: v.state_dict() for k, v in self.opts.items()}}, str(name))

    def load(self, name: str) -> None:
        r"""
        Loads saved volume and optimiser parameters from a file.

        Arguments:
            name: file to load
        """

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
        r"""
        Runs the fitting loop for the detectors over a specified number of epochs, using the provided volumes or volume generators.
        The optimisation loop can be supported by callbacks.

        Arguments:
            n_epochs: number of epochs to run for (a training and validation epoch only counts as one 'epoch)
            passive_bs: number of passive volumes to use per volume batch (detector updates occur after every volume batch in training mode)
            n_mu_per_volume: number of muons to use in total when inferring the target of a single volume
            mu_bs: number of muons to use per muon batch; multiple muon batches will be used until `n_mu_per_volume` is reached
            trn_passives: passive volumes to use for optimising the detector
            val_passives: optional passive volumes to use for evaluating the detector
            cbs: optional list of callbacks to use
            cb_savepath: location where callbacks should write/save any information

        Returns:
            The list of callbacks
        """

        if cbs is None:
            cbs = []
        sorted_cbs = self._sort_cbs(cbs)

        self.fit_params = FitParams(
            cbs=cbs,
            cyclic_cbs=sorted_cbs["cyclic_cbs"],
            warmup_cbs=sorted_cbs["warmup_cbs"],
            metric_log=sorted_cbs["metric_log"][0],
            metric_cbs=sorted_cbs["metric_cbs"],
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
        r"""
        Uses the detectors to predict the provided volumes
        The prediction loop can be supported by callbacks.

        Arguments:
            passives: passive volumes to predict
            n_mu_per_volume: number of muons to use in total when inferring the target of a single volume
            mu_bs: number of muons to use per muon batch; multiple muon batches will be used until n_mu_per_volume is reached
            pred_cb: the prediction callback to use for recording predictions
            cbs: optional list of callbacks to use
            cb_savepath: location where callbacks should write/save any information

        Returns:
            The object returned by the `pred_cb`'s `get_preds` method
        """

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
        r"""
        Returns the learning rate of the specified optimiser.

        Arguments:
            opt: string name of the optimiser requested

        Returns:
            The learning rate of the specified optimiser
        """

        return self.opts[opt].param_groups[0]["lr"]

    def set_opt_lr(self, lr: float, opt: str) -> None:
        r"""
        Sets the learning rate of the specified optimiser.

        Arguments:
            lr: new learning rate for the optimiser
            opt: string name of the optimiser requested
        """

        self.opts[opt].param_groups[0]["lr"] = lr

    def get_opt_mom(self, opt: str) -> float:
        r"""
        Returns the momentum coefficient/beta_1 of the specified optimiser.

        Arguments:
            opt: string name of the optimiser requested

        Returns:
            The momentum coefficient/beta_1 of the specified optimiser
        """

        if "betas" in self.opts[opt].param_groups[0]:
            return self.opts[opt].param_groups[0]["betas"][0]
        else:
            return self.opts[opt].param_groups[0]["momentum"]

    def set_opt_mom(self, mom: float, opt: str) -> None:
        r"""
        Sets the learning rate of the specified optimiser.

        Arguments:
            mom: new momentum coefficient/beta_1 for the optimiser
            opt: string name of the optimiser requested
        """

        if "betas" in self.opts[opt].param_groups[0]:
            self.opts[opt].param_groups[0]["betas"] = (mom, self.opts[opt].param_groups[0]["betas"][1])
        else:
            self.opts[opt].param_groups[0]["momentum"] = mom

    @staticmethod
    def _sort_cbs(cbs: List[Callback]) -> Dict[str, Optional[List[Callback]]]:
        r"""
        Sorts callbacks into lists according to their type and whether other callbacks might need to be aware of them.

        Arguments:
            cbs: all callbacks being used

        Returns:
            cyclical callbacks: list of callbacks that act over a range of epochs
            logger callbacks: list of callbacks that record telemetry of the optimisation process
            metric: list of callbacks that compute performance metrics about the detector
        """

        sorted_cbs: Dict[str, Optional[List[Callback]]] = defaultdict(list)
        n_warmup = 0
        for c in cbs:
            if isinstance(c, CyclicCallback):
                sorted_cbs["cyclic_cbs"].append(c)  # CBs that might prevent a wrapper from stopping training due to a hyper-param cycle
            if isinstance(c, WarmupCallback):
                sorted_cbs["warmup_cbs"].append(c)  # CBs that might act on a warmup cycle whilst optimisation is frozen
                n_warmup += c.n_warmup
            if isinstance(c, MetricLogger):
                sorted_cbs["metric_log"].append(c)  # CB that logs losses and eval_metrics
            if isinstance(c, EvalMetric):
                sorted_cbs["metric_cbs"].append(c)  # CBs that computes additional performance metrics
        print(f'{len(sorted_cbs["warmup_cbs"])} warmup callbacks found, with a total warmup period of {n_warmup} epochs.')
        if len(sorted_cbs["metric_log"]) == 0:
            sorted_cbs["metric_log"].append(None)
        return sorted_cbs

    def _fit_epoch(self) -> None:
        r"""
        Runs through one training epoch (state = 'train'), using :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volumes`.
        If validation volumes are present, will then run through one validation epoch (state = 'valid'),
        again using :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volumes`.
        """

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

    def _scan_volumes(self, passives: PassiveYielder) -> None:
        r"""
        Scans all volumes by splitting them into volume batches.
        Each volume is scanned via using :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volume`.
        After each volume batch, if in 'train' state,the detector parameters will be updated using the loss of the volume batch and the optimisers.
            If not enough volumes remain to form a complete batch and in 'train' state, the method will end prematurely.
        """

        if self.fit_params.state == "test":
            self.fit_params.passive_bar = master_bar(passives)
        for i, (passive, target) in enumerate(self.fit_params.passive_bar if self.fit_params.state == "test" else passives):
            self.fit_params.volume_id = i
            if self.fit_params.state != "test" and i % self.fit_params.passive_bs == 0:  # Volume batch start
                self.fit_params.loss_val = None
                for c in self.fit_params.cbs:
                    c.on_volume_batch_begin()

            self.volume.load_properties(passive, target)
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
                    if self.fit_params.mean_loss is not None and not self.fit_params.skip_opt_step:
                        for o in self.opts.values():
                            o.step()
                    for c in self.fit_params.cbs:
                        c.on_step_end()
                    for d in self.volume.get_detectors():
                        d.conform_detector()

                if len(passives) - (i + 1) < self.fit_params.passive_bs:
                    break

    def _scan_volume(self) -> None:
        r"""
        Passes multiple batches of muons through a single volume, and infers the volume target.
        If in 'train' or 'valid' state, also computes the loss on the detector.
        """

        # Scan volume with muon batches
        self.fit_params.pred, self.fit_params.inv_weight = None, None
        if self.fit_params.state != "test":
            muon_bar = progress_bar(range(self.fit_params.n_mu_per_volume // self.fit_params.mu_bs), display=False, leave=False)
        else:
            muon_bar = progress_bar(range(self.fit_params.n_mu_per_volume // self.fit_params.mu_bs), parent=self.fit_params.passive_bar)
        self.fit_params.volume_inferrer = self.partial_volume_inferrer(volume=self.volume)
        for _ in muon_bar:
            self.fit_params.mu = MuonBatch(self.mu_generator(self.fit_params.mu_bs), init_z=self.volume.h, device=self.fit_params.device)
            for c in self.fit_params.cbs:
                c.on_mu_batch_begin()
            self.volume(self.fit_params.mu)
            self.fit_params.sb = self.partial_scatter_inferrer(mu=self.fit_params.mu, volume=self.volume)
            for c in self.fit_params.cbs:
                c.on_scatter_end()
            self.fit_params.volume_inferrer.add_scatters(self.fit_params.sb)
            for c in self.fit_params.cbs:
                c.on_mu_batch_end()

        # Predict volume based on all muon batches
        for c in self.fit_params.cbs:
            c.on_x0_pred_begin()
        self.fit_params.pred, self.fit_params.inv_weight = self.fit_params.volume_inferrer.get_prediction()
        for c in self.fit_params.cbs:
            c.on_x0_pred_end()

        # Compute loss for volume
        if self.fit_params.state != "test" and self.loss_func is not None and self.fit_params.pred is not None:
            loss = self.loss_func(pred=self.fit_params.pred, inv_pred_weight=self.fit_params.inv_weight, volume=self.volume)
            if self.fit_params.loss_val is None:
                self.fit_params.loss_val = loss
            else:
                self.fit_params.loss_val = self.fit_params.loss_val + loss


class PanelVolumeWrapper(AbsVolumeWrapper):
    r"""
    Volume wrapper for volumes with :class:`~tomopt.volume.panel.DetectorPanel`-based detectors.

    Volume wrappers are designed to contain a :class:`~tomopt.volume.volume.Volume` and provide means of optimising the detectors it contains,
    via their :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.fit` method.

    Wrappers also provide for various quality-of-life methods, such as saving and loading detector configurations,
    and computing predictions with a fixed detector (:meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.predict`)

    Fitting of a detector proceeds as training and validation epochs, each of which contains multiple batches of passive volumes.
    For each volume in a batch, the loss is evaluated using multiple batches of muons.
    The whole loop is:

    1. for epoch in `n_epochs`:
        A. `loss` = 0
        B. for `p`, `passive` in enumerate(`trn_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
            c. Compute loss based on precision and cost, and add to `loss`
            d. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. Backpropagate `loss` and update detector parameters
                iii. `loss` = 0
            e. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break

        C. `val_loss` = 0
        D. for `p`, `passive` in enumerate(`val_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
                iii. Compute loss based on precision and cost, and add to `val_loss`
            c. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        E. `val_loss` = `val_loss`/`p`

    In implementation, the loop is broken up into several functions:
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._fit_epoch` runs one full epoch of volumes
            and updates for both training and validation
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volumes` runs over all training/validation volumes,
            updating parameters when necessary
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volume` irradiates a single volume with muons multiple batches,
            and computes the loss for that volume

    The optimisation and prediction loops are supported by a stateful callback mechanism.
    The base callback is :class:`~tomopt.optimisation.callbacks.callback.Callback`, which can interject at various points in the loops.
    All aspects of the optimisation and prediction are stored in a :class:`~tomopt.optimisation.wrapper.volume_wrapper.FitParams` data class,
    since the callbacks are also stored there, and the callbacks have a reference to the wrapper, they are able to read/write to the `FitParams` and be
    aware of other callbacks that are running.

    Accounting for the interjection calls (`on_*_begin` & `on_*_end`), the full optimisation loop is:

    1. Associate callbacks with wrapper (`set_wrapper`)
    2. `on_train_begin`
    3. for epoch in `n_epochs`:
        A. `state` = "train"
        B. `on_epoch_begin`
        C. for `p`, `passive` in enumerate(`trn_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. load `passive` into passive volume
            c. `on_volume_begin`
            d. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            e. `on_x0_pred_begin`
            f. Compute overall x0 prediction
            g. `on_x0_pred_end`
            h. Compute loss based on precision and cost, and add to `loss`
            i. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
                iii. Zero parameter gradients
                iv. `on_backwards_begin`
                v. Backpropagate `loss` and compute parameter gradients
                vi. `on_backwards_end`
                vii. Update detector parameters
                viii. Ensure detector parameters are within physical boundaries (`AbsDetectorLayer.conform_detector`)
                viv. `loss` = 0
            j. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        D. `on_epoch_end`
        E. `state` = "valid"
        F. `on_epoch_begin`
        G. for `p`, `passive` in enumerate(`val_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. `on_volume_begin`
            c. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            d. `on_x0_pred_begin`
            e. Compute overall x0 prediction
            f. `on_x0_pred_end`
            g. Compute loss based on precision and cost, and add to `loss`
            h. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
            i. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        H. `on_epoch_end`
    4. `on_train_end`

    Arguments:
        volume: the volume containing the detectors to be optimised
        xy_pos_opt: uninitialised optimiser to be used for adjusting the xy position of panels
        z_pos_opt: uninitialised optimiser to be used for adjusting the z position of panels
        xy_span_opt: uninitialised optimiser to be used for adjusting the xy size of panels
        budget_opt: optional uninitialised optimiser to be used for adjusting the fractional assignment of budget to the panels
        loss_func: optional loss function (required if planning to optimise the detectors)
        partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
        partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
        mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
    """

    def __init__(
        self,
        volume: Volume,
        *,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: PartialOpt,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss] = None,
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ):
        super().__init__(
            volume=volume,
            partial_opts={
                "xy_pos_opt": xy_pos_opt,
                "z_pos_opt": z_pos_opt,
                "xy_span_opt": xy_span_opt,
                "budget_opt": budget_opt,
            },
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
        )

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: PartialOpt,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss],
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        r"""
        Instantiates a new `PanelVolumeWrapper` and loads saved detector and optimiser parameters

        Arguments:
            name: file name with saved detector and optimiser parameters
            volume: the volume containing the detectors to be optimised
            xy_pos_opt: uninitialised optimiser to be used for adjusting the xy position of panels
            z_pos_opt: uninitialised optimiser to be used for adjusting the z position of panels,
            xy_span_opt: uninitialised optimiser to be used for adjusting the xy size of panels,
            budget_opt: optional uninitialised optimiser to be used for adjusting the fractional assignment of budget to the panels
            loss_func: optional loss function (required if planning to optimise the detectors)
            partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
            partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
            mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
        """

        vw = cls(
            volume=volume,
            xy_pos_opt=xy_pos_opt,
            z_pos_opt=z_pos_opt,
            xy_span_opt=xy_span_opt,
            budget_opt=budget_opt,
            loss_func=loss_func,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
            mu_generator=mu_generator,
        )
        vw.load(name)
        return vw

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        r"""
        Initialises the optimisers by associating them to the detector parameters.

        Arguments:
            kwargs: uninitialised optimisers passed as keyword arguments
        """

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
        if kwargs["budget_opt"] is not None:
            self.opts["budget_opt"] = kwargs["budget_opt"]((p for p in [self.volume.budget_weights]))


class HeatMapVolumeWrapper(AbsVolumeWrapper):
    r"""
    Volume wrapper for volumes with :class:`~tomopt.volume.heatmap.DetectorHeatMap`-based detectors.

    Volume wrappers are designed to contain a :class:`~tomopt.volume.volume.Volume` and provide means of optimising the detectors it contains,
    via their :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.fit` method.

    Wrappers also provide for various quality-of-life methods, such as saving and loading detector configurations,
    and computing predictions with a fixed detector (:meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.predict`)

    Fitting of a detector proceeds as training and validation epochs, each of which contains multiple batches of passive volumes.
    For each volume in a batch, the loss is evaluated using multiple batches of muons.
    The whole loop is:

    1. for epoch in `n_epochs`:
        A. `loss` = 0
        B. for `p`, `passive` in enumerate(`trn_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
            c. Compute loss based on precision and cost, and add to `loss`
            d. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. Backpropagate `loss` and update detector parameters
                iii. `loss` = 0
            e. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break

        C. `val_loss` = 0
        D. for `p`, `passive` in enumerate(`val_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
                iii. Compute loss based on precision and cost, and add to `val_loss`
            c. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        E. `val_loss` = `val_loss`/`p`

    In implementation, the loop is broken up into several functions:
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._fit_epoch` runs one full epoch of volumes
            and updates for both training and validation
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volumes` runs over all training/validation volumes,
            updating parameters when necessary
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volume` irradiates a single volume with muons multiple batches,
            and computes the loss for that volume

    The optimisation and prediction loops are supported by a stateful callback mechanism.
    The base callback is :class:`~tomopt.optimisation.callbacks.callback.Callback`, which can interject at various points in the loops.
    All aspects of the optimisation and prediction are stored in a :class:`~tomopt.optimisation.wrapper.volume_wrapper.FitParams` data class,
    since the callbacks are also stored there, and the callbacks have a reference to the wrapper, they are able to read/write to the `FitParams` and be
    aware of other callbacks that are running.

    Accounting for the interjection calls (`on_*_begin` & `on_*_end`), the full optimisation loop is:

    1. Associate callbacks with wrapper (`set_wrapper`)
    2. `on_train_begin`
    3. for epoch in `n_epochs`:
        A. `state` = "train"
        B. `on_epoch_begin`
        C. for `p`, `passive` in enumerate(`trn_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. load `passive` into passive volume
            c. `on_volume_begin`
            d. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            e. `on_x0_pred_begin`
            f. Compute overall x0 prediction
            g. `on_x0_pred_end`
            h. Compute loss based on precision and cost, and add to `loss`
            i. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
                iii. Zero parameter gradients
                iv. `on_backwards_begin`
                v. Backpropagate `loss` and compute parameter gradients
                vi. `on_backwards_end`
                vii. Update detector parameters
                viii. Ensure detector parameters are within physical boundaries (`AbsDetectorLayer.conform_detector`)
                viv. `loss` = 0
            j. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        D. `on_epoch_end`
        E. `state` = "valid"
        F. `on_epoch_begin`
        G. for `p`, `passive` in enumerate(`val_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. `on_volume_begin`
            c. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            d. `on_x0_pred_begin`
            e. Compute overall x0 prediction
            f. `on_x0_pred_end`
            g. Compute loss based on precision and cost, and add to `loss`
            h. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
            i. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        H. `on_epoch_end`
    4. `on_train_end`

    Arguments:
        volume: the volume containing the detectors to be optimised
        mu_opt: uninitialised optimiser to be used for adjusting the xy position of Gaussians
        norm_opt: uninitialised optimiser to be used for adjusting the normalisation of Gaussians
        sig_opt: uninitialised optimiser to be used for adjusting the scale of Gaussians
        z_pos_opt: uninitialised optimiser to be used for adjusting the z position of panels
        loss_func: optional loss function (required if planning to optimise the detectors)
        partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
        partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
        mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
    """

    def __init__(
        self,
        volume: Volume,
        *,
        mu_opt: PartialOpt,
        norm_opt: PartialOpt,
        sig_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ):
        super().__init__(
            volume=volume,
            partial_opts={"mu_opt": mu_opt, "norm_opt": norm_opt, "sig_opt": sig_opt, "z_pos_opt": z_pos_opt},
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
        )

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        mu_opt: PartialOpt,
        norm_opt: PartialOpt,
        sig_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        loss_func: Optional[AbsDetectorLoss],
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        r"""
        Instantiates a new `HeatMapVolumeWrapper` and loads saved detector and optimiser parameters

        Arguments:
            name: file name with saved detector and optimiser parameters
            volume: the volume containing the detectors to be optimised
            mu_opt: uninitialised optimiser to be used for adjusting the xy position of Gaussians
            norm_opt: uninitialised optimiser to be used for adjusting the normalisation of Gaussians
            sig_opt: uninitialised optimiser to be used for adjusting the scale of Gaussians
            z_pos_opt: uninitialised optimiser to be used for adjusting the z position of panels
            loss_func: optional loss function (required if planning to optimise the detectors)
            partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
            partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
            mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
        """

        vw = cls(
            volume=volume,
            mu_opt=mu_opt,
            norm_opt=norm_opt,
            sig_opt=sig_opt,
            z_pos_opt=z_pos_opt,
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
        )
        vw.load(name)
        return vw

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        r"""
        Initialises the optimisers by associating them to the detector parameters.

        Arguments:
            kwargs: uninitialised optimisers passed as keyword arguments
        """

        all_dets = self.volume.get_detectors()
        dets: List[PanelDetectorLayer] = []
        for d in all_dets:
            if isinstance(d, PanelDetectorLayer):
                dets.append(d)
        self.opts = {
            "mu_opt": kwargs["mu_opt"]((p.mu for l in dets for p in l.panels)),
            "norm_opt": kwargs["norm_opt"]((p.norm for l in dets for p in l.panels)),
            "sig_opt": kwargs["sig_opt"]((p.sig for l in dets for p in l.panels)),
            "z_pos_opt": kwargs["z_pos_opt"]((p.z for l in dets for p in l.panels)),
        }
