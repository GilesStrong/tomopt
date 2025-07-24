from __future__ import annotations

from typing import List, Optional, Type

from fastprogress import progress_bar

from tomopt.core import PartialOpt
from tomopt.inference import AbsVolumeInferrer, PanelX0Inferrer, ScatterBatch
from tomopt.muon import AbsMuonGenerator, MuonBatch
from tomopt.optimisation.loss.loss import AbsDetectorLoss
from tomopt.optimisation.wrapper import AbsVolumeWrapper
from tomopt.volume import Volume

from .layer import SegmentedPanelDetectorLayer
from .loss import AngularResLoss

_all__ = ["SegmentedPanelVolumeWrapper"]


class SegmentedPanelVolumeWrapper(AbsVolumeWrapper):
    r"""
    An implementation of a wrapper class designed for segmented layer and panel classes.
    """

    def __init__(
        self,
        volume: Volume,
        *,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: Optional[PartialOpt] = None,
        gap_opt: PartialOpt,
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
                "gap_opt": gap_opt,
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
        xy_span_opt: Optional[PartialOpt] = None,
        gap_opt: Optional[PartialOpt] = None,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss],
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        r"""
        Instantiates a new `SegmentedPanelVolumeWrapper` and loads saved detector and optimiser parameters

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
            gap_opt=gap_opt,
            budget_opt=budget_opt,
            loss_func=loss_func,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
            mu_generator=mu_generator,
        )
        vw.load(name)
        return vw

    def _scan_volume(self) -> None:
        r"""
        Passes multiple batches of muons through a single volume, and infers the volume target.
        If in 'train' or 'valid' state, also computes the loss on the detector.
        """
        # Scan volume with muon batches
        self.fit_params.pred = None
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
        self.fit_params.pred = self.fit_params.volume_inferrer.get_prediction()
        for c in self.fit_params.cbs:
            c.on_x0_pred_end()

        # Compute loss for volume
        if self.fit_params.state != "test" and self.loss_func is not None and self.fit_params.pred is not None:
            print(self.loss_func.__class__.__name__)
            loss = (
                self.loss_func(pred=self.fit_params.pred, volume=self.volume, sb=self.fit_params.sb)
                if isinstance(self.loss_func, AngularResLoss)
                else self.loss_func(pred=self.fit_params.pred, volume=self.volume)
            )
            if self.fit_params.loss_val is None:
                self.fit_params.loss_val = loss
            else:
                self.fit_params.loss_val = self.fit_params.loss_val + loss

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        r"""
        Initialises the optimisers by associating them to the detector parameters.

        Arguments:
            kwargs: uninitialised optimisers passed as keyword arguments
        """

        all_dets = self.volume.get_detectors()
        dets: List[SegmentedPanelDetectorLayer] = []
        for d in all_dets:
            if isinstance(d, SegmentedPanelDetectorLayer):
                dets.append(d)
        self.opts = {
            "xy_pos_opt": kwargs["xy_pos_opt"]((p.xy for l in dets for p in l.panels)),
            "z_pos_opt": kwargs["z_pos_opt"]((p.z for l in dets for p in l.panels)),
            "xy_span_opt": kwargs["xy_span_opt"]((p.xy_span for l in dets for p in l.panels)),
            "gap_opt": kwargs["gap_opt"]((l.gap_size for l in dets)),
            # "gap_opt": kwargs["gap_opt"]((l.gap_size for l in dets)),
        }
        if kwargs["budget_opt"] is not None:
            self.opts["budget_opt"] = kwargs["budget_opt"]((p for p in [self.volume.budget_weights]))
