from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..wrapper.volume_wrapper import AbsVolumeWrapper

r"""
Implements the base class from which all callback should inherit.
"""

__all__: List[str] = []


class Callback:
    r"""
    Implements the base class from which all callback should inherit.
    Callbacks are used as part of the fitting, validation, and prediction methods of :class:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper`.
    They can interject at various points, but by default do nothing. Please check in the :class:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper`
    to see when exactly their interjections are called.

    When writing new callbacks, the :class:`~tomopt.optimisation.wrapper.volume_wrapper.VolumeWrapper`
    they are associated with will be their `wrapper` attribute.
    Their wrapper will have a `fit_params` attribute (:class:`~tomopt.optimisation.wrapper.volume_wrapper.FitParams`) which is a data-class style object.
    It contains all the objects associated with the fit and predictions, including other callbacks.
    Callback interjections should read/write to `wrapper.fit_params`, rather than returning values.

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
    """

    wrapper: Optional[AbsVolumeWrapper] = None

    def __init__(self) -> None:
        pass

    def set_wrapper(self, wrapper: AbsVolumeWrapper) -> None:
        r"""
        Arguments:
            wrapper: Volume wrapper to associate with the callback
        """

        self.wrapper = wrapper

    def on_train_begin(self) -> None:
        r"""
        Runs when detector fitting begins.
        """

        if self.wrapper is None:
            raise AttributeError(f"The wrapper for {type(self).__name__} callback has not been set. Please call set_wrapper before on_train_begin.")

    def on_epoch_begin(self) -> None:
        r"""
        Runs when a new training or validations epoch begins.
        """

        pass

    def on_volume_batch_begin(self) -> None:
        r"""
        Runs when a new batch of passive volume layouts is begins.
        """

        pass

    def on_volume_begin(self) -> None:
        r"""
        Runs when a new passive volume layout is loaded.
        """

        pass

    def on_mu_batch_begin(self) -> None:
        r"""
        Runs when a new batch of muons begins.
        """

        pass

    def on_scatter_end(self) -> None:
        r"""
        Runs when a scatters for the latest muon batch have been computed, but not yet added to the volume inferrer.
        """

        pass

    def on_mu_batch_end(self) -> None:
        r"""
        Runs when a batch muons ends and scatters have been added to the volume inferrer.
        """

        pass

    def on_x0_pred_begin(self) -> None:
        r"""
        Runs when the all the muons for a volume have propagated, and the volume inferrer is about to make its final prediction.
        """

        pass

    def on_x0_pred_end(self) -> None:
        r"""
        Runs after the volume inferrer has made its final prediction, but before the loss is computed.
        """

        pass

    def on_volume_end(self) -> None:
        r"""
        Runs when a passive volume layout has been predicted.
        """

        pass

    def on_volume_batch_end(self) -> None:
        r"""
        Runs when a batch of passive volume layouts is ends.
        """

        pass

    def on_backwards_begin(self) -> None:
        r"""
        Runs when the loss for a batch of passive volumes has been computed, but not yet backpropagated.
        """

        pass

    def on_backwards_end(self) -> None:
        r"""
        Runs when the loss for a batch of passive volumes has been backpropagated, but parameters have not yet been updated.
        """

        pass

    def on_epoch_end(self) -> None:
        r"""
        Runs when a training or validations epoch ends.
        """

        pass

    def on_train_end(self) -> None:
        r"""
        Runs when detector fitting ends.
        """

        pass

    def on_pred_begin(self) -> None:
        r"""
        Runs when the wrapper is about to begin in prediction mode.
        """

        if self.wrapper is None:
            raise AttributeError(f"The wrapper for {type(self).__name__} callback has not been set. Please call set_wrapper before on_pred_begin.")

    def on_pred_end(self) -> None:
        r"""
        Runs when the wrapper has finished in prediction mode.
        """

        pass
