from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..wrapper.volume_wrapper import AbsVolumeWrapper

r"""
Implements the base class from which all callback shhould inherit.
"""

__all__: List[str] = []


class Callback:
    r"""
    Implements the base class from which all callback shhould inherit.
    Callbacks are used as part of the fitting, validation, and prediction methods of :class:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper`.
    They can interject at various points, but by default do nothing. Please check in the :class:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper`
    to see when exactly their interjections are called.

    When writing new callbacks, the :class:`~tomopt.optimisation.wrapper.volume_wrapper.VolumeWrapper`
    they are assoicated with will be their `wrapper` attribute.
    Their wrapper will have a `fit_params` attribute (:class:`~tomopt.optimisation.wrapper.volume_wrapper.FitParams`) which is a data-class style object.
    It contains all the objects associated with the fit and predictions, including other callbacks.
    Callback interjections should read/write to `wrapper.fit_params`, rather than returning values.
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
        Runs when a scatters for the latest muon batch have been computed, but not yet added to the volume inferer.
        """

        pass

    def on_mu_batch_end(self) -> None:
        r"""
        Runs when a batch muons ends and scatters have been added to the volume inferer.
        """

        pass

    def on_x0_pred_begin(self) -> None:
        r"""
        Runs when the all the muons for a volume have propagated, and the volume inferer is about to make its final prediciton.
        """

        pass

    def on_x0_pred_end(self) -> None:
        r"""
        Runs after the volume inferer has made its final prediciton, but before the loss is computed.
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
