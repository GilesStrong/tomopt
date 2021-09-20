from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..wrapper.volume_wrapper import AbsVolumeWrapper

__all__: List[str] = []


class Callback:
    wrapper: Optional[AbsVolumeWrapper] = None

    def __init__(self) -> None:
        pass

    def set_wrapper(self, wrapper: AbsVolumeWrapper) -> None:
        self.wrapper = wrapper

    def on_train_begin(self) -> None:
        if self.wrapper is None:
            raise AttributeError(f"The wrapper for {type(self).__name__} callback has not been set. Please call set_wrapper before on_train_begin.")

    def on_train_end(self) -> None:
        pass

    def on_epoch_begin(self) -> None:
        pass

    def on_epoch_end(self) -> None:
        pass

    def on_volume_begin(self) -> None:
        pass

    def on_volume_end(self) -> None:
        pass

    def on_volume_batch_begin(self) -> None:
        pass

    def on_volume_batch_end(self) -> None:
        pass

    def on_mu_batch_begin(self) -> None:
        pass

    def on_mu_batch_end(self) -> None:
        pass

    def on_scatter_end(self) -> None:
        pass

    def on_backwards_begin(self) -> None:
        pass

    def on_backwards_end(self) -> None:
        pass

    def on_x0_pred_begin(self) -> None:
        pass

    def on_x0_pred_end(self) -> None:
        pass

    def on_pred_begin(self) -> None:
        if self.wrapper is None:
            raise AttributeError(f"The wrapper for {type(self).__name__} callback has not been set. Please call set_wrapper before on_pred_begin.")

    def on_pred_end(self) -> None:
        pass
