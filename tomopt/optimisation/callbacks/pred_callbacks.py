from typing import List
import numpy as np

from .callback import Callback

__all__ = ["PredHandler"]


class PredHandler(Callback):
    r"""
    Default callback for predictions. Collects predictions for a range of volumes and returns them as list of numpy arrays
    """

    def on_pred_begin(self) -> None:
        super().__init__()
        self.preds: List[np.ndarray] = []

    def get_preds(self) -> List[np.ndarray]:
        return self.preds

    def on_x0_pred_end(self) -> None:
        if self.wrapper.fit_params.state == "test":
            self.preds.append(self.wrapper.fit_params.pred.detach().cpu().numpy())
