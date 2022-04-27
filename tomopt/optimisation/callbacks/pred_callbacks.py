from typing import List, Tuple, Dict
import numpy as np

from .callback import Callback
from ...utils import x0targs_to_classtargs

__all__ = ["PredHandler", "ClassPredHandler"]


class PredHandler(Callback):
    r"""
    Default callback for predictions. Collects predictions for a range of volumes and returns them as list of numpy arrays
    """

    def on_pred_begin(self) -> None:
        super().on_pred_begin()
        self.preds: List[Tuple[np.ndarray, np.ndarray]] = []

    def get_preds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self.preds

    def on_x0_pred_end(self) -> None:
        if self.wrapper.fit_params.state == "test":
            self.preds.append((self.wrapper.fit_params.pred.detach().cpu().numpy(), self.wrapper.volume.get_rad_cube().detach().cpu().numpy()))


class ClassPredHandler(PredHandler):
    r"""
    :class:`.PredHandler` which returns the volume target as the target value, rather than the voxel-wise X0s.
    """

    def __init__(self, x02id: Dict[float, int]):
        self.x02id = x02id

    def on_x0_pred_end(self) -> None:
        if self.wrapper.fit_params.state == "test":
            targ = self.wrapper.volume.target.detach().cpu().numpy()
            targ = x0targs_to_classtargs(targ, self.x02id)
            self.preds.append((self.wrapper.fit_params.pred.detach().cpu().numpy(), targ))
