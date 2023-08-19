from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from fastcore.all import Path

from ...utils import x0targs_to_classtargs
from .callback import Callback

r"""
Provides callbacks that return predictions during application of the detectors
"""

__all__ = ["PredHandler", "VolumeTargetPredHandler", "Save2HDF5PredHandler"]


class PredHandler(Callback):
    r"""
    Default callback for predictions. Collects predictions and true voxelwise X0 pairs for a range of volumes and returns them as list of tuples of numpy arrays
    when :meth:`~tomopt.optimisation.callbacks.pred_callbacks.PredHandler.get_preds` is called.
    """

    def on_pred_begin(self) -> None:
        r"""
        Prepares to record predictions
        """

        super().on_pred_begin()
        self.preds: List[Tuple[np.ndarray, np.ndarray]] = []

    def get_preds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        r"""
        Returns:
            List of predicted and target pairs
        """

        return self.preds

    def on_x0_pred_end(self) -> None:
        r"""
        Records predictions and true volume layout for the latest volume
        """

        if self.wrapper.fit_params.state == "test":
            self.preds.append((self.wrapper.fit_params.pred.detach().cpu().numpy(), self.wrapper.volume.get_rad_cube().detach().cpu().numpy()))


class VolumeTargetPredHandler(PredHandler):
    r"""
    Returns the volume target as the target value, rather than the voxel-wise X0s.
    If an x02id lookup is provided, it transforms the target from an X0 value to a material class ID.

    Arguments:
        x02id: optional map from X0 values to class IDs
    """

    def __init__(self, x02id: Optional[Dict[float, int]] = None):
        self.x02id = x02id

    def on_x0_pred_end(self) -> None:
        r"""
        Records predictions and volume target for the latest volume
        """

        if self.wrapper.fit_params.state == "test":
            targ = self.wrapper.volume.target.detach().cpu().numpy()
            if self.x02id is not None:
                targ = x0targs_to_classtargs(targ, self.x02id)
            self.preds.append((self.wrapper.fit_params.pred.detach().cpu().numpy(), targ))


class Save2HDF5PredHandler(VolumeTargetPredHandler):
    r"""
    Saves predictions and targets to an HDF5 file, rather than caching and returning them.
    Samples are written incrementally. Can optionally save volume targets rather than voxel-wise X0 targets
    If an x02id lookup is provided, it transforms the target from an X0 value to a material class ID.

    Arguments:
        path: savename of file to save predictions and targets
        use_volume_target: if True, saves the volume target value instead of the volume X0s
        overwrite: if True will overwrite existing files with the same path, otherwise will append to them
        x02id: optional map from X0 values to class IDs
        compression: optional string representation of any compression to use when saving data
    """

    def __init__(
        self, path: Path, use_volume_target: bool, overwrite: bool = False, x02id: Optional[Dict[float, int]] = None, compression: Optional[str] = "lzf"
    ):
        if isinstance(path, str):
            path = Path(path)
        self.path, self.use_volume_target, self.x02id, self.compression = path, use_volume_target, x02id, compression
        if self.path.exists() and overwrite:
            self.path.unlink()

    def _open_file(self) -> h5py.File:
        r"""
        Returns:
            Save file to write data to
        """

        if self.path.exists():
            return h5py.File(self.path, "r+")
        return h5py.File(self.path, "w")

    def _write_data(self, pred: np.ndarray, targ: np.ndarray) -> None:
        r"""
        Writes a new prediction-target pair to the save file

        Arguments:
            pred: volume predictions to save
            targ: volume targets to save
        """

        with self._open_file() as h5:
            if "preds" in h5:
                ds = h5["preds"]
                ds.resize((ds.shape[0] + 1), axis=0)
                ds[-1] = pred[None].astype("float32")
            else:
                h5.create_dataset(
                    "preds", data=pred[None].astype("float32"), dtype="float32", compression=self.compression, chunks=True, maxshape=(None, *pred.shape)
                )
            if "targs" in h5:
                ds = h5["targs"]
                ds.resize((ds.shape[0] + 1), axis=0)
                ds[-1] = targ[None].astype("float32")
            else:
                h5.create_dataset(
                    "targs", data=targ[None].astype("float32"), dtype="float32", compression=self.compression, chunks=True, maxshape=(None, *targ.shape)
                )

    def on_x0_pred_end(self) -> None:
        r"""
        Records predictions and true volume layout or target for the latest volume
        """

        if self.wrapper.fit_params.state == "test":
            if self.use_volume_target:
                targ = self.wrapper.volume.target.detach().cpu().numpy()
                if self.x02id is not None:
                    targ = x0targs_to_classtargs(targ, self.x02id)
            else:
                targ = self.wrapper.volume.get_rad_cube().detach().cpu().numpy()
            pred = self.wrapper.fit_params.pred.detach().cpu().numpy()
            self._write_data(pred, targ)
