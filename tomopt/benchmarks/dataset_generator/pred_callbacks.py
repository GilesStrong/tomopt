from typing import List, Tuple

import torch
from torch import Tensor

from tomopt.optimisation.callbacks import Callback

"""_DESCRIPTION_

Defines customized callbacks inheriting from the tomopt callbacks.
To be used in prediction mode to record hits and poca points for each scatter batch and for each volume.
"""

__all__ = ["PredHitRecord", "PredPocaRecord"]


class PredHitRecord(Callback):
    r"""
    Extended hit recorder. Used in prediction mode.
    Records reconstructed hits, generated hits, and hit uncertainties,
    separated into above and below panels.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize lists to accumulate hits for the volumes in the
        # passive batch.
        self.reco_hits_batch: List[Tensor] = []
        self.gen_hits_batch: List[Tensor] = []
        self.hit_uncs_batch: List[Tensor] = []

    def on_volume_begin(self) -> None:
        """
        Initializes the lists for the new volume.
        This method is called at the beginning of each volume prediction.
        """
        self.reco_hits: List[Tensor] = []
        self.gen_hits: List[Tensor] = []
        self.hit_uncs: List[Tensor] = []

    def on_scatter_end(self) -> None:
        """
        Saves the hits, generated hits, and uncertainties of the latest muon batch.
        This method is called at the end of each scatter event.
        """
        self.n_above = self.wrapper.fit_params.sb.n_hits_above

        # Record reconstructed hits
        reco_hits = self.wrapper.fit_params.sb._reco_hits.detach().cpu().clone()
        self.reco_hits.append(reco_hits)

        # Record generated hits
        gen_hits = self.wrapper.fit_params.sb._gen_hits.detach().cpu().clone()
        self.gen_hits.append(gen_hits)

        # Record hit uncertainties
        hit_uncs = self.wrapper.fit_params.sb._hit_uncs.detach().cpu().clone()
        self.hit_uncs.append(hit_uncs)

    def on_x0_pred_end(self) -> None:
        """
        Called at the end of the volume prediction.
        This method is called after all muon batches have been processed.
        """
        # Concatenate hits from all batches
        reco_hits = torch.cat(self.reco_hits, dim=0)
        gen_hits = torch.cat(self.gen_hits, dim=0)
        hit_uncs = torch.cat(self.hit_uncs, dim=0)

        self.reco_hits_batch.append(reco_hits)
        self.gen_hits_batch.append(gen_hits)
        self.hit_uncs_batch.append(hit_uncs)

    def split_above_below(self, record: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Splits the record into above and below hits.

        Arguments:
            record: (n_muons, n_panels (above + below), xyz)

        Returns:
            Tuple of (above_hits, below_hits)
        """
        above_hits = record[:, : self.n_above]
        below_hits = record[:, self.n_above :]
        return above_hits, below_hits


class PredPocaRecord(Callback):
    r"""
    Scattering poca point recorder. Used in prediction mode.
    Records poca xyz positions and uncertainties, and their scattering
    angles and their uncertainties.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize lists to accumulate hits for the volumes in the
        # passive batch.
        self.poca_xyz_batch: List[Tensor] = []
        self.poca_xyz_unc_batch: List[Tensor] = []
        self.poca_theta_mcs_batch: List[Tensor] = []
        self.poca_theta_mcs_unc_batch: List[Tensor] = []

    def on_volume_begin(self) -> None:
        """
        Initializes the lists for the new volume.
        This method is called at the beginning of each volume prediction.
        """
        self.poca_xyz: List[Tensor] = []
        self.poca_xyz_unc: List[Tensor] = []
        self.poca_theta_mcs: List[Tensor] = []
        self.poca_theta_mcs_unc: List[Tensor] = []

    def on_scatter_end(self) -> None:
        """
        Saves the hits, generated hits, and uncertainties of the latest muon batch.
        This method is called at the end of each scatter event.
        """
        self.n_above = self.wrapper.fit_params.sb.n_hits_above

        # Record poca xyz positions
        poca_xyz = self.wrapper.fit_params.sb.poca_xyz.detach().cpu().clone()
        self.poca_xyz.append(poca_xyz)

        # Record poca xyz uncertainties
        poca_xyz_unc = self.wrapper.fit_params.sb.poca_xyz_unc.detach().cpu().clone()
        self.poca_xyz_unc.append(poca_xyz_unc)

        # Record scattering angles
        poca_theta_mcs = self.wrapper.fit_params.sb.total_scatter.detach().cpu().clone()
        self.poca_theta_mcs.append(poca_theta_mcs)

        # Record scattering angles uncertainties
        poca_theta_mcs_unc = self.wrapper.fit_params.sb.total_scatter_unc.detach().cpu().clone()
        self.poca_theta_mcs_unc.append(poca_theta_mcs_unc)

    def on_x0_pred_end(self) -> None:
        """
        Called at the end of the volume prediction.
        This method is called after all muon batches have been processed.
        """
        # Concatenate hits from all batches
        poca_xyz = torch.cat(self.poca_xyz, dim=0)
        self.poca_xyz_batch.append(poca_xyz)

        poca_xyz_unc = torch.cat(self.poca_xyz_unc, dim=0)
        self.poca_xyz_unc_batch.append(poca_xyz_unc)

        poca_theta_mcs = torch.cat(self.poca_theta_mcs, dim=0)
        self.poca_theta_mcs_batch.append(poca_theta_mcs)

        poca_theta_mcs_unc = torch.cat(self.poca_theta_mcs_unc, dim=0)
        self.poca_theta_mcs_unc_batch.append(poca_theta_mcs_unc)

    def split_above_below(self, record: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Splits the record into above and below hits.

        Arguments:
            record: (n_muons, n_panels (above + below), xyz)

        Returns:
            Tuple of (above_hits, below_hits)
        """
        above_hits = record[:, : self.n_above]
        below_hits = record[:, self.n_above :]
        return above_hits, below_hits
