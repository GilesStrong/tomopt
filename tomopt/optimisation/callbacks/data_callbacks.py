import torch
from torch import Tensor

from tomopt.volume.volume import Volume

from .callback import Callback
from ...muon import MuonBatch
from ...muon.generation import AbsMuonGenerator

r"""
Provides callbacks designed to affect the data used during fitting/predictions.
"""

__all__ = ["MuonResampler"]


class MuonResampler(Callback):
    r"""
    Resamples muons to only include those which will impact the passive volume at some point, even if they only hit the bottom layer.
    """

    @staticmethod
    def check_mu_batch(mu: MuonBatch, volume: Volume) -> Tensor:
        r"""
        Checks the provided muon batch to determine which muons will impact the passive volume at any point

        Arguments:
            mu: incoming batch of muons
            volume: Volume containing the passive volume to test against

        Returns:
            (muons) Boolean tensor where True indicates that the muon will hit the passive volume
        """

        mu = mu.copy()
        ok_mask = torch.zeros(len(mu)).bool()
        for l in volume.get_passives():
            mu.propagate_dz(mu.z - l.z)
            ok_mask += mu.get_xy_mask((0, 0), volume.lw)
        return ok_mask

    @staticmethod
    def resample(mus: Tensor, volume: Volume, gen: AbsMuonGenerator) -> Tensor:
        r"""
        Resamples muons until all muons will hit the passive volume.

        Arguments:
            mus: xy_p_theta_phi tensor designed to initialise a :class:`~tomopt.muon.muon_batch.MuonBatch`
            volume: Volume containing the passive volume to test against
            gen: Muon generator for sampling replacement muons

        Returns:
            xy_p_theta_phi tensor designed to initialise a :class:`~tomopt.muon.muon_batch.MuonBatch`
        """

        if mus.size(1) == 6:
            mus = mus[:, sorted([MuonBatch.x_dim, MuonBatch.y_dim, MuonBatch.p_dim, MuonBatch.th_dim, MuonBatch.ph_dim])]

        n = len(mus)
        ok_mask = torch.zeros(len(mus)).bool()
        while ok_mask.sum() < n:
            # Check muons
            check_mask = ~ok_mask
            mu = MuonBatch(mus[check_mask], init_z=volume.h, device=volume.device)
            tmp_ok_mask = MuonResampler.check_mu_batch(mu, volume=volume)

            # Update and resample N.B. Have to assign to masked tensor rather than double masking full tensor
            resample_mask = ~tmp_ok_mask
            check = mus[check_mask]
            check[resample_mask] = gen(int(resample_mask.sum().item()))
            mus[check_mask] = check
            ok_mask[check_mask] += tmp_ok_mask
        return mus

    def on_mu_batch_begin(self) -> None:
        r"""
        Resamples muons prior to propagation through the volume such that all muons will hit the passive volume.

        # TODO Add check for realistic validation
        """

        self.wrapper.fit_params.mu.muons[:, sorted([MuonBatch.x_dim, MuonBatch.y_dim, MuonBatch.p_dim, MuonBatch.th_dim, MuonBatch.ph_dim])] = self.resample(
            self.wrapper.fit_params.mu.muons, volume=self.wrapper.volume, gen=self.wrapper.mu_generator
        )
