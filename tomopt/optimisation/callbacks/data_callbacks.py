import torch
from torch import Tensor

from tomopt.volume.volume import Volume

from .callback import Callback
from ...muon import MuonBatch
from ...muon.generation import AbsMuonGenerator

__all__ = ["MuonResampler"]


class MuonResampler(Callback):
    """Resmaple muons to only inlcude those which will impact the passive volume at somepoint"""

    def on_mu_batch_begin(self) -> None:
        # TODO Add check for realistic validation
        self.wrapper.fit_params.mu.muons = self.resample(self.wrapper.fit_params.mu.muons, volume=self.wrapper.volume, gen=self.wrapper.mu_generator)

    @staticmethod
    def check_mu_batch(mu: MuonBatch, volume: Volume) -> Tensor:
        mu = mu.copy()
        ok_mask = torch.zeros(len(mu)).bool()
        for l in volume.get_passives():
            mu.propagate(mu.z - l.z)
            ok_mask += mu.get_xy_mask((0, 0), volume.lw)
        return ok_mask

    @staticmethod
    def resample(mus: Tensor, volume: Volume, gen: AbsMuonGenerator) -> Tensor:
        n = len(mus)
        ok_mask = torch.zeros(len(mus)).bool()
        while ok_mask.sum() < n:
            # Check muons
            check_mask = ~ok_mask
            mu = MuonBatch(mus[check_mask], init_z=volume.h)
            tmp_ok_mask = MuonResampler.check_mu_batch(mu, volume=volume)

            # Update and resample N.B. Have to assign to masked tensor rather than double masking full tensor
            resample_mask = ~tmp_ok_mask
            check = mus[check_mask]
            check[resample_mask] = gen(int(resample_mask.sum().item()))
            mus[check_mask] = check
            ok_mask[check_mask] += tmp_ok_mask
        return mus
