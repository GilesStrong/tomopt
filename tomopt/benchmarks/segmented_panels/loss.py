import torch
from torch import Tensor

from tomopt.inference import ScatterBatch
from tomopt.optimisation.loss import AbsDetectorLoss
from tomopt.volume import Volume


class AngularResLoss(AbsDetectorLoss):

    def forward(self, pred: Tensor, volume: Volume, sb: ScatterBatch) -> Tensor:  # type: ignore[override]
        r"""
        Computes the loss for the predictions of a single volume using the current state of the detector

        Arguments:
            pred: the predictions from the inference
            volume: Volume containing the passive volume that was being predicted and the detector being optimised

        Returns:
            The loss for the predictions and detector
        """

        self.sub_losses = {}
        self.sub_losses["error"] = self._get_inference_loss(pred, volume, sb)
        self.sub_losses["cost"] = self._get_cost_loss(volume)
        return self.sub_losses["error"] + self.sub_losses["cost"]

    def _get_inference_loss(self, pred: Tensor, volume: Volume, sb: ScatterBatch) -> Tensor:  # type: ignore[override]
        gen_above, gen_below = sb.hits["above"]["gen_xyz"], sb.hits["below"]["gen_xyz"]
        reco_above, reco_below = sb.hits["above"]["reco_xyz"], sb.hits["below"]["reco_xyz"]
        uncs_above, uncs_below = sb.hits["above"]["unc_xyz"], sb.hits["below"]["unc_xyz"]

        # For reconstructed tracks
        reco_above_vecs, _ = ScatterBatch.get_muon_trajectory(reco_above, uncs_above, volume.lw)
        reco_below_vecs, _ = ScatterBatch.get_muon_trajectory(reco_below, uncs_below, volume.lw)

        # For generated tracks
        # Note: gen hits are "perfect" so pass dummy uncertainties (small or constant)
        dummy_uncs = torch.ones_like(gen_above) * 1e-8

        gen_above_vecs, _ = ScatterBatch.get_muon_trajectory(gen_above, dummy_uncs, volume.lw)
        gen_below_vecs, _ = ScatterBatch.get_muon_trajectory(gen_below, dummy_uncs, volume.lw)

        def zenith_angle_deviation(v_reco: Tensor, v_gen: Tensor) -> Tensor:
            """
            Computes the difference between the zenith angles of two batches of vectors with respect to the z-axis.

            Arguments:
                v_reco, v_gen: (n_muons, 3) tensors representing the reconstructed and generated vectors.

            Returns:
                (n_muons,) tensor of differences in zenith angles in degrees
            """
            # Normalize vectors
            v_reco = v_reco / v_reco.norm(dim=1, keepdim=True)
            v_gen = v_gen / v_gen.norm(dim=1, keepdim=True)

            # Compute zenith angles (cosine of the zenith angle is the z-component of the normalized vector)
            zenith_reco = torch.acos(v_reco[:, 2])  # acos of z-component
            zenith_gen = torch.acos(v_gen[:, 2])

            # Difference in zenith angles
            delta_zenith = zenith_reco - zenith_gen

            # Convert from radians to degrees
            delta_zenith_degrees = delta_zenith * 180 / torch.pi

            return delta_zenith_degrees

        diff_zenith_incoming = zenith_angle_deviation(reco_above_vecs, gen_above_vecs)
        diff_zenith_outgoing = zenith_angle_deviation(reco_below_vecs, gen_below_vecs)

        return (diff_zenith_incoming.std() + diff_zenith_outgoing.std()) / 2
