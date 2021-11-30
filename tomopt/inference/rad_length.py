from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Dict
import numpy as np

import torch
from torch import Tensor
from torch.distributions import Normal

from .scattering import AbsScatterBatch, PanelScatterBatch, VoxelScatterBatch
from ..volume import VoxelDetectorLayer, PanelDetectorLayer
from ..core import SCATTER_COEF_A
from ..utils import jacobian

__all__ = ["VoxelX0Inferer", "PanelX0Inferer"]


class AbsX0Inferer(metaclass=ABCMeta):
    muon_mask: Optional[Tensor] = None

    def __init__(self, scatters: AbsScatterBatch):
        self.scatters = scatters
        self.mu, self.volume, self.hits = self.scatters.mu, self.scatters.volume, self.scatters.hits
        self.size, self.lw = self.volume.passive_size, self.volume.lw
        self.mask = self.scatters.get_scatter_mask()
        self.device = self.mu.device

    @staticmethod
    def _x0_from_dtheta(delta_z: float, mom: Tensor, dtheta: Tensor, theta_xy_in: Tensor, theta_xy_out: Tensor) -> Tensor:
        theta2 = dtheta.pow(2).sum(1)
        n_x0 = 0.5 * theta2 * ((mom / SCATTER_COEF_A) ** 2)
        theta_in = theta_xy_in.pow(2).sum(1).sqrt()
        theta_out = theta_xy_out.pow(2).sum(1).sqrt()
        cos_theta_in = torch.cos(theta_in)
        cos_theta_out = torch.cos(theta_out)
        cos_mean = (cos_theta_in + cos_theta_out) / 2

        pred = delta_z / (n_x0 * cos_mean)

        if pred.isnan().sum() > 0:
            print(pred)
            raise ValueError("Prediction contains NaN values")

        return pred

    @staticmethod
    def _x0_from_dtheta_unc(
        pred: Tensor, dtheta: Tensor, theta_xy_in: Tensor, theta_xy_out: Tensor, dtheta_unc: Tensor, theta_xy_in_unc: Tensor, theta_xy_out_unc: Tensor
    ) -> Tensor:

        # Compute dvar/dhit_x
        jac = torch.cat([torch.nan_to_num(jacobian(pred, x)).sum(1) for x in [dtheta, theta_xy_in, theta_xy_out]], dim=-1)
        unc = torch.cat([dtheta_unc, theta_xy_in_unc, theta_xy_out_unc], dim=-1)

        # Compute unc^2 = unc_x*unc_y*dvar/dhit_x*dvar/dhit_y summing over all x,y inclusive combinations
        idxs = torch.combinations(torch.arange(0, unc.shape[-1]), with_replacement=True)
        unc_2 = (jac[:, idxs] * unc[:, idxs]).prod(-1)

        pred_unc = unc_2.sum(-1).sqrt()

        if pred_unc.isnan().sum() > 0:
            print(pred_unc)
            raise ValueError("Prediction uncertainties contains NaN values")

        return pred_unc

    def x0_from_dtheta(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        TODO: Debias by considering each voxel on muon paths
        Maybe like:
        Debias dtheta
        dtheta_unc2 = dtheta_unc.pow(2)
        dtheta_dbias = dtheta.pow(2)-dtheta_unc2
        m = [dtheta_dbias < dtheta_unc2]
        dtheta_dbias[m] = dtheta_unc2[m]
        dtheta_dbias = dtheta_dbias.sqrt()
        dtheta = dtheta_dbias
        """

        if self.mask.sum() == 0:
            return None, None

        mom = self.mu.reco_mom[self.mask] if self.muon_mask is None else self.mu.reco_mom[self.muon_mask][self.mask]
        dtheta = self.scatters.dtheta[self.mask]
        dtheta_unc = self.scatters.dtheta_unc[self.mask]
        theta_xy_in = self.scatters.theta_in[self.mask]
        theta_xy_out = self.scatters.theta_out[self.mask]
        theta_xy_in_unc = self.scatters.theta_in_unc[self.mask]
        theta_xy_out_unc = self.scatters.theta_out_unc[self.mask]

        pred = self._x0_from_dtheta(delta_z=self.size, mom=mom, dtheta=dtheta, theta_xy_in=theta_xy_in, theta_xy_out=theta_xy_out)
        pred_unc = self._x0_from_dtheta_unc(
            pred=pred,
            dtheta=dtheta,
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
            dtheta_unc=dtheta_unc,
            theta_xy_in_unc=theta_xy_in_unc,
            theta_xy_out_unc=theta_xy_out_unc,
        )

        return pred, pred_unc

    def x0_from_dxy(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # TODO: FIX this
        # dxy = torch.sqrt(scatters['dxy'][mask].pow(2).sum(1))
        # dh = dxy/((math.sqrt(2)*torch.cos(scatters['theta_in'][mask].pow(2).sum(1)))+1e-17)
        # theta0 = torch.arcsin(dh/self.size)
        # x0_pred_dxy = (theta0*p/b)**2
        return None, None

    @abstractmethod
    def compute_efficiency(self) -> Tensor:
        pass

    def average_preds(
        self, x0_dtheta: Optional[Tensor], x0_dtheta_unc: Optional[Tensor], x0_dxy: Optional[Tensor], x0_dxy_unc: Optional[Tensor], efficiency: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Assign x0 inference to neighbourhood of voxels according to scatter-location uncertainty
        TODO: Implement differing x0 accoring to location via Gaussian spread
        TODO: Don't assume that location uncertainties are uncorrelated
        TODO: Rescale total probability to one (Gaussians extend outside passive volume)
        """

        loc, loc_unc = self.scatters.location[self.mask], self.scatters.location_unc[self.mask]  # loc is (x,y,z)
        shp_xyz = (
            len(loc),
            round(self.volume.lw.cpu().numpy()[0] / self.volume.passive_size),
            round(self.volume.lw.cpu().numpy()[1] / self.volume.passive_size),
            len(self.volume.get_passives()),
        )
        shp_zxy = shp_xyz[0], shp_xyz[3], shp_xyz[1], shp_xyz[2]
        bounds = (
            self.volume.passive_size
            * np.mgrid[
                0 : round(self.volume.lw.detach().cpu().numpy()[0] / self.volume.passive_size) : 1,
                0 : round(self.volume.lw.detach().cpu().numpy()[1] / self.volume.passive_size) : 1,
                round(self.volume.get_passive_z_range()[0].detach().cpu().numpy()[0] / self.volume.passive_size) : round(
                    self.volume.get_passive_z_range()[1].detach().cpu().numpy()[0] / self.volume.passive_size
                ) : 1,
            ]
        )
        bounds[2] = np.flip(bounds[2])  # z is reversed
        int_bounds = torch.tensor(bounds.reshape(3, -1).transpose(-1, -2), device=self.device)

        wpreds, weights = [], []
        for x0, unc in ((x0_dtheta, x0_dtheta_unc), (x0_dxy, x0_dxy_unc)):
            if x0 is None or unc is None:
                continue
            x0 = x0[:, None, None, None].expand(shp_zxy).clone()
            coef = efficiency[:, None, None, None].expand(shp_zxy).clone() / ((1e-17) + (unc[:, None, None, None].expand(shp_zxy).clone() ** 2))

            # Gaussian spread
            dists = {}
            for i, d in enumerate(["x", "y", "z"]):
                dists[d] = Normal(loc[:, i], loc_unc[:, i] + 1e-7)  # location uncertainty is sometimes zero, causing errors

            def comp_int(low: Tensor, high: Tensor, dists: Dict[str, Normal]) -> Tensor:
                return torch.prod(torch.stack([dists[d].cdf(high[i]) - dists[d].cdf(low[i]) for i, d in enumerate(dists)]), dim=0)

            prob = (
                torch.stack([comp_int(l, l + self.volume.passive_size, dists) for l in int_bounds.unbind()])
                .transpose(-1, -2)
                .reshape(shp_xyz)
                .permute(0, 3, 1, 2)
            )  # preds are (z,x,y)  TODO: vmap this? Might not be possible since it tries to run Normal.cdf batchwise.
            prob = prob + 1e-15  # Sometimes probability is zero
            coef = coef * prob

            wpreds.append(x0 * coef)
            weights.append(coef)

        if len(wpreds) == 0:
            return None, None
        wpred, weight = torch.cat(wpreds, dim=0), torch.cat(weights, dim=0)
        wpred, weight = wpred.sum(0), weight.sum(0)
        pred = wpred / weight

        if weight.isnan().sum() > 0:
            print(weight)
            raise ValueError("Weight contains NaN values")
        if (weight == 0).sum() > 0:
            print(weight)
            raise ValueError("Weight contains values at zero")
        if pred.isnan().sum() > 0:
            print(pred)
            raise ValueError("Prediction contains NaN values")

        return pred, weight

    def pred_x0(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        x0_dtheta, x0_dtheta_unc = self.x0_from_dtheta()
        x0_dxy, x0_dxy_unc = self.x0_from_dxy()
        eff = self.compute_efficiency()

        pred, weight = self.average_preds(x0_dtheta=x0_dtheta, x0_dtheta_unc=x0_dtheta_unc, x0_dxy=x0_dxy, x0_dxy_unc=x0_dxy_unc, efficiency=eff)

        return pred, weight


class VoxelX0Inferer(AbsX0Inferer):
    def __init__(self, scatters: VoxelScatterBatch):
        super().__init__(scatters=scatters)
        self.muon_mask = self.mu.get_xy_mask(
            (0, 0), self.lw
        )  # Scatter mask assumes that muons are prefiltered to only include those which stay inside the volume

    def compute_efficiency(self) -> Tensor:
        r"""
        Does not yet handle more than two detectors per position
        """

        dets = self.volume.get_detectors()
        if len(dets) != 4:
            raise NotImplementedError("VoxelX0Inferer.compute_efficiency does not yet handle more than two detectros per position")

        eff = None
        for p, l, i in zip(("above", "above", "below", "below"), dets, (0, 1, 0, 1)):
            if not isinstance(l, VoxelDetectorLayer):
                raise ValueError(f"Detector {l} is not a VoxelDetectorLayer")
            x = l.abs2idx(self.hits[p]["reco_xy"][:, i][self.mask])
            e = torch.clamp(l.efficiency[x[:, 0], x[:, 1]], min=0.0, max=1.0)
            if eff is None:
                eff = e
            else:
                eff = eff * e
        if eff is None:
            eff = torch.zeros(0, device=self.device)
        return eff


class PanelX0Inferer(AbsX0Inferer):
    def __init__(self, scatters: PanelScatterBatch):
        super().__init__(scatters=scatters)

    def compute_efficiency(self) -> Tensor:
        eff = None
        for pos, hits in enumerate([self.scatters.above_gen_hits, self.scatters.below_gen_hits]):
            leff = None
            det = self.volume.get_detectors()[pos]
            if not isinstance(det, PanelDetectorLayer):
                raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
            panel_idxs = det.get_panel_zorder()
            effs = torch.stack([det.panels[i].get_efficiency(hits[i][self.mask, :2]) for i in panel_idxs], dim=0)
            for r in range(2, len(effs) + 1):  # Muon goes through any combination of at least 2 panels
                c = torch.combinations(torch.arange(0, len(effs)), r=r)
                e = effs[c].prod(1).sum(0)
                if leff is None:
                    leff = e
                else:
                    leff = leff + e
            if eff is None:
                eff = leff
            else:
                eff = eff * leff  # Muons detected above & below passive volume
        return eff
