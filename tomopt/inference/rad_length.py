from typing import Tuple, Optional

import torch
from torch import Tensor

from . import ScatterBatch
from ..core import X0, SCATTER_COEF_A

__all__ = ["X0Inferer"]


class X0Inferer:
    def __init__(self, scatters: ScatterBatch, default_pred: Optional[float] = X0["beryllium"]):
        self.scatters, self.default_pred = scatters, default_pred
        self.mu, self.volume, self.hits = self.scatters.mu, self.scatters.volume, self.scatters.hits
        self.size, self.lw = self.volume.size, self.volume.lw
        self.mask = self.scatters.get_scatter_mask()
        if self.default_pred is not None:
            self.default_weight = 1 / (self.default_pred ** 2)
            self.default_weight_t = Tensor([self.default_weight]).to(self.mu.device)
            self.default_pred_t = Tensor([self.default_pred]).to(self.mu.device)

    def x0_from_dtheta(self) -> Tuple[Tensor, Tensor]:
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

        mom = self.mu.reco_mom[self.mu.get_xy_mask(self.lw)][self.mask]
        dtheta = self.scatters.dtheta[self.mask]
        dtheta_unc = self.scatters.dtheta_unc[self.mask]
        theta_xy_in = self.scatters.theta_in[self.mask]
        theta_xy_out = self.scatters.theta_out[self.mask]
        theta_xy_in_unc = self.scatters.theta_in_unc[self.mask]
        theta_xy_out_unc = self.scatters.theta_out_unc[self.mask]

        # Prediction
        theta2 = dtheta.pow(2).sum(1)
        n_x0 = 0.5 * theta2 * ((mom / SCATTER_COEF_A) ** 2)
        theta_in = theta_xy_in.pow(2).sum(1).sqrt()
        theta_out = theta_xy_out.pow(2).sum(1).sqrt()
        cos_theta_in = torch.cos(theta_in)
        cos_theta_out = torch.cos(theta_out)
        cos_mean = (cos_theta_in + cos_theta_out) / 2
        pred = self.size / (n_x0 * cos_mean)

        # Uncertainty TODO probably best check this
        theta2_unc = (2 * dtheta * dtheta_unc).pow(2).sum(1).sqrt()
        n_x0_unc = 0.5 * theta2_unc * ((mom / SCATTER_COEF_A) ** 2)
        theta_in2_unc = (2 * theta_xy_in * theta_xy_in_unc).pow(2).sum(1).sqrt()
        theta_in_unc = 0.5 * theta_in2_unc / theta_in
        theta_out2_unc = (2 * theta_xy_out * theta_xy_out_unc).pow(2).sum(1).sqrt()
        theta_out_unc = 0.5 * theta_out2_unc / theta_out
        cos_theta_in_unc = torch.sin(theta_in) * theta_in_unc
        cos_theta_out_unc = torch.sin(theta_out) * theta_out_unc
        cos_mean_unc = torch.sqrt(cos_theta_in_unc.pow(2) + cos_theta_out_unc.pow(2)) / 2
        inv_cos_mean_unc = cos_mean_unc / cos_mean.pow(2)
        inv_n_x0_unc = n_x0_unc / n_x0.pow(2)
        pred_unc = pred * torch.sqrt((inv_n_x0_unc * n_x0).pow(2) + (inv_cos_mean_unc * cos_mean).pow(2))

        return pred, pred_unc

    def x0_from_dxy(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # TODO: FIX this
        # dxy = torch.sqrt(scatters['dxy'][mask].pow(2).sum(1))
        # dh = dxy/((math.sqrt(2)*torch.cos(scatters['theta_in'][mask].pow(2).sum(1)))+1e-17)
        # theta0 = torch.arcsin(dh/self.size)
        # x0_pred_dxy = (theta0*p/b)**2
        return None, None

    def compute_efficiency(self) -> Tensor:
        eff = None
        for p, l, i in zip(("above", "above", "below", "below"), self.volume.get_detectors(), (0, 1, 0, 1)):
            x = l.abs2idx(self.hits[p]["xy"][:, i][self.mask])
            e = l.efficiency[x[:, 0], x[:, 1]]
            if eff is None:
                eff = e
            else:
                eff = eff * e
        if eff is None:
            eff = torch.zeros(0)
        return eff

    def average_preds(
        self, x0_dtheta: Optional[Tensor], x0_dtheta_unc: Optional[Tensor], x0_dxy: Optional[Tensor], x0_dxy_unc: Optional[Tensor], efficiency: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        TODO: Use location uncertainty to spread muon inferrence over neighbouring voxels around central location
        """

        loc, loc_unc = self.scatters.location[self.mask], self.scatters.location_unc[self.mask]  # noqa F841 will use loc_unc to infer around central voxel
        loc_idx = self.volume.lookup_xyz_coords(loc, passive_only=True)
        idxs = torch.stack((torch.arange(len(loc)).long(), loc_idx[:, 2], loc_idx[:, 0], loc_idx[:, 1]), dim=1)
        shp = (len(loc), len(self.volume.get_passives()), *(self.volume.lw / self.volume.size).long())

        wpreds, weights = [], []
        for x0, unc in ((x0_dtheta, x0_dtheta_unc), (x0_dxy, x0_dxy_unc)):
            if x0 is None or unc is None:
                continue
            coef = efficiency / ((1e-17) + (unc ** 2))
            p = torch.sparse_coo_tensor(idxs.T, x0 * coef, size=shp)
            w = torch.sparse_coo_tensor(idxs.T, coef, size=shp)
            wpreds.append(p.to_dense())
            weights.append(w.to_dense())

        wpred, weight = torch.cat(wpreds, dim=0), torch.cat(weights, dim=0)
        wpred, weight = wpred.sum(0), weight.sum(0)
        pred = wpred / weight
        return pred, weight

    def add_default_pred(self, pred: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor]:
        pred = torch.nan_to_num(pred, self.default_pred)
        pred = torch.where(pred == 0.0, self.default_pred_t, pred)
        weight = torch.nan_to_num(weight, self.default_weight)
        weight = torch.where(weight == 0.0, self.default_weight_t, weight)
        return pred, weight

    def pred_x0(self, inc_default: bool = True) -> Tuple[Tensor, Tensor]:
        x0_dtheta, x0_dtheta_unc = self.x0_from_dtheta()
        x0_dxy, x0_dxy_unc = self.x0_from_dxy()
        eff = self.compute_efficiency()

        pred, weight = self.average_preds(x0_dtheta=x0_dtheta, x0_dtheta_unc=x0_dtheta_unc, x0_dxy=x0_dxy, x0_dxy_unc=x0_dxy_unc, efficiency=eff)
        if inc_default and self.default_pred is not None:
            pred, weight = self.add_default_pred(pred, weight)

        return pred, weight
