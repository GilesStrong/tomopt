from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Dict
import numpy as np

import torch
from torch import Tensor
from torch.distributions import Normal

from .scattering import AbsScatterBatch, PanelScatterBatch, VoxelScatterBatch
from ..volume import VoxelDetectorLayer, PanelDetectorLayer
from ..core import X0, SCATTER_COEF_A
from ..utils import jacobian

__all__ = ["VoxelX0Inferer", "PanelX0Inferer"]


class AbsX0Inferer(metaclass=ABCMeta):
    def __init__(self, scatters: AbsScatterBatch, default_pred: Optional[float] = X0["beryllium"], use_gaussian_spread: bool = True):
        self.scatters, self.default_pred, self.use_gaussian_spread = scatters, default_pred, use_gaussian_spread
        self.mu, self.volume, self.hits = self.scatters.mu, self.scatters.volume, self.scatters.hits
        self.size, self.lw = self.volume.passive_size, self.volume.lw
        self.mask = self.scatters.get_scatter_mask()
        self.device = self.mu.device
        if self.default_pred is not None:
            self.default_weight = 1 / (self.default_pred ** 2)
            self.default_weight_t = torch.tensor([self.default_weight], device=self.device)
            self.default_pred_t = torch.tensor([self.default_pred], device=self.device)
        self.average_preds = self.average_preds_gaussian if self.use_gaussian_spread else self.average_preds_single

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

        mom = self.mu.reco_mom[self.mu.get_xy_mask((0, 0), self.lw)][self.mask]
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

        unc2_sum = None
        vals = [dtheta, theta_xy_in, theta_xy_out]
        uncs = [dtheta_unc, theta_xy_in_unc, theta_xy_out_unc]
        for i, (vi, unci) in enumerate(zip(vals, uncs)):
            for j, (vj, uncj) in enumerate(zip(vals, uncs)):
                if j < i:
                    continue
                dv_dx_2 = jacobian(pred, vi).sum(1) * jacobian(pred, vj).sum(1) if i != j else jacobian(pred, vi).sum(1) ** 2  # Muons, pred, unc_xyz
                unc_2 = (dv_dx_2 * unci * uncj).sum(1)  # Muons, pred
                if unc2_sum is None:
                    unc2_sum = unc_2
                else:
                    unc2_sum = unc2_sum + unc_2
        pred_unc = torch.sqrt(unc2_sum)

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

    def average_preds_single(
        self, x0_dtheta: Optional[Tensor], x0_dtheta_unc: Optional[Tensor], x0_dxy: Optional[Tensor], x0_dxy_unc: Optional[Tensor], efficiency: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Assign entirety of x0 inference to single voxels per muon
        """

        loc, loc_unc = self.scatters.location[self.mask], self.scatters.location_unc[self.mask]  # noqa F841 will use loc_unc to infer around central voxel
        loc_idx = self.volume.lookup_passive_xyz_coords(loc)
        idxs = torch.stack((torch.arange(len(loc)).long(), loc_idx[:, 2], loc_idx[:, 0], loc_idx[:, 1]), dim=1)
        shp = (len(loc), len(self.volume.get_passives()), *(self.volume.lw / self.volume.passive_size).long())

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

    def average_preds_gaussian(
        self, x0_dtheta: Optional[Tensor], x0_dtheta_unc: Optional[Tensor], x0_dxy: Optional[Tensor], x0_dxy_unc: Optional[Tensor], efficiency: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Assign x0 inference to neighbourhood of voxels according to scatter-location uncertainty
        TODO: Implement differing x0 accoring to location via Gaussian spread
        TODO: Don't assume that location uncertainties are uncorrelated
        """

        loc, loc_unc = self.scatters.location[self.mask], self.scatters.location_unc[self.mask]  # loc is (x,y,z)
        shp_xyz = (
            len(loc),
            round(self.volume.lw.cpu().numpy()[0] / self.volume.passive_size),
            round(self.volume.lw.cpu().numpy()[1] / self.volume.passive_size),
            len(self.volume.get_passives()),
        )
        shp_zxy = shp_xyz[0], shp_xyz[3], shp_xyz[1], shp_xyz[2]
        int_bounds = (
            self.volume.passive_size
            * np.mgrid[
                0 : round(self.volume.lw.detach().cpu().numpy()[0] / self.volume.passive_size) : 1,
                0 : round(self.volume.lw.detach().cpu().numpy()[1] / self.volume.passive_size) : 1,
                round(self.volume.get_passive_z_range()[0].detach().cpu().numpy()[0] / self.volume.passive_size) : round(
                    self.volume.get_passive_z_range()[1].detach().cpu().numpy()[0] / self.volume.passive_size
                ) : 1,
            ]
        )
        int_bounds[2] = np.flip(int_bounds[2])  # z is reversed
        int_bounds = torch.tensor(int_bounds.reshape(3, -1).transpose(-1, -2), device=self.device)

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
        return pred, weight

    def add_default_pred(self, pred: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor]:
        pred = torch.nan_to_num(pred, self.default_pred)
        pred = torch.where(pred == 0.0, self.default_pred_t.type(pred.type()), pred)
        weight = torch.nan_to_num(weight, self.default_weight)
        weight = torch.where(weight == 0.0, self.default_weight_t.type(weight.type()), weight)
        return pred, weight

    def pred_x0(self, inc_default: bool = True) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        x0_dtheta, x0_dtheta_unc = self.x0_from_dtheta()
        x0_dxy, x0_dxy_unc = self.x0_from_dxy()
        eff = self.compute_efficiency()

        pred, weight = self.average_preds(x0_dtheta=x0_dtheta, x0_dtheta_unc=x0_dtheta_unc, x0_dxy=x0_dxy, x0_dxy_unc=x0_dxy_unc, efficiency=eff)
        if inc_default and self.default_pred is not None and pred is not None:
            pred, weight = self.add_default_pred(pred, weight)

        return pred, weight


class VoxelX0Inferer(AbsX0Inferer):
    def __init__(self, scatters: VoxelScatterBatch, default_pred: Optional[float] = X0["beryllium"], use_gaussian_spread: bool = True):
        super().__init__(scatters=scatters, default_pred=default_pred, use_gaussian_spread=use_gaussian_spread)

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
    def __init__(self, scatters: PanelScatterBatch, default_pred: Optional[float] = X0["beryllium"], use_gaussian_spread: bool = True):
        super().__init__(scatters=scatters, default_pred=default_pred, use_gaussian_spread=use_gaussian_spread)

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
