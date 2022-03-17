from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Dict, List, Union

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .scattering import AbsScatterBatch
from ..volume import VoxelDetectorLayer, PanelDetectorLayer, Volume
from ..core import SCATTER_COEF_A
from ..utils import jacobian

__all__ = ["VoxelX0Inferer", "PanelX0Inferer", "DeepVolumeInferer"]


class AbsVolumeInferer(metaclass=ABCMeta):
    mask_muons = False

    def __init__(self, volume: Volume):
        self.scatter_batches: List[AbsScatterBatch] = []
        self.volume = volume
        self.size, self.lw, self.device = self.volume.passive_size, self.volume.lw, self.volume.device

    def add_scatters(self, scatters: AbsScatterBatch) -> None:
        self.scatter_batches.append(scatters)

    @abstractmethod
    def compute_efficiency(self, scatters: AbsScatterBatch) -> Tensor:
        pass

    @abstractmethod
    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        pass


class AbsX0Inferer(AbsVolumeInferer):
    def __init__(self, volume: Volume):
        super().__init__(volume=volume)
        self.x0_dthetas: List[Optional[Tensor]] = []
        self.x0_dtheta_uncs: List[Optional[Tensor]] = []
        self.x0_dxys: List[Optional[Tensor]] = []
        self.x0_dxy_uncs: List[Optional[Tensor]] = []
        self.efficiencies: List[Tensor] = []
        self.voxel_preds: List[Tensor] = []
        self.voxel_weights: List[Tensor] = []

    def add_scatters(self, scatters: AbsScatterBatch) -> None:
        super().add_scatters(scatters=scatters)
        # Compute muon-wise X0 predictions & efficiencies
        x, u = self.x0_from_dtheta(scatters=scatters)
        self.x0_dthetas.append(x)
        self.x0_dtheta_uncs.append(u)

        x, u = self.x0_from_dxy(scatters=scatters)
        self.x0_dxys.append(x)
        self.x0_dxy_uncs.append(u)

        self.efficiencies.append(self.compute_efficiency(scatters=scatters))

        # Get X0 prediction for all voxels
        p, w = self.get_voxel_x0_preds(
            x0_dtheta=self.x0_dthetas[-1],
            x0_dtheta_unc=self.x0_dtheta_uncs[-1],
            x0_dxy=self.x0_dxys[-1],
            x0_dxy_unc=self.x0_dxy_uncs[-1],
            efficiency=self.efficiencies[-1],
            scatters=scatters,
        )
        self.voxel_preds.append(p)
        self.voxel_weights.append(w)

    @staticmethod
    def _x0_from_dtheta(delta_z: float, mom: Tensor, theta_msc: Tensor, theta_in: Tensor, theta_out: Tensor) -> Tensor:
        cos_theta = (theta_in.cos() + theta_out.cos()) / 2
        return 2 * ((SCATTER_COEF_A / mom) ** 2) * delta_z / (theta_msc.pow(2) * cos_theta)

    @staticmethod
    def _x0_from_dtheta_unc(pred: Tensor, in_vars: Tensor, uncs: Tensor) -> Tensor:
        jac = torch.nan_to_num(jacobian(pred, in_vars)).sum(1)  # Compute dvar/dhit_x

        # Compute unc^2 = unc_x*unc_y*dvar/dhit_x*dvar/dhit_y summing over all x,y inclusive combinations
        idxs = torch.combinations(torch.arange(0, uncs.shape[-1]), with_replacement=True)
        unc_2 = (jac[:, idxs] * uncs[:, idxs]).prod(-1)

        pred_unc = unc_2.sum(-1).sqrt()
        return pred_unc

    def x0_from_dtheta(self, scatters: AbsScatterBatch) -> Tuple[Optional[Tensor], Optional[Tensor]]:
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

        mu = scatters.mu

        if self.mask_muons:  # Scatter mask assumes that muons are prefiltered to only include those which stay inside the volume
            muon_mask = mu.get_xy_mask((0, 0), self.lw)

        scatter_vars, scatter_uncs = [], []
        scatter_vars.append((mu.reco_mom if self.mask_muons is False else mu.reco_mom[muon_mask])[:, None])  # 0
        scatter_uncs.append(torch.zeros(len(scatter_vars[0]), 1))

        scatter_vars.append(scatters.theta_msc)  # 1
        scatter_uncs.append(scatters.theta_msc_unc)

        scatter_vars.append(scatters.theta_in)  # 2
        scatter_uncs.append(scatters.theta_in_unc)

        scatter_vars.append(scatters.theta_out)  # 3
        scatter_uncs.append(scatters.theta_out_unc)

        in_vars = torch.cat(
            scatter_vars,
            dim=-1,
        )

        mom = in_vars[:, 0]
        theta_msc = in_vars[:, 1]
        theta_in = in_vars[:, 2]
        theta_out = in_vars[:, 3]

        uncs = torch.cat(
            scatter_uncs,
            dim=-1,
        )

        pred = self._x0_from_dtheta(delta_z=self.size, mom=mom, theta_msc=theta_msc, theta_in=theta_in, theta_out=theta_out)
        pred_unc = self._x0_from_dtheta_unc(pred=pred, in_vars=in_vars, uncs=uncs)

        return pred, pred_unc

    def x0_from_dxy(self, scatters: AbsScatterBatch) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # TODO: FIX this
        # dxy = torch.sqrt(scatters['dxy'][mask].pow(2).sum(1))
        # dh = dxy/((math.sqrt(2)*torch.cos(scatters['theta_in'][mask].pow(2).sum(1)))+1e-17)
        # theta0 = torch.arcsin(dh/self.size)
        # x0_pred_dxy = (theta0*p/b)**2
        return None, None

    def get_voxel_x0_preds(
        self,
        x0_dtheta: Optional[Tensor],
        x0_dtheta_unc: Optional[Tensor],
        x0_dxy: Optional[Tensor],
        x0_dxy_unc: Optional[Tensor],
        efficiency: Tensor,
        scatters: AbsScatterBatch,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Assign x0 inference to neighbourhood of voxels according to scatter-location uncertainty
        TODO: Implement differing x0 accoring to location via Gaussian spread
        TODO: Don't assume that location uncertainties are uncorrelated
        TODO: Rescale total probability to one (Gaussians extend outside passive volume)
        """

        loc, loc_unc = scatters.location, scatters.location_unc  # loc is (x,y,z)
        # Only consider non-NaN predictions
        mask = ((loc == loc).prod(1) * (loc_unc == loc_unc).prod(1)).bool()
        if x0_dtheta is not None and x0_dtheta_unc is not None:
            mask = (mask * (~x0_dtheta.isnan()) * (~x0_dtheta.isinf()) * (~x0_dtheta_unc.isnan()) * (~x0_dtheta_unc.isinf())).bool()
        if x0_dxy is not None and x0_dxy_unc is not None:
            mask = (mask * (~x0_dxy.isnan()) * (~x0_dxy.isinf()) * (~x0_dxy_unc.isnan()) * (~x0_dxy_unc.isinf())).bool()

        loc, loc_unc, efficiency = loc[mask], loc_unc[mask], efficiency[mask]
        shp_xyz = (
            len(loc),
            round(self.volume.lw.cpu().numpy()[0] / self.volume.passive_size),
            round(self.volume.lw.cpu().numpy()[1] / self.volume.passive_size),
            len(self.volume.get_passives()),
        )
        shp_zxy = shp_xyz[0], shp_xyz[3], shp_xyz[1], shp_xyz[2]

        wpreds, weights = [], []
        for x0, unc in ((x0_dtheta, x0_dtheta_unc), (x0_dxy, x0_dxy_unc)):
            if x0 is None or unc is None:
                continue
            x0, unc = x0[mask], unc[mask]
            x0 = x0[:, None, None, None].expand(shp_zxy).clone()
            coef = efficiency[:, None, None, None].expand(shp_zxy).clone() / ((1e-17) + (unc[:, None, None, None].expand(shp_zxy).clone() ** 2))

            # Gaussian spread
            dists = {}
            for i, d in enumerate(["x", "y", "z"]):
                dists[d] = Normal(loc[:, i], loc_unc[:, i] + 1e-7)  # location uncertainty is sometimes zero, causing errors

            def comp_int(low: Tensor, high: Tensor, dists: Dict[str, Normal]) -> Tensor:
                return torch.prod(torch.stack([dists[d].cdf(high[i]) - dists[d].cdf(low[i]) for i, d in enumerate(dists)]), dim=0)

            prob = (
                torch.stack([comp_int(l, l + self.volume.passive_size, dists) for l in self.volume.edges.unbind()])
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

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # Volume-level X0 prediciton per voxel already made per batch -> combine and reaverage
        if len(self.scatter_batches) == 0:
            print("Warning: unable to scan volume with prescribed number of muons.")
            return None, None
        elif len(self.scatter_batches) == 1:
            return self.voxel_preds[0], self.voxel_weights[0]
        else:
            preds = torch.stack(self.voxel_preds, dim=0)
            weights = torch.stack(self.voxel_weights, dim=0)
            wpred = (preds * weights).sum(0)
            weight = weights.sum(0)
            pred = wpred / weight
            return pred, weight


class VoxelX0Inferer(AbsX0Inferer):
    mask_muons = True

    def compute_efficiency(self, scatters: AbsScatterBatch) -> Tensor:
        r"""
        Does not yet handle more than two detectors per position
        """

        hits = scatters.hits

        dets = self.volume.get_detectors()
        if len(dets) != 4:
            raise NotImplementedError("VoxelX0Inferer.compute_efficiency does not yet handle more than two detectros per position")

        eff = None
        for p, l, i in zip(("above", "above", "below", "below"), dets, (0, 1, 0, 1)):
            if not isinstance(l, VoxelDetectorLayer):
                raise ValueError(f"Detector {l} is not a VoxelDetectorLayer")
            x = l.abs2idx(hits[p]["reco_xy"][:, i])
            e = torch.clamp(l.efficiency[x[:, 0], x[:, 1]], min=0.0, max=1.0)
            if eff is None:
                eff = e
            else:
                eff = eff * e
        if eff is None:
            eff = torch.zeros(0, device=self.device)
        return eff


class PanelX0Inferer(AbsX0Inferer):
    def compute_efficiency(self, scatters: AbsScatterBatch) -> Tensor:
        eff = None
        for pos, hits in enumerate([scatters.above_gen_hits, scatters.below_gen_hits]):
            leff = None
            det = self.volume.get_detectors()[pos]
            if not isinstance(det, PanelDetectorLayer):
                raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
            panel_idxs = det.get_panel_zorder()
            effs = torch.stack([det.panels[i].get_efficiency(hits[:, i, :2]) for i in panel_idxs], dim=0)
            # Muon goes through any combination of at least 2 panels
            p_miss = 1 - effs
            c = torch.combinations(torch.arange(0, len(effs)), r=len(effs) - 1)
            c = c[torch.arange(len(effs) - 1, -1, -1)]  # Reverse order to match panel hit
            p_one_hit = (effs * p_miss[c].prod(1)).sum(0)
            p_no_hit = p_miss.prod(0)
            leff = 1 - p_one_hit - p_no_hit
            if eff is None:
                eff = leff
            else:
                eff = eff * leff  # Muons detected above & below passive volume
        return eff


class DeepVolumeInferer(AbsVolumeInferer):
    def __init__(self, model: Union[torch.jit._script.RecursiveScriptModule, nn.Module], base_inferer: AbsX0Inferer, volume: Volume):
        super().__init__(volume=volume)
        self.model, self.base_inferer = model, base_inferer
        self.voxel_centres = self.volume.centres

        self.in_vars: List[Tensor] = []
        self.in_var_uncs: List[Tensor] = []
        self.efficiencies: List[Tensor] = []
        self.in_var: Optional[Tensor] = None
        self.in_var_unc: Optional[Tensor] = None
        self.efficiency: Optional[Tensor] = None

    def compute_efficiency(self, scatters: AbsScatterBatch) -> Tensor:
        return self.base_inferer.compute_efficiency(scatters=scatters)

    def get_base_predictions(self, scatters: AbsScatterBatch) -> Tuple[Tensor, Tensor]:
        x, u = self.base_inferer.x0_from_dtheta(scatters=scatters)
        return x[:, None], u[:, None]

    def add_scatters(self, scatters: AbsScatterBatch) -> None:
        self.scatter_batches.append(scatters)
        x0, x0_unc = self.get_base_predictions(scatters)
        self.in_vars.append(torch.cat((scatters.dtheta_xy, scatters.dxy, x0, scatters.location), dim=-1))
        self.in_var_uncs.append(torch.cat((scatters.dtheta_xy_unc, scatters.dxy_unc, x0_unc, scatters.location_unc), dim=-1))
        self.efficiencies.append(self.compute_efficiency(scatters=scatters))

    def _build_inputs(self, in_var: Tensor) -> Tensor:
        data = in_var[None, :].repeat_interleave(len(self.voxel_centres), dim=0)
        data[:, :, -3:] -= self.voxel_centres[:, None].repeat_interleave(len(in_var), dim=1)
        data = torch.cat((data, torch.norm(data[:, :, -3:], dim=-1, keepdim=True)), dim=-1)  # dR
        return data

    def _get_weight(self) -> Tensor:
        """Maybe alter this to include resolution/pred uncertainties"""
        return self.efficiency.sum()

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        self.in_var = torch.cat(self.in_vars, dim=0)
        self.in_var_unc = torch.cat(self.in_var_uncs, dim=0)
        self.efficiency = torch.cat(self.efficiencies, dim=0)

        inputs = self._build_inputs(self.in_var)
        pred = self.model(inputs[None])
        weight = self._get_weight()
        return pred, weight
