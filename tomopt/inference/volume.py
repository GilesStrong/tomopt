from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Dict, List, Type, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Normal

from .scattering import AbsScatterBatch
from ..volume import VoxelDetectorLayer, PanelDetectorLayer, Volume
from ..core import SCATTER_COEF_A
from ..utils import jacobian

__all__ = ["VoxelX0Inferer", "PanelX0Inferer", "DenseBlockClassifierFromX0s"]  # "DeepVolumeInferer", "WeightedDeepVolumeInferer"]


class AbsVolumeInferer(metaclass=ABCMeta):
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
    _n_mu: Optional[int] = None
    _muon_scatter_vars: Optional[Tensor] = None  # (mu, vars)
    _muon_scatter_var_uncs: Optional[Tensor] = None  # (mu, vars)
    _muon_probs_per_voxel_zxy: Optional[Tensor] = None  # (mu, z,x,y)
    _muon_efficiency: Tensor = None  # (mu, eff)
    _vox_zxy_x0_preds: Optional[Tensor] = None  # (z,x,y)
    _vox_zxy_x0_pred_uncs: Optional[Tensor] = None  # (z,x,y)
    _var_order_szs = [("poca", 3), ("tot_scatter", 1), ("theta_in", 1), ("theta_out", 1), ("mom", 1)]

    def __init__(self, volume: Volume):
        super().__init__(volume=volume)
        self._set_var_dimensions()
        # set shapes
        self.shp_xyz = [
            round(self.lw.cpu().numpy()[0] / self.size),
            round(self.lw.cpu().numpy()[1] / self.size),
            len(self.volume.get_passives()),
        ]
        self.shp_zxy = [self.shp_xyz[2], self.shp_xyz[0], self.shp_xyz[1]]

    def _set_var_dimensions(self) -> None:
        # Configure dimension indexing
        dims = {}
        i = 0
        for var, sz in self._var_order_szs:
            dims[var] = slice(i, i + sz)
            i += sz
        self._poca_dim = dims["poca"]
        self._tot_scatter_dim = dims["tot_scatter"]
        self._theta_in_dim = dims["theta_in"]
        self._theta_out_dim = dims["theta_out"]
        self._mom_dim = dims["mom"]

    def _combine_scatters(self) -> None:
        vals: Dict[str, Tensor] = {}
        uncs: Dict[str, Tensor] = {}

        if len(self.scatter_batches) == 0:
            raise ValueError("No scatter batches have been added")

        vals["poca"] = torch.cat([sb.poca_xyz for sb in self.scatter_batches], dim=0)
        uncs["poca"] = torch.cat([sb.poca_xyz_unc for sb in self.scatter_batches], dim=0)
        vals["tot_scatter"] = torch.cat([sb.total_scatter for sb in self.scatter_batches], dim=0)
        uncs["tot_scatter"] = torch.cat([sb.total_scatter_unc for sb in self.scatter_batches], dim=0)
        vals["theta_in"] = torch.cat([sb.theta_in for sb in self.scatter_batches], dim=0)
        uncs["theta_in"] = torch.cat([sb.theta_in_unc for sb in self.scatter_batches], dim=0)
        vals["theta_out"] = torch.cat([sb.theta_out for sb in self.scatter_batches], dim=0)
        uncs["theta_out"] = torch.cat([sb.theta_out_unc for sb in self.scatter_batches], dim=0)
        vals["mom"] = torch.cat([sb.mu.mom[:, None] for sb in self.scatter_batches], dim=0)
        uncs["mom"] = torch.zeros_like(vals["mom"])

        mask = torch.ones(len(vals["poca"])).bool()
        for var_sz in self._var_order_szs:
            mask *= ~(vals[var_sz[0]].isnan().any(1))
            mask *= ~(vals[var_sz[0]].isinf().any(1))
            mask *= ~(uncs[var_sz[0]].isnan().any(1))
            mask *= ~(uncs[var_sz[0]].isinf().any(1))

        self._muon_scatter_vars = torch.cat([vals[var_sz[0]][mask] for var_sz in self._var_order_szs], dim=1)  # (mu, vars)
        self._muon_scatter_var_uncs = torch.cat([uncs[var_sz[0]][mask] for var_sz in self._var_order_szs], dim=1)  # (mu, vars)
        self._muon_efficiency = torch.cat([self.compute_efficiency(scatters=sb) for sb in self.scatter_batches], dim=0)[mask]  # (mu, eff)
        self._n_mu = len(self._muon_scatter_vars)

    @staticmethod
    def x0_from_scatters(deltaz: float, total_scatter: Tensor, theta_in: Tensor, theta_out: Tensor, mom: Tensor) -> Tensor:
        cos_theta = (theta_in.cos() + theta_out.cos()) / 2
        return ((SCATTER_COEF_A / mom) ** 2) * deltaz / (total_scatter.pow(2) * cos_theta)

    def get_voxel_zxy_x0_pred_uncs(self) -> Tensor:
        jac = torch.nan_to_num(jacobian(self.vox_zxy_x0_preds, self._muon_scatter_vars))  # Compute dx0/dvar  (z,x,y,mu,var)
        unc = self._muon_scatter_var_uncs
        unc = torch.where(torch.isinf(unc), torch.tensor([0]).type(unc.type()), unc)[None, None, None]  # (1,1,1,mu,var)
        jac, unc = jac.reshape(jac.shape[0], jac.shape[1], jac.shape[2], -1), unc.reshape(unc.shape[0], unc.shape[1], unc.shape[2], -1)  # (z,x,y,mu*var)

        # Compute unc^2 = unc_x*unc_y*dx0/dx*dx0/dy summing over all x,y inclusive combinations
        idxs = torch.combinations(torch.arange(0, unc.shape[-1]), with_replacement=True)
        unc_2 = (jac[:, :, :, idxs] * unc[:, :, :, idxs]).prod(-1)  # (z,x,y,N)

        pred_unc = unc_2.sum(-1).sqrt()  # (z,x,y)
        return pred_unc

    @staticmethod
    def _weighted_rms(x: Tensor, wgt: Tensor) -> Tensor:
        return ((x.square() * wgt).sum(0) / wgt.sum(0)).sqrt()

    @staticmethod
    def _weighted_mean(x: Tensor, wgt: Tensor) -> Tensor:
        return (x * wgt).sum(0) / wgt.sum(0)

    def get_voxel_zxy_x0_preds(self) -> Tensor:
        r"""
        Assign x0 inference to neighbourhood of voxels according to scatter-poca_xyz uncertainty
        TODO: Implement differing x0 accoring to poca_xyz via Gaussian spread
        TODO: Don't assume that poca_xyz uncertainties are uncorrelated
        """

        # Compute variable weights per voxel per muon, variable weights applied to squared variables, therefore use error propagation
        vox_prob_eff_wgt = self.muon_efficiency.reshape(self.n_mu, 1, 1, 1) * self.muon_probs_per_voxel_zxy  # (mu,z,x,y)

        mu_tot_scatter2_var = ((2 * self.muon_total_scatter * self.muon_total_scatter_unc) ** 2).reshape(self.n_mu, 1, 1, 1)
        mu_theta_in2_var = ((2 * self.muon_theta_in * self.muon_theta_in_unc) ** 2).reshape(self.n_mu, 1, 1, 1)
        mu_theta_out2_var = ((2 * self.muon_theta_out * self.muon_theta_out_unc) ** 2).reshape(self.n_mu, 1, 1, 1)
        mu_mom2_var = ((2 * self.muon_mom * self.muon_mom_unc) ** 2).reshape(self.n_mu, 1, 1, 1)

        # Compute weighted RMS of scatter variables per voxel
        vox_tot_total_scatter = self._weighted_rms(
            self.muon_total_scatter.reshape(self.n_mu, 1, 1, 1), torch.nan_to_num(vox_prob_eff_wgt / mu_tot_scatter2_var)
        )  # (z,x,y)
        vox_theta_in = self._weighted_rms(self.muon_theta_in.reshape(self.n_mu, 1, 1, 1), torch.nan_to_num(vox_prob_eff_wgt / mu_theta_in2_var))
        vox_theta_out = self._weighted_rms(self.muon_theta_out.reshape(self.n_mu, 1, 1, 1), torch.nan_to_num(vox_prob_eff_wgt / mu_theta_out2_var))
        vox_mom = self._weighted_rms(
            self.muon_mom.reshape(self.n_mu, 1, 1, 1), torch.nan_to_num(vox_prob_eff_wgt / (1 if (mu_mom2_var == 0).all() else mu_mom2_var))
        )  # Muon momentum may not have uncertainty

        vox_x0_preds = self.x0_from_scatters(
            deltaz=self.size, total_scatter=vox_tot_total_scatter, theta_in=vox_theta_in, theta_out=vox_theta_out, mom=vox_mom
        )  # (z,x,y)

        if vox_x0_preds.isnan().any():
            raise ValueError("Prediction contains NaN values")

        return vox_x0_preds

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if len(self.scatter_batches) == 0:
            print("Warning: unable to scan volume with prescribed number of muons.")
            return None, None
        return self.vox_zxy_x0_preds, self.vox_zxy_inv_weights

    @property
    def vox_zxy_x0_preds(self) -> Tensor:
        if self._vox_zxy_x0_preds is None:
            self._vox_zxy_x0_preds = self.get_voxel_zxy_x0_preds()
            self._vox_zxy_x0_pred_uncs = None
        return self._vox_zxy_x0_preds

    @property
    def vox_zxy_x0_pred_uncs(self) -> Tensor:
        if self._vox_zxy_x0_pred_uncs is None:
            self._vox_zxy_x0_pred_uncs = self.get_voxel_zxy_x0_pred_uncs()
        return self._vox_zxy_x0_pred_uncs

    @property
    def vox_zxy_inv_weights(self) -> Tensor:
        return (
            None  # self.muon_efficiency.reshape(self.n_mu, 1, 1, 1) / (self.vox_zxy_x0_pred_uncs**2)  # These divide the loss per voxel: vox_loss / inv_weight
        )

    @property
    def muon_probs_per_voxel_zxy(self) -> Tensor:  # (mu,z,x,y)
        if self._muon_probs_per_voxel_zxy is None:
            # Gaussian spread
            dists = {}
            for i, d in enumerate(["x", "y", "z"]):
                dists[d] = Normal(self.muon_poca_xyz[:, i], self.muon_poca_xyz_unc[:, i] + 1e-7)  # poca_xyz uncertainty is sometimes zero, causing errors

            def comp_int(low: Tensor, high: Tensor, dists: Dict[str, Normal]) -> Tensor:
                return torch.prod(torch.stack([dists[d].cdf(high[i]) - dists[d].cdf(low[i]) for i, d in enumerate(dists)], dim=0), dim=0)

            probs = (
                torch.stack([comp_int(l, l + self.volume.passive_size, dists) for l in self.volume.edges.unbind()])
                .transpose(-1, -2)  # prob, mu --> mu, prob
                .reshape([self.n_mu] + self.shp_xyz)  # mu, x, y, z
                .permute(0, 3, 1, 2)  # mu, z, x, y
            )
            self._muon_probs_per_voxel_zxy = probs + 1e-15  # Sometimes probability is zero
        return self._muon_probs_per_voxel_zxy

    @property
    def n_mu(self) -> int:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._n_mu

    @property
    def muon_poca_xyz(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._poca_dim]

    @property
    def muon_poca_xyz_unc(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._poca_dim]

    @property
    def muon_total_scatter(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._tot_scatter_dim]

    @property
    def muon_total_scatter_unc(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._tot_scatter_dim]

    @property
    def muon_theta_in(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._theta_in_dim]

    @property
    def muon_theta_in_unc(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._theta_in_dim]

    @property
    def muon_theta_out(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._theta_out_dim]

    @property
    def muon_theta_out_unc(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._theta_out_dim]

    @property
    def muon_mom(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._mom_dim]

    @property
    def muon_mom_unc(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._mom_dim]

    @property
    def muon_efficiency(self) -> Tensor:
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_efficiency


class VoxelX0Inferer(AbsX0Inferer):
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


# class DeepVolumeInferer(AbsVolumeInferer):
#     def __init__(
#         self,
#         model: Union[torch.jit._script.RecursiveScriptModule, nn.Module],
#         base_inferer: AbsX0Inferer,
#         volume: Volume,
#         grp_feats: List[str],
#         include_unc: bool = False,
#     ):
#         super().__init__(volume=volume)
#         self.model, self.base_inferer, self.include_unc = model, base_inferer, include_unc
#         self.voxel_centres = self.volume.centres
#         self.tomopt_device = self.volume.device
#         self.model_device = next(self.model.parameters()).device

#         self.in_vars: List[Tensor] = []
#         self.in_var_uncs: List[Tensor] = []
#         self.efficiencies: List[Tensor] = []
#         self.in_var: Optional[Tensor] = None
#         self.in_var_unc: Optional[Tensor] = None
#         self.efficiency: Optional[Tensor] = None

#         self.grp_feats = grp_feats
#         self.in_feats = []
#         if "pred_x0" in self.grp_feats:
#             self.in_feats += ["pred_x0"]
#         if "delta_angles" in self.grp_feats:
#             self.in_feats += ["dtheta", "dphi"]
#         if "total_scatter" in self.grp_feats:
#             self.in_feats += ["total_scatter"]
#         if "track_angles" in self.grp_feats:
#             self.in_feats += ["theta_x_in", "theta_y_in", "theta_x_out", "theta_y_out"]
#         if "track_xy" in self.grp_feats:
#             self.in_feats += ["x_in", "y_in", "x_out", "y_out"]
#         if "poca" in self.grp_feats:
#             self.in_feats += ["poca_x", "poca_y", "poca_z"]
#         if "dpoca" in self.grp_feats:
#             self.in_feats += ["dpoca_x", "dpoca_y", "dpoca_z", "dpoca_r"]
#         if "voxels" in self.grp_feats:
#             self.in_feats += ["vox_x", "vox_y", "vox_z"]

#     def compute_efficiency(self, scatters: AbsScatterBatch) -> Tensor:
#         return self.base_inferer.compute_efficiency(scatters=scatters)

#     def get_base_predictions(self, scatters: AbsScatterBatch) -> Tuple[Tensor, Tensor]:
#         x, u = self.base_inferer.muon_x0_from_scatters(scatters=scatters)
#         return x[:, None], u[:, None]

#     def _build_vars(self, scatters: AbsScatterBatch, pred_x0: Tensor, pred_x0_unc: Tensor) -> None:
#         feats, uncs = [], []
#         if "pred_x0" in self.grp_feats:
#             feats += [pred_x0]
#             if self.include_unc:
#                 uncs += [pred_x0_unc]
#         if "delta_angles" in self.grp_feats:
#             feats += [scatters.dtheta, scatters.dphi]
#             if self.include_unc:
#                 uncs += [scatters.dtheta_unc, scatters.dphi_unc]
#         if "total_scatter" in self.grp_feats:
#             feats += [scatters.total_scatter]
#             if self.include_unc:
#                 uncs += [scatters.total_scatter_unc]
#         if "track_angles" in self.grp_feats:
#             feats += [scatters.theta_xy_in, scatters.theta_xy_out]
#             if self.include_unc:
#                 uncs += [scatters.theta_xy_in_unc, scatters.theta_xy_out_unc]
#         if "track_xy" in self.grp_feats:
#             feats += [scatters.xyz_in[:, :2], scatters.xyz_out[:, :2]]
#             if self.include_unc:
#                 uncs += [scatters.xyz_in_unc[:, :2], scatters.xyz_out_unc[:, :2]]
#         if "poca" in self.grp_feats:
#             feats += [scatters.poca_xyz]
#             if self.include_unc:
#                 uncs += [scatters.poca_xyz_unc]
#         if "dpoca" in self.grp_feats:
#             feats += [scatters.poca_xyz]
#             if self.include_unc:
#                 uncs += [scatters.poca_xyz_unc]

#         self.in_vars.append(torch.cat(feats, dim=-1))
#         if self.include_unc:
#             self.in_var_uncs.append(torch.cat(uncs, dim=-1))
#         self.efficiencies.append(self.compute_efficiency(scatters=scatters)[:, None])

#     def add_scatters(self, scatters: AbsScatterBatch) -> None:
#         self.scatter_batches.append(scatters)
#         pred_x0, pred_x0_unc = self.get_base_predictions(scatters)
#         self._build_vars(scatters, pred_x0, pred_x0_unc)

#     def _build_inputs(self, in_var: Tensor) -> Tensor:
#         data = in_var[None, :].repeat_interleave(len(self.voxel_centres), dim=0)
#         if "dpoca" in self.grp_feats:
#             i = self.in_feats.index("dpoca_x")
#             j = self.in_feats.index("dpoca_r")
#             data[:, :, i:j] -= self.voxel_centres[:, None].repeat_interleave(len(in_var), dim=1)
#             data = torch.cat((data, torch.norm(data[:, :, i:j], dim=-1, keepdim=True)), dim=-1)  # dR
#         # Add voxel centres
#         if "voxels" in self.grp_feats:
#             data = torch.cat((data, self.voxel_centres[:, None].repeat_interleave(len(in_var), dim=1)), dim=-1)
#         return data

#     def _get_weight(self) -> Tensor:
#         """Maybe alter this to include resolution/pred uncertainties"""
#         return self.efficiency.sum()

#     def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
#         self.in_var = torch.cat(self.in_vars, dim=0)
#         if self.include_unc:
#             self.in_var_unc = torch.cat(self.in_var_uncs, dim=0)
#         self.efficiency = torch.cat(self.efficiencies, dim=0)

#         inputs = self._build_inputs(self.in_var)
#         pred = self.model(inputs[None].to(self.model_device)).to(self.tomopt_device)
#         weight = self._get_weight()
#         return pred, weight


# class WeightedDeepVolumeInferer(DeepVolumeInferer):
#     def __init__(
#         self,
#         model: Union[torch.jit._script.RecursiveScriptModule, nn.Module],
#         base_inferer: AbsX0Inferer,
#         volume: Volume,
#         grp_feats: List[str],
#         include_unc: bool = False,
#     ):
#         super().__init__(model=model, base_inferer=base_inferer, volume=volume, grp_feats=grp_feats, include_unc=include_unc)
#         self.in_var_weights: List[Tensor] = []

#     def add_scatters(self, scatters: AbsScatterBatch) -> None:
#         self.scatter_batches.append(scatters)
#         pred_x0, pred_x0_unc = self.get_base_predictions(scatters)
#         self._build_vars(scatters, pred_x0, pred_x0_unc)
#         self.in_var_weights.append((pred_x0_unc / pred_x0) ** 2)

#     def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
#         self.in_var = torch.cat(self.in_vars, dim=0)
#         if self.include_unc:
#             self.in_var_unc = torch.cat(self.in_var_uncs, dim=0)
#         self.efficiency = torch.cat(self.efficiencies, dim=0)
#         self.in_var_weight = torch.cat(self.in_var_weights, dim=0)

#         weight = self.efficiency / self.in_var_weight
#         weighted_vars = torch.cat((weight, self.in_var), dim=1)
#         inputs = self._build_inputs(weighted_vars)
#         pred = self.model(inputs[None].to(self.model_device)).to(self.tomopt_device)
#         weight = self._get_weight()
#         return pred, weight


class DenseBlockClassifierFromX0s(AbsVolumeInferer):
    r"""
    Transforms voxel-wise X0 preds into binary classification statistic under the hypothesis of a small, dense block against a light-weight background
    """

    def __init__(
        self,
        n_block_voxels: int,
        partial_x0_inferer: Type[AbsX0Inferer],
        volume: Volume,
        use_avgpool: bool = True,
        cut_coef: float = 1e4,
        ratio_offset: float = -1.0,
        ratio_coef: float = 1.0,
    ):
        super().__init__(volume=volume)
        self.use_avgpool, self.cut_coef, self.ratio_offset, self.ratio_coef = use_avgpool, cut_coef, ratio_offset, ratio_coef
        self.x0_inferer = partial_x0_inferer(volume=self.volume)
        self.frac = n_block_voxels / self.volume.centres.numel()

    def add_scatters(self, scatters: AbsScatterBatch) -> None:
        self.x0_inferer.add_scatters(scatters)

    def _get_inv_weight(self) -> Tensor:
        """Maybe alter this to include resolution/pred uncertainties"""
        return self.x0_inferer.muon_efficiency

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        vox_preds, _ = self.x0_inferer.get_prediction()
        if self.use_avgpool:
            vox_preds = F.avg_pool3d(vox_preds[None], kernel_size=3, stride=1, padding=1, count_include_pad=False)[0]

        flat_preds = vox_preds.flatten()
        cut = flat_preds.kthvalue(1 + round(self.frac * (flat_preds.numel() - 1))).values

        w_bkg = torch.sigmoid(self.cut_coef * (flat_preds - cut))
        w_blk = 1 - w_bkg

        mean_bkg = (w_bkg * flat_preds).sum() / w_bkg.sum()
        mean_blk = (w_blk * flat_preds).sum() / w_blk.sum()

        r = 2 * (mean_bkg - mean_blk) / (mean_bkg + mean_blk)
        r = (r + self.ratio_offset) * self.ratio_coef
        pred = torch.sigmoid(r)
        weight = self._get_inv_weight()
        return pred[None, None], weight


class AbsIntClassifierFromX0(AbsVolumeInferer):
    """Abstract class for inferring integers through multiclass classification from voxelwise X0 predictions"""

    def __init__(
        self,
        partial_x0_inferer: Type[AbsX0Inferer],
        volume: Volume,
        output_probs: bool = True,
        class2float: Optional[Callable[[Tensor, Volume], Tensor]] = None,
    ):
        super().__init__(volume=volume)
        self.output_probs, self.class2float = output_probs, class2float
        self.x0_inferer = partial_x0_inferer(volume=self.volume)

    def add_scatters(self, scatters: AbsScatterBatch) -> None:
        self.x0_inferer.add_scatters(scatters)

    def _get_inv_weight(self) -> Tensor:
        """Maybe alter this to include resolution/pred uncertainties"""
        return self.x0_inferer.muon_efficiency

    @abstractmethod
    def x02probs(self, vox_preds: Tensor, vox_inv_weights: Tensor) -> Tensor:
        """Convert voxelwise X0 predictions to int probabilities"""
        pass

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        vox_preds, vox_inv_weights = self.x0_inferer.get_prediction()

        probs = self.x02probs(vox_preds, vox_inv_weights)
        weight = self._get_inv_weight()
        if self.output_probs:
            return probs, weight
        else:
            pred = torch.argmax(probs, dim=-1)
            if self.class2float is None:
                return pred, weight
            else:
                return self.class2float(pred, self.volume), weight
