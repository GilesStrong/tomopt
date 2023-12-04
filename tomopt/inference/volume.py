from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Dict, List, Type, Callable
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Normal

from .scattering import ScatterBatch
from ..volume import Volume
from ..core import SCATTER_COEF_A
from ..utils import jacobian


r"""
Provides implementations of classes designed to infer targets of passive volumes
using the variables computed by e.g. :class:`~tomopt.inference.scattering.ScatterBatch`.
"""

__all__ = [
    "AbsVolumeInferrer",
    "AbsX0Inferrer",
    "AbsIntClassifierFromX0",
    "PanelX0Inferrer",
    "DenseBlockClassifierFromX0s",
]  # "DeepVolumeInferrer", "WeightedDeepVolumeInferrer"]


class AbsVolumeInferrer(metaclass=ABCMeta):
    r"""
    Abstract base class for volume inference.

    Inheriting classes are expected to be fed multiple :class:`~tomopt.inference.scattering.ScatterBatch` s,
    via :meth:`~tomopt.inference.volume.AbsVolumeInferrer.add_scatters`, for a single :class:`~tomopt.volume.volume.Volume`
    and return a volume prediction based on all of the muon batches when :meth:`~tomopt.inference.volume.AbsVolumeInferrer.get_prediction` is called.

    Arguments:
        volume: volume through which the muons will be passed
    """

    def __init__(self, volume: Volume):
        r"""
        Initialises the inference class for the provided volume.
        """

        self.scatter_batches: List[ScatterBatch] = []
        self.volume = volume
        self.size, self.lw, self.device = self.volume.passive_size, self.volume.lw, self.volume.device

    @abstractmethod
    def _reset_vars(self) -> None:
        r"""
        Inheriting classes must override this method to reset any variable/predictions made from the added scatter batches.
        """

        pass

    @abstractmethod
    def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
        r"""
        Inheriting classes must override this method to provide a computation of the per-muon efficiency, given the individual muon hit efficiencies.
        """

        pass

    @abstractmethod
    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Inheriting classes must override this method to provide a prediction computed using the added scatter batches.
        Predictions can be accompanied by an optional "inverse weight" designed to divide the loss of the predictions: loss(pred,targs)/inv_weight
        E.g. the sum of muon efficiencies.
        """

        pass

    def add_scatters(self, scatters: ScatterBatch) -> None:
        r"""
        Appends a new set of muon scatter variables.
        When :meth:`~tomopt.inference.volume.AbsVolumeInferrer.get_prediction` is called, the prediction will be based on all
        :class:`~tomopt.inference.scattering.ScatterBatch` s added up to that point
        """

        self._reset_vars()  # Ensure that any previously computed predictions are wiped
        self.scatter_batches.append(scatters)


class AbsX0Inferrer(AbsVolumeInferrer):
    r"""
    Abstract base class for inferring the X0 of every voxel in the passive volume.

    The inference is based on the PoCA approach of assigning the entirety of the muon scattering to a single point,
    and the X0 computation is based on inversion of the PDG scattering model described in
    https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf.

    Once all scatter batches have been added, the inference proceeds thusly:
        - For each muon i, a probability p_ij, is computed according to the probability that the PoCA was located in voxel j.
        - These probabilities are computed by integrating over the voxel the PDF of 3 uncorrelated Gaussians centred on the PoCA, with scales equal the uncertainty on the PoCA position in x,y,z.
        - p_ij is multiplied by muon efficiency e_i to compute a muon/voxel weight w_ij.
        - Inversion of the PDG model gives: :math:`X_0 = \left(\frac{0.0136}{p^{\mathrm{rms}}}\right)^2\frac{\delta z}{\cos\left(\bar{\theta}^{\mathrm{rms}}\right)}\frac{2}{\theta^{\mathrm{rms}}_{\mathrm{tot.}}}`
        - In order to account for the muon weights and compute different X0s for the voxels whilst using the whole muon population:
            - Weighted RMSs are computed for each of the scattering terms in the right-hand side of the equation.
            - In addition to the muon weight w_ij, the variances of the squared values of the scattering variables is used to divide w_ij.
        - The result is a set of X0 predictions X0_j.

    .. important::
        Inversion of the PDG model does NOT account for the natural log term.

    .. important::
        To simplify the computation code, this class relies heavily on lazy computation and memoisation; be careful if calling private methods manually.

    Arguments:
        volume: volume through which the muons will be passed
    """

    _n_mu: Optional[int] = None
    _muon_scatter_vars: Optional[Tensor] = None  # (mu, vars)
    _muon_scatter_var_uncs: Optional[Tensor] = None  # (mu, vars)
    _muon_probs_per_voxel_zxy: Optional[Tensor] = None  # (mu, z,x,y)
    _muon_efficiency: Tensor = None  # (mu, eff)
    _vox_zxy_x0_preds: Optional[Tensor] = None  # (z,x,y)
    _vox_zxy_x0_pred_uncs: Optional[Tensor] = None  # (z,x,y)
    _var_order_szs = [("poca", 3), ("tot_scatter", 1), ("theta_in", 1), ("theta_out", 1), ("mom", 1)]

    def __init__(self, volume: Volume):
        r"""
        Initialises the inference class for the provided volume.
        """

        super().__init__(volume=volume)
        self._set_var_dimensions()
        # set shapes
        self.shp_xyz = [
            round(self.lw.cpu().numpy()[0] / self.size),
            round(self.lw.cpu().numpy()[1] / self.size),
            len(self.volume.get_passives()),
        ]
        self.shp_zxy = [self.shp_xyz[2], self.shp_xyz[0], self.shp_xyz[1]]

    @staticmethod
    def x0_from_scatters(deltaz: float, total_scatter: Tensor, theta_in: Tensor, theta_out: Tensor, mom: Tensor) -> Tensor:
        r"""
        Computes the X0 of a voxel, by inverting the PDG scattering model in terms of the scattering variables

        .. important::
            Inversion of the PDG model does NOT account for the natural log term.

        Arguments:
            deltaz: height of the voxels
            total_scatter: (voxels,1) tensor of the (RMS of the) total angular scattering of the muon(s)
            theta_in: (voxels,1) tensor of the (RMS of the) theta of the muon(s), as inferred using the incoming trajectory/ies
            theta_out: (voxels,1) tensor of the (RMS of the) theta of the muon(s), as inferred using the outgoing trajectory/ies
            mom: (voxels,1) tensor of the (RMS of the) momentum/a of the muon(s)

        Returns:
            (voxels,1) estimated X0 in metres
        """

        cos_theta = (theta_in.cos() + theta_out.cos()) / 2
        return ((SCATTER_COEF_A / mom) ** 2) * deltaz / (total_scatter.pow(2) * cos_theta)

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Computes the predicted X0 per voxel as a (z,x,y) tensor via PDG scatter-model inversion for the provided scatter batches.

        Returns:
            pred: (z,x,y) voxelwise X0 predictions
            inv_weight: sum of muon efficiencies
        """

        if len(self.scatter_batches) == 0:
            print("Warning: unable to scan volume with prescribed number of muons.")
            return None, None
        return self.vox_zxy_x0_preds, self.inv_weights

    @staticmethod
    def _weighted_rms(x: Tensor, wgt: Tensor) -> Tensor:
        r"""
        Computes the weighted root mean squared value of the provided list of variable values

        Arguments:
            x: (N,*) tensor of variable values
            wgt: (N,*) weight to assign per row in the x tensor

        Returns:
            Weighted RMS of the variable
        """

        return ((x.square() * wgt).sum(0) / wgt.sum(0)).sqrt()

    @staticmethod
    def _weighted_mean(x: Tensor, wgt: Tensor) -> Tensor:
        r"""
        Computes the weighted mean value of the provided list of variable values

        Arguments:
            x: (N,*) tensor of variable values
            wgt: (N,*) weight to assign per row in the x tensor

        Returns:
            Weighted mean of the variable
        """

        return (x * wgt).sum(0) / wgt.sum(0)

    def _reset_vars(self) -> None:
        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self._n_mu = None
        self._muon_scatter_vars = None  # (mu, vars)
        self._muon_scatter_var_uncs = None  # (mu, vars)
        self._muon_probs_per_voxel_zxy = None  # (mu, z,x,y)
        self._muon_efficiency = None  # (mu, eff)
        self._vox_zxy_x0_preds = None  # (z,x,y)
        self._vox_zxy_x0_pred_uncs = None  # (z,x,y)

    def _set_var_dimensions(self) -> None:
        r"""
        Configures the indexing of the dependent variable and uncertainty tensors
        """

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
        r"""
        Combines scatter data from all the batches added so far.
        Any muons with NaN or Inf entries will be filtered out of the resulting tensors.

        To aid in uncertainty computation, a pair of tensors are created with the all scatter variables and their uncertainties.
        These are then indexed to retrieve the scatter variables.
        """

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

    def _get_voxel_zxy_x0_pred_uncs(self) -> Tensor:
        r"""
        Computes the uncertainty on the predicted voxelwise X0s, via gradient-based error propagation.

        .. warning::
            This computation assumes un-correlated uncertainties, which is probably incorrect.
            TODO: correct this to consider correlated uncertainties

        .. warning::
            This method is incredibly slow and not recommended for use

        Returns:
            (z,x,y) tensor of uncertainties on voxelwise X0s
        """

        jac = torch.nan_to_num(jacobian(self.vox_zxy_x0_preds, self._muon_scatter_vars))  # Compute dx0/dvar  (z,x,y,mu,var)
        unc = self._muon_scatter_var_uncs
        unc = torch.where(torch.isinf(unc), torch.tensor([0]).type(unc.type()), unc)[None, None, None]  # (1,1,1,mu,var)
        jac, unc = jac.reshape(jac.shape[0], jac.shape[1], jac.shape[2], -1), unc.reshape(unc.shape[0], unc.shape[1], unc.shape[2], -1)  # (z,x,y,mu*var)

        # Compute unc^2 = sum[(unc_x*dx0/dx)^2] summing over all n hits
        unc_2 = (jac * unc).square()  # (z,x,y,N)

        pred_unc = unc_2.sum(-1).sqrt()  # (z,x,y)
        return pred_unc

    def _get_voxel_zxy_x0_preds(self) -> Tensor:
        r"""
        Computes the X0 predictions per voxel using the scatter batched added.

        TODO: Implement differing x0 according to poca_xyz via Gaussian spread

        Returns:
            (z,x,y) tensor of voxelwise X0 predictions
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
            deltaz=self.size, total_scatter=vox_tot_total_scatter / math.sqrt(2), theta_in=vox_theta_in, theta_out=vox_theta_out, mom=vox_mom
        )  # (z,x,y)

        if vox_x0_preds.isnan().any():
            raise ValueError("Prediction contains NaN values")

        return vox_x0_preds

    @property
    def vox_zxy_x0_preds(self) -> Tensor:
        r"""
        Returns:
            (z,x,y) tensor of voxelwise X0 predictions
        """

        if self._vox_zxy_x0_preds is None:
            self._vox_zxy_x0_preds = self._get_voxel_zxy_x0_preds()
            self._vox_zxy_x0_pred_uncs = None
        return self._vox_zxy_x0_preds

    @property
    def vox_zxy_x0_pred_uncs(self) -> Tensor:
        r"""
        .. warning::
            Not recommended for use: long calculation; not unit-tested

        Returns:
            (z,x,y) tensor of uncertainties on voxelwise X0s
        """

        if self._vox_zxy_x0_pred_uncs is None:
            self._vox_zxy_x0_pred_uncs = self._get_voxel_zxy_x0_pred_uncs()
        return self._vox_zxy_x0_pred_uncs

    @property
    def inv_weights(self) -> Tensor:
        r"""
        Returns:
            Sum of muon efficiencies
        """

        return self.muon_efficiency.sum()

    @property
    def muon_probs_per_voxel_zxy(self) -> Tensor:  # (mu,z,x,y)
        r"""
        .. warning::
            Integration tested only

        TODO: Don't assume that poca_xyz uncertainties are uncorrelated

        Returns:
            (muons,z,x,y) tensor of probabilities that the muons' PoCAs were located in the given voxels.
        """
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
        r"""
        Returns:
            Total number muons included in the inference
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._n_mu

    @property
    def muon_poca_xyz(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) tensor of PoCA locations
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._poca_dim]

    @property
    def muon_poca_xyz_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,xyz) tensor of PoCA location uncertainties
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._poca_dim]

    @property
    def muon_total_scatter(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of total angular scatterings
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._tot_scatter_dim]

    @property
    def muon_total_scatter_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of uncertainties on the total angular scatterings
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._tot_scatter_dim]

    @property
    def muon_theta_in(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the thetas of the incoming muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._theta_in_dim]

    @property
    def muon_theta_in_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the uncertainty on the theta of the incoming muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._theta_in_dim]

    @property
    def muon_theta_out(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the thetas of the outgoing muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._theta_out_dim]

    @property
    def muon_theta_out_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the uncertainty on the theta of the outgoing muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._theta_out_dim]

    @property
    def muon_mom(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the momenta of the muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._mom_dim]

    @property
    def muon_mom_unc(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the uncertainty on the momenta of the muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_var_uncs[:, self._mom_dim]

    @property
    def muon_efficiency(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the efficiencies of the muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_efficiency


class PanelX0Inferrer(AbsX0Inferrer):
    r"""
    Class for inferring the X0 of every voxel in the passive volume using hits recorded by :class:`~tomopt.volume.layer.PanelDetectorLayer` s.

    The inference is based on the PoCA approach of assigning the entirety of the muon scattering to a single point,
    and the X0 computation is based on inversion of the PDG scattering model described in
    https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf.

    Once all scatter batches have been added, the inference proceeds thusly:
        - For each muon i, a probability p_ij, is computed according to the probability that the PoCA was located in voxel j.
        - These probabilities are computed by integrating over the voxel the PDF of 3 uncorrelated Gaussians centred on the PoCA, with scales equal the uncertainty on the PoCA position in x,y,z.
        - p_ij is multiplied by muon efficiency e_i to compute a muon/voxel weight w_ij.
        - Inversion of the PDG model gives: :math:`X_0 = \left(\frac{0.0136}{p^{\mathrm{rms}}}\right)^2\frac{\delta z}{\cos\left(\bar{\theta}^{\mathrm{rms}}\right)}\frac{2}{\theta^{\mathrm{rms}}_{\mathrm{tot.}}}`
        - In order to account for the muon weights and compute different X0s for the voxels whilst using the whole muon population:
            - Weighted RMSs are computed for each of the scattering terms in the right-hand side of the equation.
            - In addition to the muon weight w_ij, the variances of the squared values of the scattering variables is used to divide w_ij.
        - The result is a set of X0 predictions X0_j.

    .. important::
        Inversion of the PDG model does NOT account for the natural log term.

    .. important::
        To simplify the computation code, this class relies heavily on lazy computation and memoisation; be careful if calling private methods manually.

    Arguments:
        volume: volume through which the muons will be passed
    """

    def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
        r"""
        Computes the per-muon efficiency, given the individual muon hit efficiencies,
        as the probability of at least two hits above and below the passive volume.

        Arguments:
            scatters: scatter batch containing muons whose efficiency should be computed

        Returns:
            (muons) tensor of muon efficiencies
        """

        eff = None
        for effs in [scatters.above_hit_effs, scatters.below_hit_effs]:
            leff = None
            effs = effs.squeeze(-1).transpose(0, -1)
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


# class DeepVolumeInferrer(AbsVolumeInferrer):
#     def __init__(
#         self,
#         model: Union[torch.jit._script.RecursiveScriptModule, nn.Module],
#         base_inferrer: AbsX0Inferrer,
#         volume: Volume,
#         grp_feats: List[str],
#         include_unc: bool = False,
#     ):
#         super().__init__(volume=volume)
#         self.model, self.base_inferrer, self.include_unc = model, base_inferrer, include_unc
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

#     def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
#         return self.base_inferrer.compute_efficiency(scatters=scatters)

#     def get_base_predictions(self, scatters: ScatterBatch) -> Tuple[Tensor, Tensor]:
#         x, u = self.base_inferrer.muon_x0_from_scatters(scatters=scatters)
#         return x[:, None], u[:, None]

#     def _build_vars(self, scatters: ScatterBatch, pred_x0: Tensor, pred_x0_unc: Tensor) -> None:
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

#     def add_scatters(self, scatters: ScatterBatch) -> None:
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


# class WeightedDeepVolumeInferrer(DeepVolumeInferrer):
#     def __init__(
#         self,
#         model: Union[torch.jit._script.RecursiveScriptModule, nn.Module],
#         base_inferrer: AbsX0Inferrer,
#         volume: Volume,
#         grp_feats: List[str],
#         include_unc: bool = False,
#     ):
#         super().__init__(model=model, base_inferrer=base_inferrer, volume=volume, grp_feats=grp_feats, include_unc=include_unc)
#         self.in_var_weights: List[Tensor] = []

#     def add_scatters(self, scatters: ScatterBatch) -> None:
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


class DenseBlockClassifierFromX0s(AbsVolumeInferrer):
    r"""
    Class for inferreing the presence of a small amount of denser material in the passive volume.

    Transforms voxel-wise X0 preds into binary classification statistic under the hypothesis of a small, dense block against a light-weight background.
    This test statistic, s is computed as:

    .. math::

        r = 2 \frac{\bar{X0}_{0,\mathrm{bkg}} - \bar{X0}_{0,\mathrm{blk}}}{\bar{X0}_{0,\mathrm{bkg}} + \bar{X0}_{0,\mathrm{blk}}}
        s = \sigma\!(a(r+b))

    where :math:`\bar{X0}_{0,\mathrm{blk}}` is the mean X0 of the N lowest X0 voxels,
    and :math:`\bar{X0}_{0,\mathrm{bkg}}` is the mean X0 of the remaining voxels.
    a and b are rescaling coefficients and offsets.

    This results in a differentiable value constrained beween 0 and 1, with values near 0 indicating that no relatively dense material is present,
    and values nearer 1 indicating that it is present.
    In case it is expected that the dense material forms a contiguous block, the voxelwise X0s can be blurred via a stride-1 kernel-size-3 average pooling.

    In actuality, the "cut" on X0s into background and block is implemented as a sigmoid weight, centred at the necessary kth value of the X0.
    This means that the test statisitc is also differentiable w.r.t. the cut.

    Arguments:
        n_block_voxels: number of voxels expected to be occupied by the dense material, if present
        partial_x0_inferrer: (partial) class to instatiate to provide the voxelwise X0 predictions
        volume: volume through which the muons will be passed
        use_avgpool: wether to blur voxelwise X0 predicitons with a stride-1 kernel-size-3 average pooling
            useful when the dense material is expected to form a contiguous block
        cut_coef: the "sharpness" of the sigmoid weight that splits voxels into block and background.
            Higher values results in a sharper cut.
        ratio_offset: additive constant for the X0 ratio
        ratio_coef: multiplicative coefficient for the offset X0 ratio
    """

    def __init__(
        self,
        n_block_voxels: int,
        partial_x0_inferrer: Type[AbsX0Inferrer],
        volume: Volume,
        use_avgpool: bool = True,
        cut_coef: float = 1e4,
        ratio_offset: float = -1.0,
        ratio_coef: float = 1.0,
    ):
        r"""
        Initialises the inference class for the provided volume.
        """

        super().__init__(volume=volume)
        self.use_avgpool, self.cut_coef, self.ratio_offset, self.ratio_coef = use_avgpool, cut_coef, ratio_offset, ratio_coef
        self.x0_inferrer = partial_x0_inferrer(volume=self.volume)
        self.frac = n_block_voxels / self.volume.centres.numel()

    def add_scatters(self, scatters: ScatterBatch) -> None:
        r"""
        Appends a new set of muon scatter vairables.
        When :meth:`~tomopt.inference.volume.DenseBlockClassifierFromX0s.get_prediction` is called, the prediction will be based on all
        :class:`~tomopt.inference.scattering.ScatterBatch` s added up to that point
        """

        self.x0_inferrer.add_scatters(scatters)

    def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
        r"""
        Compuates the per-muon efficiency according to the method implemented by the X0 inferrer.

        Arguments:
            scatters: scatter batch containing muons whose efficiency should be computed

        Returns:
            (muons) tensor of muon efficiencies
        """

        return self.x0_inferrer.compute_efficiency(scatters=scatters)

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Computes the test statistic for the volume, with values near 0 indicating that no relatively dense material is present,
        and values nearer 1 indicating that it is present.

        Returns:
            pred: (1,1,1) volume prediction
            inv_weight: sum of muon efficiencies
        """

        vox_preds, inv_weights = self.x0_inferrer.get_prediction()
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
        return pred[None, None], inv_weights

    def _reset_vars(self) -> None:
        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self.x0_inferrer._reset_vars()


class AbsIntClassifierFromX0(AbsVolumeInferrer):
    r"""
    Abstract base class for inferring integer targets through multiclass classification from voxelwise X0 predictions.
    Inheriting classes must provide a way to convert voxelwise X0s into class probabilities of the required dimension.
    Requires a basic inferrer for providing the voxelwise X0 predictions.
    Optionally, the predictions can be returns as the raw class predictions, or the most probable class.
    In case of the latter, this class can be optionally be converted to a float value via a user-provided processing function.

    Arguments:
        partial_x0_inferrer: (partial) class to instatiate to provide the voxelwise X0 predictions
        volume: volume through which the muons will be passed
        output_probs: if True, will return the per-class probabilites, otherwise will return the argmax of the probabilities, over the last dimension
        class2float: optional function to convert class indices to a floating value
    """

    def __init__(
        self,
        partial_x0_inferrer: Type[AbsX0Inferrer],
        volume: Volume,
        output_probs: bool = True,
        class2float: Optional[Callable[[Tensor, Volume], Tensor]] = None,
    ):
        r"""
        Initialises the inference class for the provided volume.
        """

        super().__init__(volume=volume)
        self.output_probs, self.class2float = output_probs, class2float
        self.x0_inferrer = partial_x0_inferrer(volume=self.volume)

    @abstractmethod
    def x02probs(self, vox_preds: Tensor) -> Tensor:
        r"""
        Inheriting classes must override this method to convert voxelwise X0 predictions to class probabilities

        Arguments:
            vox_preds: (z,x,y) tensor of voxelwise X0 predictions

        Returns:
            (*) tensor of class probabilities
        """

        pass

    def add_scatters(self, scatters: ScatterBatch) -> None:
        r"""
        Appends a new set of muon scatter vairables.
        When :meth:`~tomopt.inference.volume.DenseBlockClassifierFromX0s.get_prediction` is called, the prediction will be based on all
        :class:`~tomopt.inference.scattering.ScatterBatch` s added up to that point
        """

        self.x0_inferrer.add_scatters(scatters)

    def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
        r"""
        Compuates the per-muon efficiency according to the method implemented by the X0 inferrer.

        Arguments:
            scatters: scatter batch containing muons whose efficiency should be computed

        Returns:
            (muons) tensor of muon efficiencies
        """

        return self.x0_inferrer.compute_efficiency(scatters=scatters)

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Computes the predicions for the volume.
        If class probabilities were requested during initialisation, then these will be returned.
        Otherwise the most probable class will be returned, and this will be converted to a float value if `class2float` is not None.

        Returns:
            pred: (*) volume prediction
            inv_weight: sum of muon efficiencies
        """

        vox_preds, inv_weights = self.x0_inferrer.get_prediction()

        probs = self.x02probs(vox_preds)
        if self.output_probs:
            return probs, inv_weights
        else:
            pred = torch.argmax(probs, dim=-1)
            if self.class2float is None:
                return pred, inv_weights
            else:
                return self.class2float(pred, self.volume), inv_weights

    def _reset_vars(self) -> None:
        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self.x0_inferrer._reset_vars()
