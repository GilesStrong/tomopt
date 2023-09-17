from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

from ...inference.scattering import ScatterBatch
from ...inference.volume import AbsIntClassifierFromX0, AbsVolumeInferrer, AbsX0Inferrer
from ...volume import Volume

__all__ = ["EdgeDetLadleFurnaceFillLevelInferrer", "PocaZLadleFurnaceFillLevelInferrer"]


class EdgeDetLadleFurnaceFillLevelInferrer(AbsIntClassifierFromX0):
    r"""
    Research tested only: no unit tests
    """

    def __init__(
        self,
        partial_x0_inferrer: Type[AbsX0Inferrer],
        volume: Volume,
        pipeline: List[str] = ["remove_ladle", "avg_3d", "avg_layers", "avg_1d", "ridge_1d_0", "negative", "max_div_min"],
        add_batch_dim: bool = True,
        output_probs: bool = True,
    ):
        super().__init__(
            partial_x0_inferrer=partial_x0_inferrer,
            volume=volume,
            output_probs=output_probs,
            class2float=self._class2float,
        )
        self.pipeline, self.add_batch_dim = pipeline, add_batch_dim

    @staticmethod
    def _class2float(preds: Tensor, volume: Volume) -> Tensor:
        return ((preds + 1) * volume.passive_size) + volume.get_passive_z_range()[0]

    @staticmethod
    def avg_3d(x: Tensor) -> Tensor:
        return F.avg_pool3d(x, kernel_size=3, padding=1, stride=1, count_include_pad=False)

    @staticmethod
    def gauss_3d(x: Tensor) -> Tensor:
        gauss = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
        gauss.weight.data = Tensor([[[[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[2, 4, 2], [4, 8, 4], [2, 4, 2]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]])
        gauss.requires_grad_(False)
        return gauss(x[:, None]).squeeze() / gauss.weight.sum()

    @staticmethod
    def avg_layers(x: Tensor) -> Tensor:
        return x.mean((-1, -2))

    @staticmethod
    def max_sub_min(x: Tensor) -> Tensor:
        maxes = F.max_pool1d(x, kernel_size=3, padding=1, stride=1)
        mins = -F.max_pool1d(-x, kernel_size=3, padding=1, stride=1)
        return maxes - mins

    @staticmethod
    def max_div_min(x: Tensor) -> Tensor:
        maxes = F.max_pool1d(x, kernel_size=3, padding=1, stride=1)
        mins = -F.max_pool1d(-x, kernel_size=3, padding=1, stride=1)
        return maxes / mins

    @staticmethod
    def edge_det(x: Tensor, kernel: Tuple[float, float, float]) -> Tensor:
        edge = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
        edge.weight.data = Tensor([[kernel]])
        edge.requires_grad_(False)
        return edge(x[:, None]).squeeze(1)

    def ridge_1d_0(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 0, -1))

    def ridge_1d_2(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 2, -1))

    def ridge_1d_4(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 4, -1))

    def ridge_1d_8(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 8, -1))

    def prewit_1d(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 0, 1))

    def laplacian_1d(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (1, -4, 1))

    @staticmethod
    def gauss_1d(x: Tensor) -> Tensor:
        gauss = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
        gauss.weight.data = Tensor([[[2, 4, 2]]])
        gauss.requires_grad_(False)
        return gauss(x[:, None]).squeeze() / 8

    @staticmethod
    def avg_1d(x: Tensor) -> Tensor:
        return F.avg_pool1d(x, kernel_size=3, padding=1, stride=1, count_include_pad=False)

    @staticmethod
    def negative(x: Tensor) -> Tensor:
        return -x

    @staticmethod
    def remove_ladle(x: Tensor) -> Tensor:
        """Assumes ladle is 1 voxel thick"""
        return x[:, 1:, 1:-1, 1:-1]

    def x02probs(self, vox_preds: Tensor) -> Tensor:
        if self.add_batch_dim:
            vox_preds = vox_preds[None]
        for f in self.pipeline:
            vox_preds = self.__getattribute__(f)(vox_preds)
        if self.add_batch_dim:
            vox_preds = vox_preds[0]
        return F.softmax(vox_preds, dim=-1)


class PocaZLadleFurnaceFillLevelInferrer(AbsVolumeInferrer):
    r"""
    Research tested only: no unit tests

    Computes fill heigh based on weighted average of z of POCAs
    """

    _n_mu: Optional[int] = None
    _muon_scatter_vars: Optional[Tensor] = None  # (mu, vars)
    _muon_scatter_var_uncs: Optional[Tensor] = None  # (mu, vars)
    _muon_probs_per_voxel_zxy: Optional[Tensor] = None  # (mu, zxy)
    _muon_efficiency: Tensor = None  # (mu, eff)
    _pred_height: Optional[Tensor] = None  # (h)
    _pred_height_unc: Optional[Tensor] = None  # (h)
    _var_order_szs = [("poca", 3)]

    def __init__(self, volume: Volume, smooth: Union[float, Tensor] = 0.1):
        r"""
        Initialises the inference class for the provided volume.
        """

        super().__init__(volume=volume)
        self._set_var_dimensions()

        self.xy_centres = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.volume.passive_size / 2,
                    self.volume.lw[0].cpu().item() - (self.volume.passive_size / 2),
                    int(self.volume.lw[0].cpu().item() / self.volume.passive_size),
                    device=self.volume.device,
                ),
                torch.linspace(
                    self.volume.passive_size / 2,
                    self.volume.lw[1].cpu().item() - (self.volume.passive_size / 2),
                    int(self.volume.lw[0].cpu().item() / self.volume.passive_size),
                    device=self.volume.device,
                ),
            ),
            -1,
        ).reshape(-1, 2)
        self.xy_edges = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    0.0,
                    self.volume.lw[0].cpu().item() - self.volume.passive_size,
                    int(self.volume.lw[0].cpu().item() / self.volume.passive_size),
                    device=self.volume.device,
                ),
                torch.linspace(
                    0.0,
                    self.volume.lw[1].cpu().item() - self.volume.passive_size,
                    int(self.volume.lw[0].cpu().item() / self.volume.passive_size),
                    device=self.volume.device,
                ),
            ),
            -1,
        ).reshape(-1, 2)
        self.smooth = smooth  # type: ignore [assignment]

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

    def _reset_vars(self) -> None:
        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self._n_mu = None
        self._muon_scatter_vars = None  # (mu, vars)
        self._muon_scatter_var_uncs = None  # (mu, vars)
        self._muon_probs_per_voxel_zxy = None  # (mu, z,x,y)
        self._muon_efficiency = None  # (mu, eff)
        self._pred_height = None
        self._pred_height_unc = None

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

    def get_prediction(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Computes the predicted fill level via a weighted average of POCA locations.

        Returns:
            pred: fill-height prediction [m]
            inv_weight: sum of muon efficiencies
        """

        if len(self.scatter_batches) == 0:
            print("Warning: unable to scan volume with prescribed number of muons.")
            return None, None
        return self.pred_height, self.inv_weights

    @property
    def pred_height(self) -> Tensor:
        r"""
        Returns:
            (h) tensor of fill-height prediction
        """

        if self._pred_height is None:
            self._pred_height = self._get_height_pred()
            self._pred_height_unc = None
        return self._pred_height

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

    @property
    def inv_weights(self) -> Tensor:
        r"""
        Returns:
            Sum of muon efficiencies
        """

        return self.muon_efficiency.sum()

    @property
    def muon_efficiency(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the efficiencies of the muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_efficiency

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
    def smooth(self) -> Tensor:
        return self._smooth

    @smooth.setter
    def smooth(self, smooth: Union[float, Tensor]) -> None:
        if not smooth > 0:
            raise ValueError("smooth argument must be positive and non-zero")
        if not isinstance(smooth, Tensor):
            smooth = Tensor([smooth], device=self.device)
        self._smooth = smooth
        self.sigmoid_grid_wgt = ((self._sig_model(self.xy_centres) - 0.5) * 2).prod(-1, keepdim=True)  # 0 at edges, 1 at centre

    def _sig_model(self, xy: Tensor) -> Tensor:
        half_width = self.volume.lw / 2
        delta = (xy - half_width) / half_width
        coef = torch.sigmoid((1 - (torch.sign(delta) * delta)) / self.smooth)
        return coef / torch.sigmoid(1 / self.smooth)

    def _get_height_pred(self) -> Tensor:
        r"""
        Computes the predicted fill-height given the POCAs in the scatter batches added.

        Returns:
            (h) tensor of fill-height prediction [m]
        """

        z_pos = self.muon_poca_xyz[:, 2:]
        z_unc = self.muon_poca_xyz_unc[:, 2:]
        eff = self.muon_efficiency.reshape(self.n_mu, 1)

        # Downweight poca near sides to reduce bias
        xy_gauss = Normal(self.muon_poca_xyz[:, None, :2], self.muon_poca_xyz_unc[:, None, :2])
        probs = (xy_gauss.cdf(self.xy_edges + self.volume.passive_size) - xy_gauss.cdf(self.xy_edges)).prod(
            -1, keepdim=True
        )  # pixelwise probs in xy  (mu, pixel, prob)
        wgt_probs = probs * self.sigmoid_grid_wgt[None]
        self.sig_wgt = wgt_probs.sum(-2)  # (mu, wgt)

        # Clamp uncertainties in case they're very small/large
        unc_low = z_unc.view(-1).kthvalue(1 + round(0.15865 * (z_unc.numel() - 1))).values.detach()
        unc_high = z_unc.view(-1).kthvalue(1 + round(0.84135 * (z_unc.numel() - 1))).values.detach()
        z_unc = torch.clip(z_unc, unc_low, unc_high)

        wgt = self.sig_wgt * eff / (z_unc**2)
        # Clamp weight in case some muons dominate
        wgt_high = wgt.view(-1).kthvalue(1 + round(0.84135 * (wgt.numel() - 1))).values.detach()
        wgt = torch.clip(wgt, 0.0, wgt_high)
        self.wgt = wgt

        mean_z = (self.wgt * z_pos).sum() / self.wgt.sum()

        return mean_z[None]
