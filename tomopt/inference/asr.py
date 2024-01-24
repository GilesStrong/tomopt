from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastprogress import progress_bar
from torch import Tensor

from ..volume import Volume
from .scattering import ScatterBatch
from .volume import AbsVolumeInferrer

__all__ = ["AngleStatisticReconstruction", "VolumeInterest"]


class VolumeInterest:
    r"""
    Class for voxelized volume definition.

    """

    def __init__(self, position: List[float], dimension: List[float], voxel_width: float = 0.01) -> None:
        """
        Initialises the volume of interest for the given position, dimension and voxl width

        Arguments:

            position:List[float] the position of the center of the voxelized volume along x,y,z in meters
            dimension = List[float] the span of the voxelized volume along x,y,z in meters
            voxel_width:float the voxel width in meters
            float_precision:int loactions and spans are rounded up to avoid floating precision errors
        """

        # VOI position
        self.xyz = Tensor(position)

        # VOI dimensions
        self.dxyz = Tensor(dimension)

        self.xyz_min = torch.round(self.xyz - self.dxyz / 2, decimals=6)
        self.xyz_max = torch.round(self.xyz + self.dxyz / 2, decimals=6)

        # Voxel width
        self.vox_width = voxel_width

        # Volume voxelization
        self.n_vox_xyz = self._compute_n_voxel(dxyz=self.dxyz, vox_width=self.vox_width)
        self.voxel_centers, self.voxel_edges = self._generate_voxels(n_vox_xyz=self.n_vox_xyz, vox_width=self.vox_width, xyz_min=self.xyz_min)

    @staticmethod
    def _compute_n_voxel(dxyz: Tensor, vox_width: float) -> Tensor:
        r"""
        Compute the number of voxels along x,y,z dimension

        Returns:
            nxyz: (3, ) tensor of number of voxels
        """

        nxyz = dxyz / vox_width
        # make sure voxel size suits the volume size
        # a.k.a make sur nb of voxel is integer along each dimension
        assert ((torch.abs(nxyz.int()) - nxyz).sum()) < 10 ** (-6), "Voxel size does not match VOI dimensions"
        return nxyz.int()

    @staticmethod
    def _generate_voxels(n_vox_xyz: Tensor, vox_width: float, xyz_min: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Compute the xyz locations of center of voxels.
        Compute the locations of low-left-front and upper-right-back edges of voxels.

        Returns:
            voxels_centers:Tensor with size (Nx,Ny,Nz,3)
            voxels_edges:tensor with size (Nx,Ny,Nz,2,3)
        """
        nx, ny, nz = n_vox_xyz[0].item(), n_vox_xyz[1].item(), n_vox_xyz[2].item()
        voxel_centers = vox_width * np.mgrid[0 : int(nx) : 1, 0 : int(ny) : 1, 0 : int(nz) : 1]

        voxel_centers = (voxel_centers + xyz_min[:, None, None, None].numpy() + vox_width / 2).transpose((1, 2, 3, 0)).round(6)

        voxel_edges = np.zeros((n_vox_xyz[0], n_vox_xyz[1], n_vox_xyz[2], 2, 3))

        voxel_edges[:, :, :, 0, :] = voxel_centers - vox_width / 2
        voxel_edges[:, :, :, 1, :] = voxel_centers + vox_width / 2

        return Tensor(voxel_centers), Tensor(voxel_edges)


class AngleStatisticReconstruction(AbsVolumeInferrer):
    r"""
    Class for scattering density inference based on the ASR algorithm.

    Once all scatter batches have been added, the inference proceeds thusly:
        - For each muon, find the "triggered voxels" (voxels traversed by both the incoming and outgoing track).
        - For each muon, append a score s to the list Lj of each individual triggered voxel j.
        - For each voxel j, compute the final scattering density score S as S = f(Lj), we f is the desired score method.
        - The result is a set of voxel-wise scattering density predictions.

    """

    _n_mu: Optional[int] = None
    _muon_scatter_vars: Optional[Tensor] = None  # (mu, vars)
    _muon_scatter_var_uncs: Optional[Tensor] = None  # (mu, vars)
    _muon_efficiency: Tensor = None  # (mu, eff)
    _vox_zxy_density_preds: Optional[Tensor] = None  # (z,x,y)
    _vox_zxy_density_preds_uncs: Optional[Tensor] = None  # (z,x,y)
    _gen_hits: Optional[Tensor] = None  # (mu, n_panels, 3)
    _voi: Optional[VolumeInterest] = None
    _asr_params: Dict[str, Any] = {"dtheta_range": None, "score_method": None, "use_p": None}
    _var_order_szs = [("tot_scatter", 1), ("theta_xy_in", 2), ("theta_xy_out", 2), ("mom", 1), ("track_in", 3), ("track_out", 3)]

    def __init__(
        self,
        volume: Volume,
    ) -> None:
        r"""
        Initialises the inference class for the provided volume and voxelized volume.
        """
        super().__init__(volume=volume)
        self._set_var_dimensions()

        # Number of detection planes
        self.n_panels = np.sum([np.sum([1 for panel in layer.panels]) for layer in self.volume.get_detectors()])  # type: ignore [union-attr]

    @staticmethod
    def _compute_theta_xy_in_out(track_in: Tensor, track_out: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Compute the incoming and outgoing muon zenith angle projections in the XZ and YZ plane

        Arguments:
            track_in: Tensor with size (self._n_mu, 3), the muon incoming track
            track_out: Tensor with size (n_mu, 3) the muon outgoing track

        Returns:
            theta_x_in: Tensor with size (n_mu)
            theta_x_out: Tensor with size (n_mu)
            theta_y_in: Tensor with size (n_mu)
            theta_y_out: Tensor with size (n_mu)

        """

        theta_x_in, theta_x_out = torch.atan(track_in[:, 0] / track_in[:, 2]), torch.atan(track_out[:, 0] / track_out[:, 2])
        theta_y_in, theta_y_out = torch.atan(track_in[:, 1] / track_in[:, 2]), torch.atan(track_out[:, 1] / track_out[:, 2])

        return theta_x_in, theta_x_out, theta_y_in, theta_y_out

    @staticmethod
    def _compute_xyz_in_out(
        n_panels: int, voi: VolumeInterest, theta_xy_in: Tuple[Tensor, Tensor], theta_xy_out: Tuple[Tensor, Tensor], hits: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Compute the location of muons when entering/exiting the volume for incoming and outgoing tracks

        .. warning::
            Assumes the same number of panels above and below the voi!

        Arguments:
            n_panels: int the number of detection panels.
            voi: VolumeInterest an instance of the VolumeInterest class
            theta_xy_in: Tuple[Tensor]  the incoming muon zenith angle projections in the XZ and YZ plane
            theta_xy_out: Tuple[Tensor]  the outcoming muon zenith angle projections in the XZ and YZ plane
            hits: Tensor with size (event, n_panels, 3) the muon hits

        Returns:
            The location of muons when entering/exiting the volume for incoming and outgoing tracks
        """

        # indices of planes directly above and below the voi
        # assumes same number of planes above and below!
        i_plane_above = int(n_panels / 2) - 1
        i_plane_below = int(n_panels / 2)

        xyz_in_voi, xyz_out_voi = torch.zeros((theta_xy_in[0].size()[0], 2, 3)), torch.zeros((theta_xy_in[0].size()[0], 2, 3))

        for i_plane, theta_xy, pm, xyz in zip([i_plane_above, i_plane_below], [theta_xy_in, theta_xy_out], [1, -1], [xyz_in_voi, xyz_out_voi]):

            dz = (abs(hits[:, i_plane, 2] - voi.xyz_max[2]), abs(hits[:, i_plane, 2] - voi.xyz_min[2]))

            for coord, theta in zip([0, 1], theta_xy):

                xyz[:, 0, coord] = hits[:, i_plane, coord] - dz[1] * torch.tan(theta) * (pm)
                xyz[:, 1, coord] = hits[:, i_plane, coord] - dz[0] * torch.tan(theta) * (pm)

            xyz[:, 0, 2], xyz[:, 1, 2] = voi.xyz_min[2], voi.xyz_max[2]

        return xyz_in_voi, xyz_out_voi

    @staticmethod
    def _compute_discrete_track(
        n_mu: int,
        voi: VolumeInterest,
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
        xyz_in_out_voi: Tuple[Tensor, Tensor],
        n_points_per_z_layer: int = 3,
    ) -> Tuple[Tensor, Tensor]:

        r"""
        Computes a discretized version of the muon incoming and outgoing track within the voi.
        The number of points is defined as n_points = n_points_per_z_layer * n_z_layer,
        where n_z_layer is the number of layer of voxels along z.

        Arguments:
            n_mu: int the number muons included in the inference.
            voi: VolumeInterest an instance of the VolumeInterest class.
            theta_xy_in: Tuple[Tensor]  the incoming muon zenith a.ngle projections in the XZ and YZ plane.
            theta_xy_out: Tuple[Tensor]  the outcoming muon zenith angle projections in the XZ and YZ plane.
            xyz_in_out_voi: The location of muons when entering/exiting the volume for incoming and outgoing tracks.
            n_points_per_z_layer:int the number of locations per voxel. Must be not too small (all the voxels are not triggered),
            nor too large (computationaly expensive). Default value is set at 3  point per voxel.

        Returns:
            The discretized incoming and outgoing tracks with size (3, n_points, n_mu)
        """

        n_points = int((voi.n_vox_xyz[2] + 1) * n_points_per_z_layer)

        # Compute the z locations cross the voi
        z_discrete = (
            torch.linspace(torch.min(voi.voxel_edges[0, 0, :, :, 2]).item(), torch.max(voi.voxel_edges[0, 0, :, :, 2]).item(), n_points)[:, None]
        ).expand(-1, n_mu)

        xyz_discrete_in, xyz_discrete_out = torch.ones((3, n_points, n_mu)), torch.ones((3, n_points, n_mu))

        for xyz_discrete, theta_in_out, xyz_in_out in zip([xyz_discrete_in, xyz_discrete_out], [theta_xy_in, theta_xy_out], xyz_in_out_voi):

            for dim, theta in zip([0, 1], theta_in_out):

                xyz_discrete[dim] = abs(z_discrete - xyz_in_out[:, 0, 2]) * torch.tan(theta) + xyz_in_out[:, 0, dim]

            xyz_discrete[2] = z_discrete

        return xyz_discrete_in, xyz_discrete_out

    @staticmethod
    def _find_sub_volume(n_mu: int, voi: VolumeInterest, xyz_in_voi: Tensor, xyz_out_voi: Tensor) -> List[List[Tensor]]:

        r"""
        Find the xy voxel indices of the sub-volume which contains both incoming and outgoing tracks.

        Arguments:
            n_mu: int the number muons included in the inference.
            voi: VolumeInterest an instance of the VolumeInterest class.
            xyz_in_voi: Tensor The location of muons when entering/exiting the volume for the incoming track.
            xyz_out_voi: Tensor The location of muons when entering/exiting the volume for the outgoing track.

        Returns:
            sub_vol_indices_min_max: List[List[Tensor]] containing the voxel indices.
        """

        print("\nSub-volumes")
        sub_vol_indices_min_max = []

        for event in progress_bar(range(n_mu)):

            x_min = torch.min(torch.min(xyz_in_voi[event, :, 0]), torch.min(xyz_out_voi[event, :, 0]))
            x_max = torch.max(torch.max(xyz_in_voi[event, :, 0]), torch.max(xyz_out_voi[event, :, 0]))

            y_min = torch.min(torch.min(xyz_in_voi[event, :, 1]), torch.min(xyz_out_voi[event, :, 1]))
            y_max = torch.max(torch.max(xyz_in_voi[event, :, 1]), torch.max(xyz_out_voi[event, :, 1]))

            sub_vol_indices = (
                (voi.voxel_edges[:, :, 0, 1, 0] > x_min)
                & (voi.voxel_edges[:, :, 0, 1, 1] > y_min)
                & (voi.voxel_edges[:, :, 0, 0, 0] < x_max)
                & (voi.voxel_edges[:, :, 0, 0, 1] < y_max)
            ).nonzero()

            if len(sub_vol_indices) != 0:
                sub_vol_indices_min_max.append([sub_vol_indices[0], sub_vol_indices[-1]])

            else:
                sub_vol_indices_min_max.append([])

        return sub_vol_indices_min_max

    @staticmethod
    def _find_triggered_voxels(
        voxel_edges: Tensor, sub_vol_indices_min_max: List[List[Tensor]], xyz_discrete_in: Tensor, xyz_discrete_out: Tensor
    ) -> List[List[Tensor]]:

        r"""
        For each muon incoming and outgoing tracks, find the associated triggered voxels.
        Only voxels triggered by both INCOMING and OUTGOING tracks are kept.

        Arguments:
            voxel_edges: (x,y,z,2,3) tensor of voxels edges
            sub_vol_indices_min_max: List[Tensor] the xy voxel indices of the sub-volume which contains both incoming and outgoing tracks.
            xyz_discrete_in: (3, n_points, n_mu) tensor of discretized incoming tracks
            xyz_discrete_out: (3, n_points, n_mu) tensor of discretized outgoing tracks

        Returns:
            triggered_voxels: List[Tensor] with length n_mu. Each tensor has size (Ni,3) with Ni the number of triggered voxel for muon i.

        """

        triggered_voxels = []

        print("\nVoxel triggering")
        for event, sub_vol_indice in enumerate(progress_bar(sub_vol_indices_min_max)):
            if len(sub_vol_indice) != 0:

                ix_min, iy_min = sub_vol_indice[0][0], sub_vol_indice[0][1]
                ix_max, iy_max = sub_vol_indice[1][0], sub_vol_indice[1][1]

                sub_voi_edges = voxel_edges[ix_min : ix_max + 1, iy_min : iy_max + 1]
                sub_voi_edges = sub_voi_edges[:, :, :, :, None, :].expand(-1, -1, -1, -1, xyz_discrete_out.size()[1], -1)

                sub_mask_in = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_discrete_in[0, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_discrete_in[0, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_discrete_in[1, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_discrete_in[1, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_discrete_in[2, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_discrete_in[2, :, event])
                )

                sub_mask_out = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_discrete_out[0, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_discrete_out[0, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_discrete_out[1, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_discrete_out[1, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_discrete_out[2, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_discrete_out[2, :, event])
                )

                vox_list = (sub_mask_in & sub_mask_out).nonzero()[:, :-1].unique(dim=0).detach()
                vox_list[:, 0] += ix_min
                vox_list[:, 1] += iy_min
                triggered_voxels.append(vox_list)
            else:
                triggered_voxels.append([])

        return triggered_voxels

    def _get_triggered_voxels(self, theta_xy_in: Tensor, theta_xy_out: Tensor, hits: Tensor) -> List[List[Tensor]]:
        r"""
        Find the voxels triggered by each individual muon for the scatter batches added.

        Returns:
            List[Tensor] with length n_mu. Each tensor has size (Ni,3) with Ni the number of triggered voxel for muon i.
        """

        xyz_in_out_voi = self._compute_xyz_in_out(
            n_panels=self.n_panels,
            voi=self.voi,
            theta_xy_in=(theta_xy_in[:, 0], theta_xy_in[:, 1]),
            theta_xy_out=(theta_xy_out[:, 0], theta_xy_out[:, 1]),
            hits=hits,
        )

        xyz_discrete_in, xyz_discrete_out = self._compute_discrete_track(
            n_mu=self.n_mu,
            voi=self.voi,
            theta_xy_in=(theta_xy_in[:, 0], theta_xy_in[:, 1]),
            theta_xy_out=(theta_xy_out[:, 0], theta_xy_out[:, 1]),
            xyz_in_out_voi=xyz_in_out_voi,
        )

        sub_vol_indices_min_max = self._find_sub_volume(n_mu=self.n_mu, voi=self.voi, xyz_in_voi=xyz_in_out_voi[0], xyz_out_voi=xyz_in_out_voi[1])

        return self._find_triggered_voxels(
            voxel_edges=self.voi.voxel_edges,
            sub_vol_indices_min_max=sub_vol_indices_min_max,
            xyz_discrete_in=xyz_discrete_in,
            xyz_discrete_out=xyz_discrete_out,
        )

    def _get_voxel_zxy_density_preds(self) -> Tensor:

        r"""
        Computes the density predictions per voxel using the scatter batches added.

        Returns:
            (z,x,y) tensor of voxelwise density predictions
        """
        # Associate an empty list to each voxel
        score_list = [
            [[[] for _ in range(self.voi.n_vox_xyz[2])] for _ in range(self.voi.n_vox_xyz[1])] for _ in range(self.voi.n_vox_xyz[0])
        ]  # type: List[List[List[List]]]

        # get triggered voxels
        triggered_voxels = self._get_triggered_voxels(hits=self.gen_hits, theta_xy_in=self.theta_xy_in, theta_xy_out=self.theta_xy_out)

        # Get scores
        score = self.score

        print("\nAssigning voxels score")
        for i, vox_list in enumerate(progress_bar(triggered_voxels)):
            for vox in vox_list:
                score_list[vox[0]][vox[1]][vox[2]].append(score[i])

        vox_density_preds = torch.zeros(tuple(self.voi.n_vox_xyz))

        print("Compute final score")
        for i in progress_bar(range(self.voi.n_vox_xyz[0])):
            for j in range(self.voi.n_vox_xyz[1]):
                for k in range(self.voi.n_vox_xyz[2]):
                    if score_list[i][j][k] != []:
                        vox_density_preds[i, j, k] = self.score_method(score_list[i][j][k])

        # Check NaNs
        if vox_density_preds.isnan().any():
            raise ValueError("Prediction contains NaN values")

        return vox_density_preds.transpose(2, 0).transpose(1, 2)

    def _get_voxel_zxy_density_pred_uncs(self) -> Tensor:
        r"""
        Computes the uncertainty on the predicted voxelwise X0s, via gradient-based error propagation.

        .. warning::
            This method is not implemented yet

        Returns:
            (z,x,y) tensor of uncertainties on voxelwise X0s
        """

        vox_density_pred_uncs = torch.zeros(tuple(self.voi.n_vox_xyz)).transpose(2, 0).transpose(1, 2)

        return vox_density_pred_uncs

    def _reset_vars(self) -> None:

        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self._n_mu = None
        self._muon_scatter_vars = None  # (mu, vars)
        self._muon_scatter_var_uncs = None  # (mu, vars)
        self._muon_efficiency = None  # (mu, eff)
        self._vox_zxy_density_preds = None  # (z, x, y)
        self._vox_zxy_density_preds_uncs = None  # (z, x, y)
        self._gen_hits = None  # (mu, n_panels, 3)

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
        self._tot_scatter_dim = dims["tot_scatter"]
        self._theta_xy_in_dim = dims["theta_xy_in"]
        self._theta_xy_out_dim = dims["theta_xy_out"]
        self._mom_dim = dims["mom"]
        self._track_in_dim = dims["track_in"]
        self._track_out_dim = dims["track_out"]

    def _combine_scatters(self) -> None:
        r"""
        Combines scatter data from all the batches added so far.
        Any muons with NaN or Inf entries will be filtered out of the resulting tensors.
        Any muons with scattering angle out of dtheta_range will be filtered out of the resulting tensors.
        """

        vals: Dict[str, Tensor] = {}
        uncs: Dict[str, Tensor] = {}

        if len(self.scatter_batches) == 0:
            raise ValueError("No scatter batches have been added")

        vals["tot_scatter"] = torch.cat([sb.total_scatter for sb in self.scatter_batches], dim=0)
        uncs["tot_scatter"] = torch.cat([sb.total_scatter_unc for sb in self.scatter_batches], dim=0)
        vals["theta_xy_in"] = torch.cat([sb.theta_xy_in for sb in self.scatter_batches], dim=0)
        uncs["theta_xy_in"] = torch.cat([sb.theta_xy_in_unc for sb in self.scatter_batches], dim=0)
        vals["theta_xy_out"] = torch.cat([sb.theta_xy_out for sb in self.scatter_batches], dim=0)
        uncs["theta_xy_out"] = torch.cat([sb.theta_xy_out_unc for sb in self.scatter_batches], dim=0)
        vals["mom"] = torch.cat([sb.mu.mom[:, None] for sb in self.scatter_batches], dim=0)
        uncs["mom"] = torch.zeros_like(vals["mom"])
        vals["track_in"] = torch.cat([sb.track_in for sb in self.scatter_batches], dim=0)
        vals["track_out"] = torch.cat([sb.track_out for sb in self.scatter_batches], dim=0)
        vals["gen_hits"] = torch.cat([sb.gen_hits for sb in self.scatter_batches], dim=0)

        mask = torch.ones(len(vals["mom"])).bool()
        for var_sz in self._var_order_szs:
            mask *= ~(vals[var_sz[0]].isnan().any(1))
            mask *= ~(vals[var_sz[0]].isinf().any(1))
            if (var_sz[0] != "track_in") & (var_sz[0] != "track_out"):
                mask *= ~(uncs[var_sz[0]].isnan().any(1))
                mask *= ~(uncs[var_sz[0]].isinf().any(1))

        mask *= (vals["tot_scatter"][:, 0] > self.dtheta_range[0]) & (vals["tot_scatter"][:, 0] < self.dtheta_range[1])
        self._muon_scatter_vars = torch.cat([vals[var_sz[0]][mask] for var_sz in self._var_order_szs], dim=1)  # (mu, vars)
        self._muon_efficiency = torch.cat([self.compute_efficiency(scatters=sb) for sb in self.scatter_batches], dim=0)[mask]  # (mu, eff)
        self._n_mu = len(self._muon_scatter_vars)
        self._gen_hits = vals["gen_hits"][mask, :, :]

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
        Computes the predicted normalized scattering density per voxel as a (z,x,y) tensor for the provided scatter batches.

        Returns:
            pred: (z,x,y) normalized voxelwise scattering density per voxel
            inv_weight: sum of muon efficiencies
        """
        if len(self.scatter_batches) == 0:
            print("Warning: unable to scan volume with prescribed number of muons.")
            return None, None

        if None in self._asr_params.values():
            raise ValueError("ASR parameters were not setup. Make sure to use the tomopt.optimisation.callbacks.ParamsASR callabck")

        if self.voi is None:
            raise ValueError("Voxelized volume was not setup. Make sure to use the tomopt.optimisation.callbacks.ParamsASR callabck")

        return self.vox_zxy_density_preds, self.inv_weights

    def set_params(self, score_method: partial, dtheta_range: Tuple[float, float], use_p: bool) -> None:
        r"""
        Set AngleStatisticReconstruction parameters.

        Arguments:
            score:method: partial method for voxel-wise scattering density computation
            dtheta_range: Tuple[float] The range of muon scattering angle. Events outside this range will not be included in inference.
            use_p: bool  Momentum information flag. If False, no momentum information is used during inference.
        """

        self._asr_params["dtheta_range"] = dtheta_range
        self._asr_params["score_method"] = score_method
        self._asr_params["use_p"] = use_p

    def reset_params(self) -> None:
        r"""
        Reset AngleStatisticReconstruction parameters
        """
        self._asr_params["dtheta_range"] = None
        self._asr_params["score_method"] = None
        self._asr_params["use_p"] = None

    def set_voi(self, voi: VolumeInterest) -> None:
        r"""
        Set AngleStatisticReconstruction voxelized volume

        Arguments:
            voi: VolumeInterest an instance of the VolumeInterest class.
        """
        self._voi = voi

    def reset_voi(self) -> None:
        r"""
        Reset AngleStatisticReconstruction voxelized volume
        """
        self._voi = None

    @property
    def vox_zxy_density_preds(self) -> Tensor:
        r"""
        Returns:
            (z,x,y) tensor of voxelwise X0 predictions
        """
        if self._vox_zxy_density_preds is None:
            self._vox_zxy_density_preds = self._get_voxel_zxy_density_preds()
            self._vox_zxy_density_preds_uncs = None
        return self._vox_zxy_density_preds

    @property
    def vox_zxy_density_pred_uncs(self) -> Tensor:
        r"""
        .. warning::
            Not implemented yet.

        Returns:
            (z,x,y) tensor of uncertainties on voxelwise X0s
        """

        if self._vox_zxy_density_preds_uncs is None:
            self._vox_zxy_density_preds_uncs = self._get_voxel_zxy_density_pred_uncs()
        return self._vox_zxy_density_preds_uncs

    @property
    def inv_weights(self) -> Tensor:
        r"""
        Returns:
            Sum of muon efficiencies
        """
        return self.muon_efficiency.sum()

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
    def theta_xy_in(self) -> Tensor:
        r"""
        Returns:
            The incoming muon projected zenith angle in XZ and YZ plane
        """
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._theta_xy_in_dim]

    @property
    def theta_xy_out(self) -> Tensor:
        r"""
        Returns:
            The outgoing muon projected zenith angle in XZ and YZ plane
        """
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._theta_xy_out_dim]

    @property
    def gen_hits(self) -> Tensor:
        r"""
        Returns:
            The generated muon hits
        """
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._gen_hits

    @property
    def track_in(self) -> Tensor:
        r"""
        Returns:
            The incoming track
        """
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._track_in_dim]

    @property
    def track_out(self) -> Tensor:
        r"""
        Returns:
            The outgoing track
        """
        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._track_out_dim]

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
    def muon_mom(self) -> Tensor:
        r"""
        Returns:
            (muons,1) tensor of the momenta of the muons
        """

        if self._muon_scatter_vars is None or self._muon_scatter_var_uncs is None:
            self._combine_scatters()
        return self._muon_scatter_vars[:, self._mom_dim]

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
    def score_method(self) -> partial:
        r"""
        Returns:
            Method for voxel-wise scattering density computation
        """

        return self._asr_params["score_method"]

    @property
    def use_p(self) -> bool:
        r"""
        Returns:
            Momentum information flag. If False, no momentum information is used during inference.
        """
        return self._asr_params["use_p"]

    @property
    def dtheta_range(self) -> Tuple[float, float]:
        r"""
        Returns:
            The range of muon scattering angle. Events outside this range will not be included in inference.
        """
        return self._asr_params["dtheta_range"]

    @property
    def voi(self) -> VolumeInterest:
        r"""
        Returns:
            The voxelized volume
        """
        return self._voi

    @property
    def score(self) -> np.ndarray:
        r"""
        Returns:
            (muons, 1) the muon wise scores
        """
        if self.use_p:
            return (self.muon_total_scatter * self.muon_mom).detach().numpy()
        else:
            return self.muon_total_scatter.detach().numpy()
