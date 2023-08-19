import json
import os
from typing import Dict, Optional, Union

import h5py
import torch
from fastcore.all import Path
from torch import Tensor

r"""
Provides container classes for different scattering models.

To save on memory and loading time, each scatter model is only loaded once, and all :class:`~tomopt.volume.layer.AbsLayer` will use the same object.
To avoid loading whenever this file is accessed, the models begin uninitialised, and will only load data when their `load_data` method is called.
"""

PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # How robust is this? Could create hidden dir in home and download resources

__all__ = ["PGEANT_SCATTER_MODEL"]


class PGeantScatterModel:
    r"""
    Class implementing access to the parameterised GEANT 4 model,
    as stored in an HDF5 file created by code in https://github.com/GilesStrong/mode_muon_tomography_scattering.
    The data file should be included in the TomOpt repository, and the package as installed by PIP.

    Data in the file is loaded into tensors and act as a lookup table to provide scattering in terms of angles (`dtheta_params`) and positions (`dxy_params`).
    These tensors have shapes (8, 35, 10000), corresponding to (material type, momentum bin, scattering),
    where 'scattering' is designed to be indexed by uniformly distributed random indices.
    The materials and momentum bin edges are included in meta data in the HDF5 file.

    .. warning::
        Currently no interpolation of X0s, or other handling for scattering in materials that were not considered in the fitting of the model, is performed.
        Instead, 'missing' materials are mapped to the nearest X0 that was used when fitting the model.

    To avoid unnecessary loading of data, the model begins uninitialised, and will only load data when `load_data` method is called.
    Prior to this, certain class attributes relating to the scattering model will not defined.
    """

    dtheta_params: Tensor  # (8, 35, 10000), (x0, mom, rnd)
    dxy_params: Tensor  # (8, 35, 10000), (x0, mom, rnd)
    step_sz: float
    x02id: Dict[float, int]
    mom_lookup: Tensor
    exp_disp_model: float
    n_bins: int
    _device: Optional[torch.device] = None

    def __init__(self) -> None:
        r"""
        To avoid unnecessary loading of data, the model begins "uninitialised", and will only load data when `load_data` method is called.
        Prior to this, certain class attributes relating to the scattering model will not defined.
        """

        self.initialised = False

    def load_data(self, filename: str = "scatter_models/dt_dx_10mm.hdf5") -> None:
        r"""
        Loads scatter-model data into memory, and defines the values of the class attributes.
        This must be called prior to calling the `compute_scattering` method.

        Arguments:
            filename: the name and location of the scattering data relative to the location of this file
        """

        self.file = h5py.File(PKG_DIR / filename, "r")
        self.dtheta_params = Tensor(self.file["model/dtheta"][()])
        self.dxy_params = Tensor(self.file["model/dxy"][()])
        self.step_sz = self.file["meta_data/deltaz"][()]
        self.x02id = {float(k): v for k, v in json.loads(self.file["meta_data/x02id"][()]).items()}
        self.mom_lookup = Tensor(self.file["meta_data/mom_bins"][1:-1])  # Exclude starting and end edges
        self.exp_disp_model = self.file["meta_data/exp_disp_model"][()]
        self.n_bins = self.file["meta_data/n_rnd"][()]
        self.initialised = True

    def extrapolate_dtheta(self, dtheta_xy_mu: Tensor, inv_costheta: Tensor) -> Tensor:
        r"""
        The model is fitted with an expected distance of travel, but due to the theta of the incoming muon, this may not always be the correct flight distance.
        This function extrapolates the angular scattering to account for these differences.

        Arguments:
            dtheta_xy_mu: (2,N) tensor of the angular scatterings in the muon frames, as sampled from the model
            inv_costheta: (N,) tensor of 1/cosine(theta) for the muons in the volume frame

        Returns:
            (2,N) tensor of the modified angular scatterings in the muon frames
        """

        return dtheta_xy_mu * torch.sqrt(inv_costheta)

    def extrapolate_dxy(self, dxy_mu: Tensor, inv_costheta: Tensor) -> Tensor:
        r"""
        The model is fitted with an expected distance of travel, but due to the theta of the incoming muon, this may not always be the correct flight distance.
        This function extrapolates the spatial scattering to account for these differences.

        Arguments:
            dxy_mu: (2,N) tensor of the spatial scatterings in the muon frames, as sampled from the model
            inv_costheta: (N,) tensor of 1/cosine(theta) for the muons in the volume frame

        Returns:
            (2,N) tensor of the modified spatial scatterings in the muon frames
        """

        return dxy_mu * (inv_costheta**self.exp_disp_model)

    def compute_scattering(self, x0: Tensor, step_sz: Union[Tensor, float], theta: Tensor, theta_x: Tensor, theta_y: Tensor, mom: Tensor) -> Dict[str, Tensor]:
        r"""
        Computes the scattering of the muons using the parameterised GEANT 4 model.

        Arguments:
            x0: (N,) tensor of the X0 of the voxel each muon is traversing
            step_sz: The step size in metres over which to compute muon scattering.
            theta: (N,) tensor of the theta angles of the muons. This is used to compute the total flight path of the muons
            theta_x: (N,) tensor of the theta_x angles of the muons. This is used to map the dx displacements from the muons' frame to the volume's
            theta_y: (N,) tensor of the theta_y angles of the muons. This is used to map the dy displacements from the muons' frame to the volume's
            mom: (N,) tensor of the absolute value of the momentum of each muon

        Returns:
            A dictionary of muon scattering variables in the volume reference frame: dtheta_vol, dphi_vol, dx_vol, & dy_vol
        """

        raise NotImplementedError("PGeant is not recommended as it needs to be updated for the new muon scattering treatment")

        if not self.initialised:
            self.load_data()  # Delay loading until required
        if self._device is None:
            self.device = theta.device

        n = len(x0)
        rnds = torch.randint(low=0, high=self.n_bins, size=(4, n), device=self.device).long()  # dtheta_x, dtheta_x, dy, dy
        mom_idxs = torch.bucketize(mom, self.mom_lookup).long()
        sign = (torch.rand((4, n), device=self.device) - 0.5).sign()

        x0_idxs = torch.zeros_like(mom, device=self.device) - 1
        for m in x0.detach().unique().cpu().numpy():
            x0_idxs[x0 == m] = self.x02id[min(self.x02id, key=lambda x: abs(x - m))]  # Get ID for closest X0 in model
        if (x0_idxs < 0).any():
            raise ValueError("Something went wrong in the x0 indexing")
        x0_idxs = x0_idxs.long()

        inv_costheta = 1 / (1e-17 + torch.cos(theta))
        dtheta_xy_mu = self.dtheta_params[x0_idxs, mom_idxs, rnds[:2]]
        dtheta_xy_mu = self.extrapolate_dtheta(dtheta_xy_mu, inv_costheta)
        dtheta_xy_mu = sign[:2] * dtheta_xy_mu
        # We compute dtheta_xy in muon ref frame, but we're free to rotate the muon,
        # since dtheta_xy doesn't depend on muon position
        # Therefore assign theta_x axis (muon ref) to be in the theta direction (vol ref),
        # and theta_y axis (muon ref) to be in the phi direction (vol ref)
        dtheta_vol = dtheta_xy_mu[0]  # dtheta_x in muon ref
        dphi_vol = dtheta_xy_mu[1]  # dtheta_y in muon ref

        dxy_mu = self.dxy_params[x0_idxs, mom_idxs, rnds[2:]]
        dxy_mu = self.extrapolate_dxy(dxy_mu, inv_costheta)
        dxy_mu = sign[2:] * dxy_mu
        # dxy is still in the muon ref frame, rescale by costheta_xy into volume frame
        dx_vol = dxy_mu[0] * torch.cos(theta_x)
        dy_vol = dxy_mu[1] * torch.cos(theta_y)

        return {"dtheta_vol": dtheta_vol, "dphi_vol": dphi_vol, "dx_vol": dx_vol, "dy_vol": dy_vol}

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.dtheta_params = self.dtheta_params.to(self._device)  # Is this dtheta_xy in the muon's ref frame = dtheta dphi in the volume ref frame?
        self.dxy_params = self.dxy_params.to(self._device)  # Is this already in the volume frame or still in the muon's ref frame?
        self.mom_lookup = self.mom_lookup.to(self._device)


PGEANT_SCATTER_MODEL = PGeantScatterModel()
