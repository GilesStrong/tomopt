from typing import Dict, Optional, Union
import h5py
import os
from fastcore.all import Path
import json

import torch
from torch import Tensor

PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # How robust is this? Could create hidden dir in home and download resources

__all__ = ["SCATTER_MODEL"]


class ScatterModel:
    dtheta_params: Tensor  # (8, 35, 10000), (x0, mom, rnd)
    dxy_params: Tensor  # (8, 35, 10000), (x0, mom, rnd)
    deltaz: float
    x02id: Dict[float, int]
    mom_lookup: Tensor
    exp_disp_model: float
    n_bins: int
    _device: Optional[torch.device] = None

    def __init__(self) -> None:
        self.initialised = False

    def load_data(self, filename: str = "scatter_models/dt_dx_10mm.hdf5") -> None:
        self.file = h5py.File(PKG_DIR / filename, "r")
        self.dtheta_params = Tensor(self.file["model/dtheta"][()])
        self.dxy_params = Tensor(self.file["model/dxy"][()])
        self.deltaz = self.file["meta_data/deltaz"][()]
        self.x02id = {float(k): v for k, v in json.loads(self.file["meta_data/x02id"][()]).items()}
        self.mom_lookup = Tensor(self.file["meta_data/mom_bins"][1:-1])  # Exclude starting and end edges
        self.exp_disp_model = self.file["meta_data/exp_disp_model"][()]
        self.n_bins = self.file["meta_data/n_rnd"][()]
        self.initialised = True

    def extrapolate_dtheta(self, dtheta_xy_mu: Tensor, inv_costheta: Tensor) -> Tensor:
        return dtheta_xy_mu * torch.sqrt(inv_costheta)

    def extrapolate_dxy(self, dxy_mu: Tensor, inv_costheta: Tensor) -> Tensor:
        return dxy_mu * (inv_costheta**self.exp_disp_model)

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.dtheta_params = self.dtheta_params.to(self._device)  # Is this dtheta_xy in the muon's ref frame = dtheta dphi in the volume ref frame?
        self.dxy_params = self.dxy_params.to(self._device)  # Is this alredy in the volume frame or still in the muon's ref frame?
        self.mom_lookup = self.mom_lookup.to(self._device)

    def compute_scattering(self, x0: Tensor, deltaz: Union[Tensor, float], theta_xy: Tensor, mom: Tensor) -> Dict[str, Tensor]:
        if deltaz != self.deltaz:
            raise ValueError(f"Model only works for a fixed delta z step of {self.deltaz}.")
        if self._device is None:
            self.device = theta_xy.device

        n = len(x0)
        rnds = torch.randint(low=0, high=self.n_bins, size=(n, 4), device=self.device).long()  # dtheta_x, dtheta_x, dy, dy
        mom_idxs = torch.bucketize(mom, self.mom_lookup).long()
        sign = (torch.rand((n, 4), device=self.device) - 0.5).sign()

        x0_idxs = torch.zeros_like(mom, device=self.device) - 1
        for m in x0.detach().unique().cpu().numpy():
            x0_idxs[x0 == m] = self.x02id[min(self.x02id, key=lambda x: abs(x - m))]  # Get ID for closest X0 in model
        if (x0_idxs < 0).any():
            raise ValueError("Something went wrong in the x0 indexing")
        x0_idxs = x0_idxs.long()

        inv_costheta = 1 / (1e-17 + torch.cos(theta_xy))
        dtheta_xy_mu = self.dtheta_params[x0_idxs, mom_idxs, rnds[:, :2]]
        dtheta_xy_mu = self.extrapolate_dtheta(dtheta_xy_mu, inv_costheta)
        dtheta_xy_mu = sign[:, :2] * dtheta_xy_mu
        # We compute dtheta_xy in muon ref frame, but we're free to rotate the muon,
        # since dtheta_xy doesn't depend on muon position
        # Therefore assign theta_x axis (muon ref) to be in the theta direction (vol ref),
        # and theta_y axis (muon ref) to be in the phi direction (vol ref)
        dtheta_vol = dtheta_xy_mu[:, 0]  # dtheta_x in muon ref
        dphi_vol = dtheta_xy_mu[:, 1]  # dtheta_y in muon ref

        dxy_mu = self.dxy_params[x0_idxs, mom_idxs, rnds[:, 2:]]
        dxy_mu = self.extrapolate_dxy(dxy_mu, inv_costheta)
        dxy_mu = sign[:, 2:] * dxy_mu
        # dxy is still in the muon ref frame, rescale by costheta_xy into volume frame
        dxy_vol = dxy_mu * torch.cos(theta_xy)

        return {"dtheta_vol": dtheta_vol, "dphi_vol": dphi_vol, "dx_vol": dxy_vol[:, 0], "dy_vol": dxy_vol[:, 1]}


SCATTER_MODEL = ScatterModel()
