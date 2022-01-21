from typing import Dict, Union, Tuple

import torch
from torch import Tensor

__all__ = ["SCATTER_MODEL"]


class ScatterModel:
    dtheta_params: Tensor  # (10000, 35, 8), (rnd, mom, x0)
    dxy_params: Tensor  # (10000, 35, 8), (rnd, mom, x0)
    deltaz: float
    x02id: Dict[float, int]
    mom_lookup: Tensor
    exp_disp_model: float
    n_bins: int

    """
    SCATTER_PARAMS: Tensor = torch.ones((2, 10000, 35, 8))  # Load params from ".scattering.file" (2, 10000, 35, 8), (dtheta|dxy, rnd, mom, x0)
    X02ID = {X0[m]: i for i, m in enumerate(X0)}
    MOM_LOOKUP = torch.tensor([0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10., 15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 150., 200., 250., 300., 350., 400.], device=DEVICE)
    EXP_DISP_MODEL: float = 1.85
    MODEL_DELTA_Z: float = 0.01
    """

    def __init__(self) -> None:
        self.initialised = False

    def load_data(self, filename: str = "scatter_params.hdf5") -> None:
        # self.dtheta_params
        # self.dxy_params
        # self.deltaz
        # self.x02id
        # self.mom_lookup
        # self.exp_disp_model
        self.initialised = True

    def extrapolate_dtheta(self, dtheta: Tensor, inv_costheta: Tensor) -> Tensor:
        return dtheta * torch.sqrt(inv_costheta)

    def extrapolate_dxy(self, dxy: Tensor, inv_costheta: Tensor) -> Tensor:
        return dxy * (inv_costheta ** self.exp_disp_model)

    def compute_scattering(self, x0: Tensor, deltaz: Union[Tensor, float], theta_xy: Tensor, mom: Tensor) -> Tuple[Tensor, Tensor]:
        if deltaz != self.deltaz:
            raise ValueError(f"Model only works for a fixed delta z step of {self.deltaz}.")

        n = len(x0)
        rnds = (self.n_bins * torch.rand((n, 4), device=theta_xy.device)).long()  # dtheta_x, dtheta_x, dy, dy
        mom_idxs = torch.bucketize(mom, self.mom_lookup)

        x0_idxs = torch.zeros_like(mom, device=theta_xy.device) - 1
        for m in x0.detach().unique().cpu().numpy():
            x0_idxs[x0 == m] = self.x02id[min(self.x02id, key=lambda x: abs(x - m))]  # Get ID for closest X0 in model
        if (x0_idxs < 0).any():
            raise ValueError("Something went wrong in the x0 indexing")

        inv_costheta = 1 / (1e-17 + torch.cos(theta_xy))
        dtheta = self.dtheta_params[rnds[:, 0:2], mom_idxs, x0_idxs]
        dtheta = self.extrapolate_dtheta(dtheta, inv_costheta)
        dxy = self.dxy_params[rnds[:, 2:4], mom_idxs, x0_idxs]
        dxy = self.extrapolate_dxy(dxy, inv_costheta)

        return dtheta, dxy


SCATTER_MODEL = ScatterModel()
