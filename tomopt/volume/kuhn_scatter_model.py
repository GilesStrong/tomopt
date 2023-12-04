from typing import Dict, Optional
import math
import numpy as np
import random
import torch
from torch import Tensor


__all__ = ["KUHN_SCATTER_MODEL"]


class KuhnScatterModel:

    _device: Optional[torch.device] = None

    def __init__(self) -> None:
        return

    def compute_scattering(self, mom: Tensor, b: Tensor, Z_A_rho: Tensor, theta: Tensor, phi: Tensor, dz: float) -> Dict[str, Tensor]:

        if self._device is None:
            self.device = theta.device

        # if (mom.shape[0]<1):
        #     return {
        #         "dtheta_m" : Tensor([]),
        #         "dtheta_x_vol": Tensor([]),
        #         "dtheta_y_vol": Tensor([]),
        #         "dx_vol": Tensor([]),
        #         "dy_vol": Tensor([]),
        #         "dz_vol": Tensor([]),
        #         "dtheta_x_m": Tensor([]),
        #         "dtheta_y_m": Tensor([]),
        #         "dx_m": Tensor([]),
        #         "dy_m": Tensor([]),
        #     }

        # delta_s = dz*100/np.cos(theta)
        delta_s = dz * 100
        mom *= 1000
        # density = self.getZA_density(x0)[:,0]
        # Z = self.getZA_density(x0)[:,1]
        # A = self.getZA_density(x0)[:,2]

        Z = Z_A_rho[0]
        A = Z_A_rho[1]
        density = Z_A_rho[2]

        # B = self.newton_rhapson(mom, density, Z, A, delta_s)
        B = b
        # constant of kuhn model
        chi2 = 0.1569 * Z * (Z + 1) * density * delta_s / (mom**2 * A)

        # width of scattering angle distribution
        theta0 = torch.sqrt(chi2) * torch.sqrt(B - 1.25)

        # cumulative distribution function at pi
        P_pi = 1 - (0.827 / B) + (chi2 / 4) * (1 / (torch.sin(theta0 / np.sqrt(2)) ** 2) - 1)

        # todo: raise error if P_pi>1

        # random R in [0,P_pi]
        R_rand = torch.empty_like(P_pi)
        dtheta = torch.empty_like(R_rand)
        dphi = torch.empty_like(R_rand)

        for i in range(R_rand.shape[0]):
            R_rand[i] = random.uniform(0.0, P_pi[i])

            # dphi is random in [0,2*pi] for each muon
            dphi[i] = random.uniform(0.0, 2 * np.pi)

            # dtheta depends on condition on R_rand
            if R_rand[i] >= 1 - 0.827 / B[i]:
                R = R_rand[i] - 1 + 0.827 / B[i]
                dtheta[i] = 2 * torch.arcsin(
                    torch.sqrt(chi2[i]) * torch.sin(theta0[i] / np.sqrt(2)) / torch.sqrt(chi2[i] - 4 * R * np.sin(theta0[i] / np.sqrt(2)) ** 2)
                )

            elif R_rand[i] < 1 - 0.827 / B[i]:
                dtheta[i] = theta0[i] * torch.log((1 - 0.827 / B[i]) / (1 - (0.827 / B[i]) - R_rand[i])) ** 0.5

        # for i in range(R_rand.shape[0]):

        # dtheta=torch.clamp(dtheta, max=math.pi / 2.2)
        dtheta_x = np.arctan(np.tan(dtheta) * np.cos(dphi))
        # dtheta_x[(dtheta >= torch.pi / 2)] = torch.nan
        dtheta_y = np.arctan(np.tan(dtheta) * np.sin(dphi))
        # dtheta_y[(dtheta >= torch.pi / 2)] = torch.nan

        z1 = torch.randn((2, dtheta.shape[0]), device=self.device)
        z2 = torch.randn((2, dtheta.shape[0]), device=self.device)

        dx_m = (delta_s / 100) * theta0 * ((z1[0] / math.sqrt(12)) + (z2[0] / 2))
        dy_m = (delta_s / 100) * theta0 * ((z1[1] / math.sqrt(12)) + (z2[1] / 2))

        phi_defined = theta != 0  # If theta is a zero, there is no phi defined
        dx_vol = torch.where(phi_defined, -dx_m * torch.sin(phi) - dy_m * torch.cos(-theta) * torch.cos(phi), dx_m)
        dy_vol = torch.where(phi_defined, dx_m * torch.cos(phi) - dy_m * torch.cos(-theta) * torch.sin(phi), dy_m)
        dz_vol = torch.where(phi_defined, dy_m * torch.sin(-theta), theta.new_zeros(dy_m.shape))

        theta_x, theta_y = np.arctan(np.tan(theta) * np.cos(phi)), np.arctan(np.tan(theta) * np.sin(phi))

        ref_point = theta.new_ones([3, len(theta)])
        ref_point[0] = torch.tan(theta_x)
        ref_point[1] = torch.tan(theta_y)

        r = torch.sqrt(ref_point[0] ** 2 + ref_point[1] ** 2 + ref_point[2] ** 2)
        # 1 -
        dx = r * torch.tan(dtheta_x)
        dy = r * torch.tan(dtheta_y)
        # 2 -
        dx_vol_angle = torch.where(phi_defined, -dx * torch.sin(phi) - dy * torch.cos(theta) * torch.cos(phi), dx)
        dy_vol_angle = torch.where(phi_defined, dx * torch.cos(phi) - dy * torch.cos(theta) * torch.sin(phi), dy)
        dz_vol_angle = torch.where(phi_defined, dy * torch.sin(theta), torch.zeros_like(dy_m))
        # 3 -
        d_out = -ref_point
        d_out[0] = d_out[0] + dx_vol_angle
        d_out[1] = d_out[1] + dy_vol_angle
        d_out[2] = d_out[2] + dz_vol_angle
        dtheta_x_vol = torch.arctan(d_out[0] / d_out[2]) - theta_x
        dtheta_y_vol = torch.arctan(d_out[1] / d_out[2]) - theta_y

        return {
            "dtheta_m": dtheta,
            "dtheta_x_vol": dtheta_x_vol,
            "dtheta_y_vol": dtheta_y_vol,
            "dx_vol": dx_vol,
            "dy_vol": dy_vol,
            "dz_vol": dz_vol,
            "dtheta_x_m": dtheta_x,
            "dtheta_y_m": dtheta_y,
            "dx_m": dx_m,
            "dy_m": dy_m,
        }


KUHN_SCATTER_MODEL = KuhnScatterModel()
