from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ...inference.scattering import ScatterBatch
from .detector import PhiDetectorPanel

__all__ = ["PhiDetScatterBatch"]


class PhiDetScatterBatch(ScatterBatch):
    r"""
    Untested: no unit tests
    """

    def _extract_hits(self) -> None:
        # reco x, reco y, gen z, must be a list to allow computation of uncertainty
        above_hits = torch.stack(
            [
                torch.cat([self.hits["above"]["reco_h"][:, i], self.hits["above"]["phi"][:, i], self.hits["above"]["z"][:, i]], dim=-1)
                for i in range(self.hits["above"]["reco_h"].shape[1])
            ],
            dim=1,
        )  # muons, panels, xyz
        below_hits = torch.stack(
            [
                torch.cat([self.hits["below"]["reco_h"][:, i], self.hits["below"]["phi"][:, i], self.hits["below"]["z"][:, i]], dim=-1)
                for i in range(self.hits["below"]["reco_h"].shape[1])
            ],
            dim=1,
        )
        _above_gen_hits = torch.stack(
            [torch.cat([self.hits["above"]["gen_xyz"][:, i], self.hits["above"]["z"][:, i]], dim=-1) for i in range(self.hits["above"]["gen_xyz"].shape[1])],
            dim=1,
        )  # muons, panels, xyz
        _below_gen_hits = torch.stack(
            [torch.cat([self.hits["below"]["gen_xyz"][:, i], self.hits["below"]["z"][:, i]], dim=-1) for i in range(self.hits["below"]["gen_xyz"].shape[1])],
            dim=1,
        )
        self._n_hits_above = above_hits.shape[1]
        self._n_hits_below = below_hits.shape[1]

        # Combine all input vars into single tensor, NB ideally would stack to new dim but can't assume same number of panels above & below
        self._reco_hits = torch.cat((above_hits, below_hits), dim=1)  # muons, all panels, reco h,phi,z
        self._gen_hits = torch.cat((_above_gen_hits, _below_gen_hits), dim=1)  # muons, all panels, true xyz

    def plot_scatter(self, idx: int, savename: Optional[Path]) -> None:
        raise NotImplementedError("Ah, I see you've just volunteered to implement this!")

    @staticmethod
    def _get_hit_uncs(zordered_panels: List[PhiDetectorPanel], hits: Tensor) -> Tensor:
        uncs: List[Tensor] = []
        for l, h in zip(zordered_panels, hits.unbind(1)):
            if not isinstance(l.resolution, Tensor):
                raise ValueError(f"{l.resolution} is not a Tensor for some reason.")  # To appease MyPy
            r = 1 / l.resolution
            uncs.append(torch.cat([r, torch.zeros((len(r), 2), device=r.device)], dim=-1))
        return torch.stack(uncs, dim=1)  # muons, panels, unc h,phi,z, zero unc for phi and z

    def _compute_tracks(self) -> None:
        def _get_panels() -> List[PhiDetectorPanel]:
            panels: List[PhiDetectorPanel] = []
            for det in self.volume.get_detectors():
                if not isinstance(det, PhiDetectorPanel):
                    raise ValueError(f"Detector {det} is not a PhiDetectorPanel")
                panels += [det.panels[j] for j in det.get_panel_zorder()]
            return panels

        self._hit_uncs = self._get_hit_uncs(_get_panels(), self.gen_hits)
        self._track_in, self._track_start_in = self.get_muon_trajectory(self.above_hits, self.above_hit_uncs, self.volume.lw)
        self._track_out, self._track_start_out = self.get_muon_trajectory(self.below_hits, self.below_hit_uncs, self.volume.lw)

    @staticmethod
    def get_muon_trajectory(hits: Tensor, uncs: Tensor, lw: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        hits = (muons,panels,(h,phi,z))
        uncs = (muons,panels,(unc,0,0))

        Assumes no uncertainty for z and phi

        Uses an analytic likelihood-maximisation: L = \prod_i G[h_i - h_{i,opt}(x0,y0,theta,phi)]
        where:
        h_{i,opt} = x_{i,opt} cos(phi_i) + y_{i,opt} sin(phi_i)
        x_{i,opt} = x0 + z_i tan(theta) cos(phi)
        y_{i,opt} = y0 + z_i tan(theta) sin(phi)

        x0, y0, z0 = track coordinates at z=0
        theta, phi = track angles

        x0 = sum_i[ ((cos(phi_i)/(unc_i^2))*(h_i-(z_i*tan(theta)*cos(phi)*cos(phi_i))-(sin(phi_i)*(y0+(z_i*tan(theta)*sin(phi))))) ] / sum_i[ ((cos(phi_i)^2)/(unc_i^2)) ]
        y0 = sum_i[ ((sin(phi_i)/(unc_i^2))*(h_i-(z_i*tan(theta)*sin(phi)*cos(phi_i))-(cos(phi_i)*(x0+(z_i*tan(theta)*sin(phi))))) ] / sun_i[ ((sin(phi_i)^2)/(unc_i^2)) ]
        theta = tan^-1[ sum_i[ (z_i/(unc_i^2))*((h_i*((cos(phi)*cos(phi_i))+(sin(phi)*sin(phi_i))))-(x0*((cos(phi_i)*sin(phi_i)*sin(phi))+(cos(phi)*cos(phi_i)*cos(phi_i))))-(y0((cos(phi)*cos(phi_i)*sin(phi_i))+(sin(phi)*sin(phi_i)*sin(phi_i))))) ] / sum_i[ ((z_i^2)/(unc_i^2))*((cos(phi)*cos(phi_i))+(sin(phi)*sin(sin(phi_i)))) ] ]
        phi = sin^-1[ ((z_i*sin(phi_i))/(unc_i^2))*(h_i-(y0*sin(phi_i))-(x0*cos(phi_i))-(z_i*tan(theta)*cos(phi)*cos(phi_i))) ] / sum_i[ (((z_i*sin(phi_i))^2)/(unc_i^2))*tan(theta) ]

        In eval mode:
            Muons with <2 hits within panels have NaN trajectory.
            Muons with >=2 hits in panels have valid trajectories
        """
        raise NotImplementedError("We still need to work out how to fit tracks efficiently!")

        # hits = torch.where(torch.isinf(hits), lw.mean().type(hits.type()) / 2, hits)
        # uncs = torch.nan_to_num(uncs)  # Set Infs to large number

        # x0: Tensor
        # y0: Tensor  # Track positions at Z0=0
        # phi: Tensor
        # theta: Tensor  # Track angles
