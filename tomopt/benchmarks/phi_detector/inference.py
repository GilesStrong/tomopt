# from typing import Tuple

# import torch
# from torch import Tensor

from ...inference.scattering import AbsScatterBatch

__all__ = ["PhiDetScatterBatch"]


class PhiDetScatterBatch(AbsScatterBatch):
    pass
    # @staticmethod
    # def get_muon_trajectory(hits: Tensor, uncs: Tensor, lw: Tensor) -> Tuple[Tensor, Tensor]:
    #     r"""
    #     hits = (muons,panels,(x,y,z))
    #     uncs = (muons,panels,(unc,unc,0))

    #     Assume no uncertainty for z

    #     In eval mode:
    #         Muons with <2 hits within panels have NaN trajectory.
    #         Muons with >=2 hits in panels have valid trajectories
    #     """

    #     hits = torch.where(torch.isinf(hits), lw.mean().type(hits.type()) / 2, hits)
    #     uncs = torch.nan_to_num(uncs)  # Set Infs to large number
