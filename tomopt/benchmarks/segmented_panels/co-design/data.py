from functools import partial
from typing import Tuple

import torch
from torch import Tensor

from tomopt.benchmarks.segmented_panels.callbacks import PredHitRecord, PredPocaRecord
from tomopt.benchmarks.segmented_panels.detector_config import get_layers

# segmented panel imports
from tomopt.benchmarks.segmented_panels.wrapper import SegmentedPanelVolumeWrapper

# tomopt imports
from tomopt.core import X0
from tomopt.optimisation.data import PassiveYielder
from tomopt.optimisation.loss import VoxelX0Loss
from tomopt.volume import Volume


def generate_three_volumes_poca(
    *,
    panel_z_spacing: float = 0.1,
    init_res: float = 1e4,
    init_gap: float = 0.1,
    n_panels: int = 3,
    n_panels_seg: int = 2,
    n_mu: int = 1000,
    mu_bs: int = 100,
    block_mat: str = "uranium",
    bkg_mat: str = "aluminium",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate PoCA scattering angle ditributions for three volumes:
      - volume with uranium cube (signal)
      - background volume 1
      - background volume 2 (null for spurious KL)

    Parameters:
        panel_z_spacing: Separation in z axis between panels in a detector layer [m]
        init_res: Initial spatial resolution of panels [m^-1]
        init_gap: Initial gap between panel segments in xy plane [m]
        n_panels: Number of detector panels per detector layer
        n_panels_seg: Number of segments per panel in x or y directions
        n_mu: Number of muons to geenrate per volume
        mu_bs: Muon batch size
        block_mat: Material of the cube
        bkg_mat: Material of the background

    Returns:
        Tuple of three tensors: (background 1 angles, background 2 angles, signal angles)
    """

    # Cube material assignment
    def cubes_rad_length(*, z: float, lw: Tensor, size: float, uranium: bool = False) -> Tensor:
        rad_length = torch.ones(list((lw / size).long())) * X0[bkg_mat]
        grid = torch.zeros_like(rad_length)

        cube_size = 2
        offset = 2
        if uranium and (z > 0.3 and z < 0.6):
            grid[-offset - cube_size : -offset, offset : offset + cube_size] = X0[block_mat]
            rad_length[grid.bool()] = grid[grid.bool()]

        return rad_length

    # Passives list: [background 1, background 2, signal]
    passives = PassiveYielder(
        [
            lambda **kwargs: cubes_rad_length(**kwargs, uranium=False),  # background 1
            lambda **kwargs: cubes_rad_length(**kwargs, uranium=False),  # background 2 (null)
            lambda **kwargs: cubes_rad_length(**kwargs, uranium=True),  # signal
        ]
    )

    # Volume construction
    volume = Volume(get_layers(init_gap=init_gap, panel_z_spacing=panel_z_spacing, n_panels=n_panels, n_panels_seg=n_panels_seg, init_res=init_res))

    # volume wrapper
    wrapper = SegmentedPanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(torch.optim.SGD, lr=1e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e3),
        z_pos_opt=partial(torch.optim.SGD, lr=1e3),
        gap_opt=partial(torch.optim.SGD, lr=1e3),
        loss_func=VoxelX0Loss(target_budget=None, cost_coef=None),
    )

    # POCA / hit recorders
    poca_rec = PredPocaRecord()
    hit_rec = PredHitRecord()

    # Run prediction
    wrapper.predict(passives=passives, n_mu_per_volume=n_mu, mu_bs=mu_bs, cbs=[poca_rec, hit_rec])

    # Extract flattened POCA scattering angles
    data_bkg = poca_rec.poca_theta_mcs_batch[0].cpu().flatten()  # background 1
    data_null = poca_rec.poca_theta_mcs_batch[1].cpu().flatten()  # background 2 (null)
    data_sig = poca_rec.poca_theta_mcs_batch[2].cpu().flatten()  # signal

    return data_bkg, data_null, data_sig
