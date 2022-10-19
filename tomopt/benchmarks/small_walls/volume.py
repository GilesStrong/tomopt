from typing import List
import numpy as np
from functools import partial

import torch
from torch import Tensor, nn

from ...volume import Volume, PanelDetectorLayer, PassiveLayer, DetectorPanel
from ...volume.layer import AbsLayer
from ...optimisation.wrapper import PanelVolumeWrapper
from ...muon import MuonGenerator2016

__all__ = ["get_small_walls_volume", "get_small_walls_volume_wrapper"]


def get_small_walls_volume(
    size: float = 1,
    passive_lwh: Tensor = Tensor([10.0, 10.0, 10.0]),
    span: float = 4.0,
    res: float = 1e4,
    eff: float = 1.0,
    det_height: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Volume:
    layers: List[AbsLayer] = []
    n_panels = 4
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=passive_lwh[:2],
            z=passive_lwh[2].item() + (2 * det_height),
            size=det_height,
            panels=[
                DetectorPanel(
                    res=res,
                    eff=eff,
                    init_xyz=(passive_lwh[0].item() / 2, passive_lwh[1].item() / 2, passive_lwh[2].item() + (2 * det_height) - (i * (det_height) / n_panels)),
                    init_xy_span=(span * passive_lwh[0].item(), span * passive_lwh[1].item()),
                    device=device,
                )
                for i in range(n_panels)
            ],
        )
    )
    for z in np.round(np.arange(passive_lwh[2], 0.0, -size), decimals=2):
        layers.append(PassiveLayer(lw=passive_lwh[:2], z=z + det_height, size=size, device=device))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=passive_lwh[:2],
            z=det_height,
            size=det_height,
            panels=[
                DetectorPanel(
                    res=res,
                    eff=eff,
                    init_xyz=(passive_lwh[0].item() / 2, passive_lwh[1].item() / 2, det_height - (i * (det_height) / n_panels)),
                    init_xy_span=(span * passive_lwh[0].item(), span * passive_lwh[1].item()),
                    device=device,
                )
                for i in range(n_panels)
            ],
        )
    )

    return Volume(nn.ModuleList(layers))


def get_small_walls_volume_wrapper(
    size: float = 1,
    passive_lwh: Tensor = Tensor([10.0, 10.0, 10.0]),
    span: float = 4.0,
    res: float = 1e4,
    eff: float = 1.0,
    det_height: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> PanelVolumeWrapper:
    volume = get_small_walls_volume(size=size, passive_lwh=passive_lwh, span=span, res=res, eff=eff, det_height=det_height, device=device)
    return PanelVolumeWrapper(
        volume,
        mu_generator=MuonGenerator2016.from_volume(volume, fixed_mom=1),
        xy_pos_opt=partial(torch.optim.SGD, lr=5e4),
        z_pos_opt=partial(torch.optim.SGD, lr=5e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e4),
        loss_func=None,
    )
