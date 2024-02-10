from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from ...volume import AbsLayer, PanelDetectorLayer, PassiveLayer, SigmoidDetectorPanel

__all__ = ["get_initial_detector", "get_baseline_detector_1", "get_baseline_detector_2"]


def get_initial_detector(*, res: float = 1e4, eff: float = 0.9, span: float = 0.8, device: torch.device = torch.device("cpu")) -> nn.ModuleList:
    lwh: Tensor = Tensor([1.0, 1.0, 1.8])
    size: float = 0.1

    layers: List[AbsLayer] = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=lwh[:2],
            z=lwh[2].item(),
            size=0.4,
            panels=[
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 1.48),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=0.1,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 1.47),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 1.46),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 1.45),
                    init_xy_span=(span, span),
                    device=device,
                ),
            ],
        )
    )
    for z in np.round(np.arange(lwh[2] - 0.4, 0.4, -size), decimals=2):
        layers.append(PassiveLayer(lw=lwh[:2], z=z, size=size, device=device))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=lwh[:2],
            z=0.4,
            size=0.4,
            panels=[
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 0.35),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=1,
                    init_xyz=(0.0, 0.0, 0.34),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 0.33),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.0, 0.0, 0.32),
                    init_xy_span=(span, span),
                    device=device,
                ),
            ],
        )
    )

    return nn.ModuleList(layers)


def get_baseline_detector_1(*, res: float = 1e4, eff: float = 0.9, span: float = 0.8, device: torch.device = torch.device("cpu")) -> nn.ModuleList:
    lwh: Tensor = Tensor([1.0, 1.0, 1.8])
    size: float = 0.1

    layers: List[AbsLayer] = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=lwh[:2],
            z=lwh[2].item(),
            size=0.4,
            panels=[
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.8),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=0.1,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.75),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.5),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.45),
                    init_xy_span=(span, span),
                    device=device,
                ),
            ],
        )
    )
    for z in np.round(np.arange(lwh[2] - 0.4, 0.4, -size), decimals=2):
        layers.append(PassiveLayer(lw=lwh[:2], z=z, size=size, device=device))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=lwh[:2],
            z=0.4,
            size=0.4,
            panels=[
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 0.35),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=1,
                    init_xyz=(0.5, 0.5, 0.30),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 0.05),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 0.0),
                    init_xy_span=(span, span),
                    device=device,
                ),
            ],
        )
    )

    return nn.ModuleList(layers)


def get_baseline_detector_2(*, res: float = 1e4, eff: float = 0.9, span: float = 0.8, device: torch.device = torch.device("cpu")) -> nn.ModuleList:
    lwh: Tensor = Tensor([1.0, 1.0, 1.8])
    size: float = 0.1

    layers: List[AbsLayer] = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=lwh[:2],
            z=lwh[2].item(),
            size=0.4,
            panels=[
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.75),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=0.1,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.65),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.55),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 1.45),
                    init_xy_span=(span, span),
                    device=device,
                ),
            ],
        )
    )
    for z in np.round(np.arange(lwh[2] - 0.4, 0.4, -size), decimals=2):
        layers.append(PassiveLayer(lw=lwh[:2], z=z, size=size, device=device))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=lwh[:2],
            z=0.4,
            size=0.4,
            panels=[
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 0.35),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=1,
                    init_xyz=(0.5, 0.5, 0.25),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 0.15),
                    init_xy_span=(span, span),
                    device=device,
                ),
                SigmoidDetectorPanel(
                    smooth=1.0,
                    res=res,
                    eff=eff,
                    init_xyz=(0.5, 0.5, 0.05),
                    init_xy_span=(span, span),
                    device=device,
                ),
            ],
        )
    )

    return nn.ModuleList(layers)
