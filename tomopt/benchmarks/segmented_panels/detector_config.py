from typing import List, Union

import torch.nn as nn
from torch import Tensor

# segmented panel imports
from tomopt.benchmarks.segmented_panels.layer import SegmentedPanelDetectorLayer

# tomopt imports
from tomopt.core import DEVICE
from tomopt.volume.layer import PassiveLayer

__all__ = ["get_layers"]


def get_layers(
    lwh: Tensor = Tensor([1, 1, 1]),
    lw: Tensor = Tensor([2.0, 2.0]),
    xy: Tensor = Tensor([0.5, 0.5]),
    init_eff: float = 0.9,
    init_res: float = 1e4,
    n_panels_seg: int = 2,
    n_panels: int = 3,
    smooth: float = 0.1,
    init_gap: float = 0.1,
    realistic_validation: bool = True,
    realistic_training: bool = False,
    det_layer_size: int = 4,
    panel_z_spacing: float = 0.1,
) -> nn.ModuleList:
    """
    Create the layers of the detector.
    The layers are created in the following order:
    1. SegmentedPanelDetectorLayer (above)
    2. PassiveLayer
    3. SegmentedPanelDetectorLayer (below)

    Note: It suffices to construct the layers withe the required parameters.
    The inner panels are created by the layer class constructor.

    TODO: fix zhigh and zlow to be passed as arguments.

    Arguments:
        lwh: The size of the passive layers in the x, y, z directions.
        lw: The size of the detector layers in the x, y directions.
        xy: The position of the layers in the x, y directions.
        init_eff: The maximum efficiency of the detector.
        init_res: The maximum resolution of the detector.
        n_panels_seg: The number of panels in each direction.
        n_panels: The number of panels along z.
        smooth: The smoothness of the surrogate.
        init_gap: The initial gap between the segmented panels.
        realistic_validation: Whether to use realistic validation or not.
        realistic_training: Whether to use realistic training or not.
        det_layer_size: Thickness of the detector layer
            (coefficient of the size argument).
    Returns:
        nn.ModuleList: The layers of the detector+passive volume.

    """
    layers: List[Union[SegmentedPanelDetectorLayer, PassiveLayer]] = []
    lwh = lwh
    size = 0.1
    lw = lw
    init_eff = init_eff
    init_res = init_res
    n_panels_seg = n_panels_seg
    n_panels = n_panels
    smooth = smooth
    init_gap = init_gap
    realistic_validation = realistic_validation
    realistic_training = realistic_training
    det_layer_size = det_layer_size
    xy = xy
    panel_z_spacing = panel_z_spacing

    layers.append(
        SegmentedPanelDetectorLayer(
            pos="above",
            lw=lw,
            z=1.2,
            zlow=0.71,
            zhigh=3,
            size=det_layer_size * size,
            gap_size=init_gap,
            res=init_res,
            eff=init_eff,
            n_panels=n_panels,
            n_panels_seg=n_panels_seg,
            smooth=smooth,
            realistic_validation=realistic_validation,
            realistic_training=realistic_training,
            xy=xy,
            panel_z_spacing=panel_z_spacing,
        )
    )

    # set passive volume
    for z in [0.7, 0.6, 0.5, 0.4]:
        layers.append(
            PassiveLayer(
                lw=lwh[:2],
                z=z,
                size=size,
                device=DEVICE,
            )
        )

    layers.append(
        SegmentedPanelDetectorLayer(
            pos="below",
            lw=lw,
            z=0.2,
            zlow=-0.1,
            zhigh=0.25,
            size=det_layer_size * size,
            gap_size=init_gap,
            res=init_res,
            eff=init_eff,
            n_panels=n_panels,
            n_panels_seg=n_panels_seg,
            smooth=smooth,
            realistic_validation=realistic_validation,
            realistic_training=realistic_training,
            xy=xy,
            panel_z_spacing=panel_z_spacing,
        )
    )

    return nn.ModuleList(layers)


def get_layers_multiple(
    lwh: Tensor = Tensor([1, 1, 1]),
    lw: Tensor = Tensor([2.0, 2.0]),
    xy: Tensor = Tensor([0.5, 0.5]),
    init_eff: float = 0.9,
    init_res: float = 1e4,
    n_panels_seg: int = 2,
    n_panels: int = 3,
    smooth: float = 0.1,
    init_gap: float = 0.1,
    realistic_validation: bool = True,
    realistic_training: bool = False,
    det_layer_size: int = 4,
    panel_z_spacing: float = 0.1,
) -> nn.ModuleList:
    """
    Create the layers of the detector.
    The layers are created in the following order:
    1. SegmentedPanelDetectorLayer (above)
    2. PassiveLayer
    3. SegmentedPanelDetectorLayer (below)

    Note: It suffices to construct the layers withe the required parameters.
    The inner panels are created by the layer class constructor.

    TODO: fix zhigh and zlow to be passed as arguments.

    Arguments:
        lwh: The size of the passive layers in the x, y, z directions.
        lw: The size of the detector layers in the x, y directions.
        xy: The position of the layers in the x, y directions.
        init_eff: The maximum efficiency of the detector.
        init_res: The maximum resolution of the detector.
        n_panels_seg: The number of panels in each direction.
        n_panels: The number of panels along z.
        smooth: The smoothness of the surrogate.
        init_gap: The initial gap between the segmented panels.
        realistic_validation: Whether to use realistic validation or not.
        realistic_training: Whether to use realistic training or not.
        det_layer_size: Thickness of the detector layer
            (coefficient of the size argument).
        panel_z_spacing: The spacing between the panels along z.
    Returns:
        nn.ModuleList: The layers of the detector+passive volume.

    """
    layers: List[Union[SegmentedPanelDetectorLayer, PassiveLayer]] = []
    lwh = lwh
    size = 0.1
    lw = lw
    init_eff = init_eff
    init_res = init_res
    n_panels_seg = n_panels_seg
    n_panels = n_panels
    smooth = smooth
    init_gap = init_gap
    realistic_validation = realistic_validation
    realistic_training = realistic_training
    det_layer_size = det_layer_size
    xy = xy
    panel_z_spacing = panel_z_spacing

    layers.append(
        SegmentedPanelDetectorLayer(
            pos="above",
            lw=lw,
            z=1.2,
            zlow=0.71,
            zhigh=3,
            size=det_layer_size * size,
            gap_size=init_gap,
            res=init_res,
            eff=init_eff,
            n_panels=n_panels,
            n_panels_seg=n_panels_seg,
            smooth=smooth,
            realistic_validation=realistic_validation,
            realistic_training=realistic_training,
            xy=xy,
            panel_z_spacing=panel_z_spacing,
        )
    )

    # set passive volume
    for z in [0.7, 0.6, 0.5, 0.4]:
        layers.append(
            PassiveLayer(
                lw=lwh[:2],
                z=z,
                size=size,
                device=DEVICE,
            )
        )

    layers.append(
        SegmentedPanelDetectorLayer(
            pos="below",
            lw=lw,
            z=0.2,
            zlow=-0.1,
            zhigh=0.25,
            size=det_layer_size * size,
            gap_size=init_gap,
            res=init_res,
            eff=init_eff,
            n_panels=n_panels,
            n_panels_seg=n_panels_seg,
            smooth=smooth,
            realistic_validation=realistic_validation,
            realistic_training=realistic_training,
            xy=xy,
            panel_z_spacing=panel_z_spacing,
        )
    )

    return nn.ModuleList(layers)
