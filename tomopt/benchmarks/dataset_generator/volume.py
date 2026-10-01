from typing import List, Optional, Union

from torch import Tensor, nn

from tomopt.core import DEVICE
from tomopt.volume.layer import PanelDetectorLayer, PassiveLayer
from tomopt.volume.panel import (
    DetectorPanel,
    SegmentedSigmoidDetectorPanel,
    SigmoidDetectorPanel,
)

__all__ = ["make_panel", "get_layers"]


def make_panel(
    detector_type: str,
    *,
    smooth: float,
    res: float,
    eff: float,
    init_xyz: tuple,
    init_xy_span: tuple,
    n_panels_seg: Optional[int] = None,
    init_gap: Optional[float] = None,
    realistic_validation: bool,
) -> DetectorPanel:
    """Create a detector panel of the requested type."""

    if detector_type == "segmented":
        return SegmentedSigmoidDetectorPanel(
            n_panels=n_panels_seg,
            smooth=smooth,
            res=res,
            eff=eff,
            init_xyz=init_xyz,
            init_xy_span=init_xy_span,
            init_gap=init_gap,
            realistic_validation=realistic_validation,
            device=DEVICE,
        )

    elif detector_type == "panel":
        return SigmoidDetectorPanel(
            smooth=smooth,
            res=res,
            eff=eff,
            init_xyz=init_xyz,
            init_xy_span=init_xy_span,
            realistic_validation=realistic_validation,
            device=DEVICE,
        )

    else:
        raise ValueError(f"Unknown detector_type '{detector_type}'. " "Expected 'segmented' or 'panel'.")


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
    det_layer_size: int = 4,
    panel_z_spacing: float = 0.1,
    detector_type: str = "segmented",
) -> nn.ModuleList:
    """
    Constructs the detector and VOI geometry.

    The detector consists of an upper detector layer, a VOI consisting
    of several passive material layers, and a lower detector layer. Each
    detector layer contains ``n_panels`` detector panels. The type of
    detector panel is selected using ``detector_type``.

    Parameters
    ----------
    lwh : Tensor, default=Tensor([1, 1, 1])
        Length, width, and "height" of the passive volume.
        Currently "height" is not used, since the z positions
        of the passive layers are set in the radiation length
        function. length and width span from (0,0) to
        (lwh[0], lwh[1]) in the x and y directions.

    lw : Tensor, default=Tensor([2.0, 2.0])
        Initial x and y span of each detector panel.

    xy : Tensor, default=Tensor([0.5, 0.5])
        Initial x and y position of the detector panels.

    init_eff : float, default=0.9
        Initial detector efficiency.

    init_res : float, default=1e4
        Initial detector spatial resolution.

    n_panels_seg : int, default=2
        Number of segments along each dimension of a
        ``SegmentedSigmoidDetectorPanel``.

    n_panels : int, default=3
        Number of detector panels in each detector layer.

    smooth : float, default=0.1
        Smoothing parameter used by the sigmoid detector model.

    init_gap : float, default=0.1
        Initial gap size between adjacent segments of a segmented
        detector panel.

    realistic_validation : bool, default=True
        Whether to use the realistic detector response during validation,
        i.e. with 0 efficiency outside panel sensitive areas.

    det_layer_size : int, default=4
        Number of size units used to define the vertical extent of each
        detector layer. A unit size is 0.1 meters.

    panel_z_spacing : float, default=0.1
        Initial spacing in the z direction between consecutive detector
        panels within a detector layer.

    detector_type : {"segmented", "panel"}, default="segmented"
        Type of detector panel used in the detector layers.

        ``"segmented"``
            Use ``SegmentedSigmoidDetectorPanel``. Each panel is divided
            into multiple segments controlled by ``n_panels_seg`` and
            ``init_gap``.

        ``"panel"``
            Use ``SigmoidDetectorPanel`` without segmentation.

    Returns
    -------
    nn.ModuleList
        Module list containing the complete detector geometry in the
        following order:

        1. Upper detector layer at z = 1.2.
        2. Passive layer at z = 0.7.
        3. Passive layer at z = 0.6.
        4. Passive layer at z = 0.5.
        5. Passive layer at z = 0.4.
        6. Lower detector layer at z = 0.2.

        For the default ``n_panels=3`` and ``panel_z_spacing=0.1``,
        the panel positions are z = 1.2, 1.15, 1.10 in the upper
        detector and z = 0.2, 0.15, 0.10 in the lower detector.
    """

    size = 0.1

    def make_panels(z: float) -> nn.ModuleList:
        """Construct all panels belonging to one detector layer."""

        panels = []

        for i in range(n_panels):
            init_xyz = (
                xy[0].cpu().detach().item(),
                xy[1].cpu().detach().item(),
                z - (i * panel_z_spacing) / (n_panels - 1),
            )

            init_xy_span = (
                lw[0].cpu().detach().item(),
                lw[1].cpu().detach().item(),
            )

            panel = make_panel(
                detector_type=detector_type,
                n_panels_seg=n_panels_seg,
                smooth=smooth,
                res=init_res,
                eff=init_eff,
                init_xyz=init_xyz,
                init_xy_span=init_xy_span,
                init_gap=init_gap,
                realistic_validation=realistic_validation,
            )

            panels.append(panel)

        return nn.ModuleList(panels)

    layers: List[Union[PanelDetectorLayer, PassiveLayer]] = []

    # Top detector layer
    top_z = 1.2
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=lw,
            z=top_z,
            size=det_layer_size * size,
            panels=make_panels(top_z),
        )
    )

    # Passive volume
    for z in [0.7, 0.6, 0.5, 0.4]:
        layers.append(
            PassiveLayer(
                lw=lwh[:2],
                z=z,
                size=size,
                device=DEVICE,
            )
        )

    # Bottom detector layer
    bottom_z = 0.2

    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=lw,
            z=bottom_z,
            size=det_layer_size * size,
            panels=make_panels(bottom_z),
        )
    )

    return nn.ModuleList(layers)
