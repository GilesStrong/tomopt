from typing import Any, List, Tuple

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor

from tomopt.volume import PassiveLayer, Volume

from .layer import SegmentedPanelDetectorLayer
from .panel import SegmentedSigmoidDetectorPanel


def draw(volume: Volume, xlim: Tuple[float, float], ylim: Tuple[float, float], zlim: Tuple[float, float]) -> None:
    r"""
    Draws the layers/panels pertaining to the volume.
    When using this in a jupyter notebook, use "%matplotlib notebook" to have an interactive plot that you can rotate.

    Arguments:
        xlim: the x axis range for the three-dimensional plot.
        ylim: the y axis range for the three-dimensional plot.
        zlim: the z axis range for the three-dimensional plot.
    """
    ax = plt.figure(figsize=(9, 9)).add_subplot(projection="3d")
    ax.computed_zorder = False
    # TODO: find a way to fix transparency overlap in order to have passive layers in front of bottom active layers.
    passivearrays: List[Any] = []
    activearrays: List[Any] = []

    for layer in volume.layers:
        if isinstance(layer, PassiveLayer):
            lw, thez, size = layer.get_lw_z_size()
            roundedz = np.round(thez.item(), 2)
            # TODO: split these to allow for different alpha values (want: more transparent in front, more opaque in the back)
            rect = [
                [
                    (0, 0, roundedz - size),
                    (0 + lw[0].item(), 0, roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                    (0, 0 + lw[1].item(), roundedz - size),
                ],
                [(0, 0, roundedz - size), (0 + lw[0].item(), 0, roundedz - size), (0 + lw[0].item(), 0, roundedz), (0, 0, roundedz)],
                [
                    (0, 0 + lw[1].item(), roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                    (0, 0 + lw[1].item(), roundedz),
                ],
                [(0, 0, roundedz - size), (0, 0 + lw[1].item(), roundedz - size), (0, 0 + lw[1].item(), roundedz), (0, 0, roundedz)],
                [
                    (0 + lw[0].item(), 0, roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                    (0 + lw[0].item(), 0, roundedz),
                ],
            ]

            col = "red" if isinstance(layer, SegmentedSigmoidDetectorPanel) else ("blue" if isinstance(layer, PassiveLayer) else "black")

            passivearrays.append([rect, col, roundedz, 1])
            continue

        if isinstance(layer, SegmentedPanelDetectorLayer):
            for i, p in layer.yield_zordered_panels():
                col = "red" if isinstance(p, SegmentedSigmoidDetectorPanel) else ("blue" if isinstance(p, PassiveLayer) else "black")

                if not isinstance(p.xy, Tensor):
                    raise ValueError("Panel xy is not a tensor, for some reason")
                if not isinstance(p.z, Tensor):
                    raise ValueError("Panel z is not a tensor, for some reason")

                n_panels = p.n_panels
                gap_size = p.gap_size.item()
                panel_size = p.get_scaled_xy_span().cpu().detach().numpy() / n_panels
                center_x = p.xy[0].cpu().detach().numpy()
                center_y = p.xy[1].cpu().detach().numpy()
                indices = np.arange(n_panels) - (n_panels - 1) / 2
                centers_x = center_x + indices * (panel_size[0] + gap_size)
                centers_y = center_y + indices * (panel_size[0] + gap_size)

                panel_x_maxs = []
                panel_y_mins = []
                panel_x_mins = []
                panel_y_maxs = []

                for cx in centers_x:
                    start = cx - panel_size[0] / 2
                    end = cx + panel_size[0] / 2
                    panel_x_mins.append(start)
                    panel_x_maxs.append(end)

                for cy in centers_y:
                    start = cy - panel_size[0] / 2
                    end = cy + panel_size[0] / 2
                    panel_y_mins.append(start)
                    panel_y_maxs.append(end)

                for i in range(n_panels):
                    for j in range(n_panels):
                        x_min = panel_x_mins[i]
                        x_max = panel_x_maxs[i]
                        y_min = panel_y_mins[j]
                        y_max = panel_y_maxs[j]

                        rect = [
                            [
                                [x_min, y_min, p.z.data[0].item()],
                                [x_max, y_min, p.z.data[0].item()],
                                [x_max, y_max, p.z.data[0].item()],
                                [x_min, y_max, p.z.data[0].item()],
                                [x_min, y_min, p.z.data[0].item()],
                            ]
                        ]
                        activearrays.append([rect, col, p.z.data[0].item(), 0.2])
        else:
            raise TypeError("Volume.draw does not yet support layers of type", type(layer))

    allarrays = activearrays + passivearrays

    for voxelandcolour in allarrays:
        ax.add_collection3d(
            Poly3DCollection(
                voxelandcolour[0],
                facecolors=voxelandcolour[1],
                linewidths=1,
                edgecolors=voxelandcolour[1],
                alpha=voxelandcolour[3],
                zorder=voxelandcolour[2],
                sort_zpos=voxelandcolour[2],
            )
        )
    plt.ylim(xlim)
    plt.xlim(ylim)
    ax.set_zlim(zlim)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    ax.set_zlabel("z [m]")

    plt.title("Volume layers")

    red_patch = mpatches.Patch(color="red", label="Active Detector Layers")
    pink_patch = mpatches.Patch(color="blue", label="Passive Layers")

    ax.legend(handles=[red_patch, pink_patch])

    plt.show()
