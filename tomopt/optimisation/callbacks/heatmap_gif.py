import os
import imageio
from typing import List

from ...volume import PanelDetectorLayer
from .callback import Callback

r"""
Skeletal script to create a gif of the heatmap during training through callbacks.
"""

__all__ = ["HeatMapGif"]


class HeatMapGif(Callback):
    r"""
    Records a gif of the first heatmap in the first detector layer during training.
    """

    def __init__(self, gif_filename: str = "heatmap.gif") -> None:
        r"""
        Initialises the callback.

        Arguments:
            gif_filename: savename for the gif (will be appneded to the callback savepath)
        """

        self.gif_filename = gif_filename
        self._reset()

    def on_train_begin(self) -> None:
        r"""
        Prepares to record a new gif
        """

        super().on_train_begin()
        self._reset()

    def on_epoch_begin(self) -> None:
        r"""
        When a new training epoch begins, saves an image of the current layout of the first heatmap in the first detector layer
        """

        if self.wrapper.fit_params.state == "train":  # Avoid doubling the length  of the GIF
            self._plot_current()

    def on_train_end(self) -> None:
        r"""
        When training, saves an image of the current layout of the first heatmap in the first detector layer
        and then combines all images into a gif
        """

        self._plot_current()
        self._create_gif()

    def _plot_current(self) -> None:
        r"""
        Saves an image of the current layout of the first heatmap in the first detector layer
        """

        filename = self.wrapper.fit_params.cb_savepath / f"temp_heatmap_{len(self._buffer_files)}.png"
        self._buffer_files.append(filename)
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, PanelDetectorLayer) and l.type_label == "heatmap":
                for p in l.panels:
                    p.plot_map(bsavefig=True, filename=filename)
                    break
            else:
                raise NotImplementedError(f"HeatMapGif does not yet support {type(l) , l.type_label}")
            break

    def _reset(self) -> None:
        r"""
        Prepares to record a new gif
        """

        self._buffer_files: List[str] = []

    def _create_gif(self) -> None:
        r"""
        Combines recorded images into a gif
        """

        with imageio.get_writer(self.wrapper.fit_params.cb_savepath / self.gif_filename, mode="I") as writer:
            for filename in self._buffer_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self._buffer_files):
            os.remove(filename)
