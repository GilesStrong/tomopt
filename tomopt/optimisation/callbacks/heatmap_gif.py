"""Skeletal script to create a gif of the heatmap during training through callbacks."""

import os
import imageio
from typing import List

from ...volume import PanelDetectorLayer
from .callback import Callback

__all__ = ["HeatMapGif"]


class HeatMapGif(Callback):
    """"""

    def __init__(self, gif_filename: str = "heatmap.gif") -> None:
        self.gif_filename = gif_filename
        self._reset()

    def _reset(self) -> None:
        self._buffer_files: List[str] = []

    def on_train_begin(self) -> None:
        super().on_train_begin()
        self._reset()

    def _plot_current(self) -> None:
        """"""

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

    def _create_gif(self) -> None:
        """"""

        with imageio.get_writer(self.wrapper.fit_params.cb_savepath / self.gif_filename, mode="I") as writer:
            for filename in self._buffer_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self._buffer_files):
            os.remove(filename)

    def on_epoch_begin(self) -> None:
        if self.wrapper.fit_params.state == "train":  # Avoid doubling the length  of the GIF
            self._plot_current()

    def on_train_end(self) -> None:
        self._plot_current()
        self._create_gif()
