"""Skeletal script to create a gif of the heatmap during training through callbacks."""

import os
import imageio
from typing import List
from ...volume import PanelDetectorLayer
from .callback import Callback

__all__ = ["HeatMapGif"]


class HeatMapGif(Callback):
    """"""

    gif_filename = "heatmap.gif"
    buffer_files: List[str] = []

    def _plot_current(self) -> None:
        """"""

        filename = f"temp{len(self.buffer_files)}.png"
        self.buffer_files.append(filename)
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, PanelDetectorLayer) and l.type_label == "heatmap":
                for p in l.panels:
                    p.plot_map(bsavefig=True, filename=filename)
                    break
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l) , l.type_label}")
            break

    def _create_gif(self) -> None:
        """"""

        with imageio.get_writer(self.gif_filename, mode="I") as writer:
            for filename in self.buffer_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self.buffer_files):
            os.remove(filename)

    def on_epoch_begin(self) -> None:
        self._plot_current()

    def on_train_end(self) -> None:
        self._plot_current()
        self._create_gif()
