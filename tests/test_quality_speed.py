import os
from functools import partial
from pathlib import Path
from timeit import default_timer
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from tomopt.core import X0
from tomopt.optimisation import MuonResampler, PanelVolumeWrapper, PassiveYielder
from tomopt.plotting import plot_pred_true_x0
from tomopt.volume import DetectorPanel, PanelDetectorLayer, PassiveLayer, Volume

DEVICE = torch.device("cpu")
PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.flaky(max_runs=2, min_passes=1)
@patch("matplotlib.pyplot.show")
def test_lead_beryllium(mock_show):
    def get_layers():
        layers = []
        lwh = Tensor([1, 1, 1])
        size = 0.1
        init_eff = 0.9
        init_res = 1e4
        n_panels = 4
        init_xy_span = [2.0, 2.0]
        layers.append(
            PanelDetectorLayer(
                pos="above",
                lw=lwh[:2],
                z=1,
                size=2 * size,
                panels=[
                    DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 1 - (i * (2 * size) / n_panels)], init_xy_span=init_xy_span, device=DEVICE)
                    for i in range(n_panels)
                ],
            )
        )
        for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            layers.append(PassiveLayer(lw=lwh[:2], z=z, size=size, device=DEVICE))
        layers.append(
            PanelDetectorLayer(
                pos="below",
                lw=lwh[:2],
                z=0.2,
                size=2 * size,
                panels=[
                    DetectorPanel(res=init_res, eff=init_eff, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * size) / n_panels)], init_xy_span=init_xy_span, device=DEVICE)
                    for i in range(n_panels)
                ],
            )
        )

        return nn.ModuleList(layers)

    volume = Volume(get_layers())
    wrapper = PanelVolumeWrapper(
        volume, xy_pos_opt=partial(torch.optim.SGD, lr=5e4), z_pos_opt=partial(torch.optim.SGD, lr=5e3), xy_span_opt=partial(torch.optim.SGD, lr=1e4)
    )

    def arb_rad_length(*, z: float, lw: Tensor, size: float) -> Tensor:
        rad_length = torch.ones(list((lw / size).long())) * X0["beryllium"]
        if z >= 0.4 and z <= 0.5:
            rad_length[5:, 5:] = X0["lead"]
        return rad_length

    tmr = default_timer()
    preds = wrapper.predict(PassiveYielder([arb_rad_length]), n_mu_per_volume=10000, mu_bs=250, cbs=[MuonResampler()])[0]
    time = default_timer() - tmr
    assert time <= 60  # 2018 MacBook Pro: ~20s, but GitHub CI is slower

    plot_pred_true_x0(*preds, savename=PKG_DIR / "lead_beryllium.png")

    pb = preds[0][1:3, 5:, 5:]
    be = np.concatenate(
        [preds[0][1:3, :5, :5].flatten(), preds[0][1:3, 5:, :5].flatten(), preds[0][1:3, :5, 5:].flatten(), preds[0][(1, 3, 4, 5), :, :].flatten()]
    )
    met = (be.mean() - pb.mean()) / (np.sqrt((pb.std() ** 2) + (be.std() * 2)))
    assert 0.3 <= met <= 0.4
