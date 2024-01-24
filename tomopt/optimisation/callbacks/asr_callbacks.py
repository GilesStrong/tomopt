import math
from functools import partial
from typing import Tuple

from numpy import quantile

from ...inference.asr import AngleStatisticReconstruction, VolumeInterest
from .callback import Callback


class ParamsASR(Callback):
    r"""
    Callback for AngleStatisticReconstruction parameter settings. 
    """

    def __init__(
        self, voi: VolumeInterest, score_method: partial = partial(quantile, q=0.5), dtheta_range: Tuple[float, float] = (0.0, math.pi / 4), use_p: bool = False
    ) -> None:

        # Scattering density parameters
        self.score_method = score_method

        # Cuts on scattering angles
        self.dtheta_range = dtheta_range

        # Voxelized volume
        self.voi = voi

        # Use momentum information
        self.use_p = use_p

    def on_x0_pred_begin(self) -> None:

        assert type(self.wrapper.fit_params.volume_inferrer) == AngleStatisticReconstruction, "wrapper inference method must be AngleStatisticReconstruction"

        self.wrapper.fit_params.volume_inferrer.set_params(score_method=self.score_method, dtheta_range=self.dtheta_range, use_p=self.use_p)

        self.wrapper.fit_params.volume_inferrer.set_voi(voi=self.voi)

    def on_x0_pred_end(self) -> None:

        assert type(self.wrapper.fit_params.volume_inferrer) == AngleStatisticReconstruction, "wrapper inference method must be AngleStatisticReconstruction"

        self.wrapper.fit_params.volume_inferrer.reset_params()
        self.wrapper.fit_params.volume_inferrer.reset_voi()
