from typing import List, Optional

from .callback import Callback

r"""
Provides "callbacks" designed to compute metrics about the performance of the detector/inference, e.g. accuracy
"""

__all__: List[str] = []


class EvalMetric(Callback):
    r"""
    Base class from which metric should inherit and implement the compuation of their metric values.
    Inheriting classes will automatically be detected by :class:`~tomopt.optimisation.callbacks.monitors.MetricLogger`
    and included in live feedback if it is the "main metric"
    """

    def __init__(self, lower_metric_better: bool, name: Optional[str] = None, main_metric: bool = True):
        r"""
        Initialises the metric.
        Arguments:
            lower_metric_better: if True, a lower value of the metric should be considered better than a higher value
            name: name to associate with the metric
            main_metric: whether this metric should be considered the "main metric"
        """

        self.lower_metric_better, self.main_metric = lower_metric_better, main_metric
        self.name = type(self).__name__ if name is None else name

    def on_train_begin(self) -> None:
        r"""
        Ensures that only one main metric is used
        """

        super().on_train_begin()
        self.metric: Optional[float] = None
        if self.main_metric:
            for c in self.wrapper.fit_params.metric_cbs:
                c.main_metric = False
            self.main_metric = True

    def get_metric(self) -> float:
        r"""
        This will be called by :meth:`~tomopt.optimisation.callbacks.monitors.MetricLogger.on_epoch_end`

        Returns:
            metric value
        """

        return self.metric
