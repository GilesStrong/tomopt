from typing import List, Optional

from .callback import Callback

__all__: List[str] = []


class EvalMetric(Callback):
    def __init__(self, lower_metric_better: bool, name: Optional[str] = None, main_metric: bool = True):
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
        Returns metric value

        Returns:
            metric value
        """

        return self.metric
