from typing import List

from .callback import Callback

r"""
Provides callbacks designed to act in cycles over a number of epochs, e.g. to affect learning rates.
"""

__all__: List[str] = []


class CyclicCallback(Callback):
    pass
