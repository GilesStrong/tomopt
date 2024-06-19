import sys
from typing import Optional

__all__ = ["prog_bar"]


def prog_bar(percent: float, bar_length: Optional[int] = 90) -> None:  # bar_length should be less than 100
    sys.stdout.write("\r")
    sys.stdout.write("Completed: [{:{}}] {:>3}%".format("=" * int(percent / (100.0 / bar_length)), bar_length, int(percent)))
    sys.stdout.flush()
