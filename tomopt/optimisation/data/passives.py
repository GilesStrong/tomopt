from typing import Callable, List, Optional, Union, Generator
from random import shuffle

from torch import Tensor

__all__ = ["PassiveGenerator", "PassiveYielder"]


class PassiveGenerator:
    def __init__(self) -> None:
        pass

    def generate(self) -> Callable[..., Tensor]:
        pass


class PassiveYielder:
    def __init__(self, passives: Union[List[Callable[..., Tensor]], PassiveGenerator], n_passives: Optional[int] = None, shuffle: bool = True):
        self.passives, self.n_passives, self.shuffle = passives, n_passives, shuffle
        if isinstance(self.passives, PassiveGenerator):
            if self.n_passives is None:
                raise ValueError("If a PassiveGenerator class is used, n_passives must be specified")
        else:
            self.n_passives = len(self.passives)

    def __len__(self) -> int:
        return self.n_passives

    def __iter__(self) -> Generator[Callable[..., Tensor], None, None]:
        if isinstance(self.passives, PassiveGenerator):
            for _ in range(self.n_passives):
                yield self.passives.generate()
        else:
            if self.shuffle:
                shuffle(self.passives)
            for p in self.passives:
                yield p
