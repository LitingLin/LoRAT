from ._common import _BaseFilter
from typing import Optional


class RandomShuffle(_BaseFilter):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
