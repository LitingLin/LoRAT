from ._common import _BaseFilter
from typing import Sequence


class IndexSelector(_BaseFilter):
    def __init__(self, indices: Sequence[int]):
        assert indices is not None
        self.indices = indices
