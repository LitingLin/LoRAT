from ._common import _BaseFilter
from typing import Optional


class RangeSelector(_BaseFilter):
    def __init__(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None):
        self.start = start
        self.stop = stop
        self.step = step
