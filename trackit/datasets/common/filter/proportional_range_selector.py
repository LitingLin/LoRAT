from ._common import _BaseFilter
from typing import Optional


class ProportionalRangeSelector(_BaseFilter):
    def __init__(self, start: Optional[float] = None, stop: Optional[float] = None):
        self.start = start
        self.stop = stop
