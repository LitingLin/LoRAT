from ._common import _BaseFilter


class LengthLimit(_BaseFilter):
    def __init__(self, value: int):
        self.value = value
