from typing import NamedTuple


class TrainData(NamedTuple):
    batch_size: int
    input: dict = {}
    target: dict = {}
    miscellanies: dict = {}
