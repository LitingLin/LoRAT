from typing import NamedTuple


class TrainData(NamedTuple):
    input: dict = {}
    target: dict = {}
    miscellanies: dict = {}
