from typing import NamedTuple
import torch


class TrainData(NamedTuple):
    batch_size: int
    input: dict | list | torch.Tensor = {}
    target: dict | list | torch.Tensor = {}
    miscellanies: dict = {}
