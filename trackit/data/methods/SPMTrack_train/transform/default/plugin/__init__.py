import numpy as np
from typing import Mapping, Sequence, Protocol
from trackit.data.protocol.train_input import TrainData
from ...._types import SiameseTrainingPair


class ExtraTransform(Protocol):
    def __call__(self, training_pair: SiameseTrainingPair, context: dict, data: dict, rng_engine: np.random.Generator,
                 length_z: int, length_x: int) -> None:
        ...


class ExtraTransform_DataCollector(Protocol):
    def __call__(self, batch: Sequence[Mapping], collated: TrainData, max_z: int, max_x: int) -> None:
        ...
