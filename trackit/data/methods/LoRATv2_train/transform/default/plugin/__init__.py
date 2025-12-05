import numpy as np
from typing import Protocol, Mapping, Sequence
from trackit.data.protocol.train_input import TrainData
from ...._types import TemporalTrackerTrainingSample


class ExtraTransform(Protocol):
    def __call__(self, training_sample: TemporalTrackerTrainingSample, context: dict, data: dict, rng_engine: np.random.Generator) -> None:
        ...


class ExtraTransform_DataCollector(Protocol):
    def __call__(self, batch: Sequence[Mapping], collated: TrainData) -> None:
        ...
