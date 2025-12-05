import numpy as np
from typing import Protocol, Any

from .._types import TemporalTrackerTrainingSample


class TemporalTrackerTrain_DataTransform(Protocol):
    def __call__(self, training_sample: TemporalTrackerTrainingSample, rng_engine: np.random.Generator) -> Any:
        ...
