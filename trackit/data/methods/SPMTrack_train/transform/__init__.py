import numpy as np
from typing import Any, Protocol

from .._types import SiameseTrainingMultiPair


class SPMTrackTrain_DataTransform(Protocol):
    def __call__(self, training_pair: SiameseTrainingMultiPair, rng_engine: np.random.Generator) -> Any:
        ...
