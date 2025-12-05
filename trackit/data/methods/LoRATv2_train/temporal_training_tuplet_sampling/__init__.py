from typing import Sequence, Optional
from dataclasses import dataclass

import numpy as np

from ...siamese_tracker_train.siamese_training_pair_sampling import SamplingResult_Element

@dataclass(frozen=True)
class TemporalTrackerTrainingSamples_SamplingResult:
    templates: Sequence[SamplingResult_Element]
    search_regions: Sequence[SamplingResult_Element]
    search_region_target_exists: Sequence[bool]


class TemporalTrackerTrainingSamplesSampler:
    def __call__(self, index: Optional[int], rng_engine: np.random.Generator) -> TemporalTrackerTrainingSamples_SamplingResult:
        ...
