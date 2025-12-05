from typing import Optional

import numpy as np

from ....siamese_tracker_train.siamese_training_pair_sampling import SiamFCTrainingPairSampler
from .. import TemporalTrackerTrainingSamples_SamplingResult
from .adaptors import TemporalTrainingSampleAdaptor

class TemporalTrackerTrainingSamplesSampler_AdaptFromSiameseSampler:
    def __init__(self, parent_sampler: SiamFCTrainingPairSampler, temporal_sampler: TemporalTrainingSampleAdaptor):
        self.parent_sampler = parent_sampler
        self.sampler = temporal_sampler

    def __call__(self, index: Optional[int], rng_engine: np.random.Generator) -> TemporalTrackerTrainingSamples_SamplingResult:
        siamese_tracker_sampling_result = self.parent_sampler(index, rng_engine)
        return self.sampler(siamese_tracker_sampling_result, rng_engine)
