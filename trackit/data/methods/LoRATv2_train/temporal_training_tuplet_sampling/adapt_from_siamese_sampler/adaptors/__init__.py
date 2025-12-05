import numpy as np

from .....siamese_tracker_train.siamese_training_pair_sampling import SiameseTrainingPairSamplingResult
from ... import TemporalTrackerTrainingSamples_SamplingResult


class TemporalTrainingSampleAdaptor:
    def __call__(self, siamese_training_pair: SiameseTrainingPairSamplingResult, rng_engine: np.random.Generator) -> TemporalTrackerTrainingSamples_SamplingResult:
        raise NotImplementedError()
