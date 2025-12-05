import numpy as np

from . import TemporalTrainingSampleAdaptor, TemporalTrackerTrainingSamples_SamplingResult, SiameseTrainingPairSamplingResult


class PassThroughWrapper(TemporalTrainingSampleAdaptor):
    def __call__(self, siamese_training_pair: SiameseTrainingPairSamplingResult, rng_engine: np.random.Generator) -> TemporalTrackerTrainingSamples_SamplingResult:
        return TemporalTrackerTrainingSamples_SamplingResult((siamese_training_pair.z,),
                                                             (siamese_training_pair.x,),
                                                             (siamese_training_pair.is_positive,))
