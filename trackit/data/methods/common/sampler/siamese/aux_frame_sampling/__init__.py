import numpy as np
from .. import SiameseTrainingPairSamplingResult


class AuxFrameSampling:
    def __call__(self, siamese_training_pair: SiameseTrainingPairSamplingResult, rng_engine: np.random.Generator):
        raise NotImplementedError()
