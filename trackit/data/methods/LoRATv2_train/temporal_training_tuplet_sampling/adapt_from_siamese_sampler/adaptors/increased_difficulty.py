from typing import Sequence

import numpy as np

from trackit.data.source import TrackingDataset
from . import TemporalTrainingSampleAdaptor, SiameseTrainingPairSamplingResult, TemporalTrackerTrainingSamples_SamplingResult
from ... import SamplingResult_Element
from .....siamese_tracker_train.siamese_training_pair_sampling._algos import _get_search_frame_candidates, SiamesePairSamplingMethod


class IncreasedDifficultyAuxFrameSampling(TemporalTrainingSampleAdaptor):
    def __init__(self, datasets: Sequence[TrackingDataset]):
        self.datasets = datasets

    def __call__(self, siamese_training_pair: SiameseTrainingPairSamplingResult, rng_engine: np.random.Generator):
        z = siamese_training_pair.z
        dataset = self.datasets[z.dataset_index]
        sequence = dataset[z.sequence_index]
        track = sequence.get_track_by_id(z.track_id)
        all_bounding_box_existence_flag = track.get_all_object_existence_flag()

        aux_frame_index_candidates, _ = \
            _get_search_frame_candidates(z.frame_index, len(track), None,
                                         all_bounding_box_existence_flag, SiamesePairSamplingMethod.interval)
        if aux_frame_index_candidates is None:
            aux_frame_index = z.frame_index
        else:
            aux_frame_index = rng_engine.choice(aux_frame_index_candidates)

        search_regions = [siamese_training_pair.z, siamese_training_pair.x]
        search_region_is_object_existence = [siamese_training_pair.is_positive] + [True] * (len(search_regions) - 1)

        return TemporalTrackerTrainingSamples_SamplingResult((SamplingResult_Element(z.dataset_index, z.sequence_index, z.track_id, aux_frame_index),),
                                                             tuple(search_regions),
                                                             search_region_is_object_existence)
