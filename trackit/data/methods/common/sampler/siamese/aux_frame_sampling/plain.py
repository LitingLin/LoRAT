from typing import Sequence, Tuple, Optional

import numpy as np

from trackit.data.source import TrackingDataset
from . import AuxFrameSampling
from .. import SamplingResult_Element, SiameseTrainingPairSamplingResult
from .._types import SiamesePairSamplingMethod
from .._algos import _get_search_frame_candidates


class PlainAuxFrameSampling(AuxFrameSampling):
    def __init__(self,
                 datasets: Sequence[TrackingDataset],
                 aux_frame_max_gaps: Tuple[int, ...],
                 sampling_method: SiamesePairSamplingMethod):
        self.datasets = datasets
        self.aux_frame_max_gaps = aux_frame_max_gaps
        assert sampling_method in [SiamesePairSamplingMethod.causal, SiamesePairSamplingMethod.interval]
        self.sampling_method = sampling_method

    def __call__(self, siamese_training_pair: SiameseTrainingPairSamplingResult, rng_engine: np.random.Generator):
        if siamese_training_pair.is_positive:
            auxiliary_frame_indices = []
            x = siamese_training_pair.x
            dataset = self.datasets[x.dataset_index]
            sequence = dataset[x.sequence_index]
            track = sequence.get_track_by_id(x.track_id)
            all_bounding_box_existence_flag = track.get_all_object_existence_flag()
            sampling_method = self.sampling_method
            if self.sampling_method == SiamesePairSamplingMethod.causal:
                if x.frame_index < siamese_training_pair.z.frame_index:
                    sampling_method = SiamesePairSamplingMethod.reverse_causal
            for aux_frame_max_gap in self.aux_frame_max_gaps:
                aux_frame_index_candidates, _ = \
                    _get_search_frame_candidates(x.frame_index, len(track), aux_frame_max_gap, all_bounding_box_existence_flag, sampling_method)
                if aux_frame_index_candidates is None:
                    aux_frame_index = x.frame_index
                else:
                    aux_frame_index = rng_engine.choice(aux_frame_index_candidates)
                auxiliary_frame_indices.append(aux_frame_index)

            return SiameseTrainingPairSamplingResult(siamese_training_pair.z, siamese_training_pair.x,
                                                     siamese_training_pair.is_positive,
                                                     tuple(SamplingResult_Element(
                                                           x.dataset_index, x.sequence_index, x.track_id, frame_index)
                                                           for frame_index in auxiliary_frame_indices))
        else:
            raise NotImplementedError('Negative sampling is not implemented yet.')
