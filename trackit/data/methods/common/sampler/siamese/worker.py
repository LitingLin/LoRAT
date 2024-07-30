from typing import Sequence, Optional
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler

from . import SamplingResult_Element, SiameseTrainingPairSamplingResult

from ._algos import get_random_positive_siamese_training_pair_from_track, _get_random_track, _get_random_frame_from_track
from ._types import SiamesePairSamplingMethod, SiamesePairNegativeSamplingMethod
from ._distractor import DistractorGenerator
from .aux_frame_sampling import AuxFrameSampling


class SiamFCTrainingPairSampler:
    def __init__(self, datasets: Sequence[TrackingDataset],
                 dataset_weights: np.ndarray,
                 sequence_picker: Optional[RandomAccessiblePerSequenceSampler],
                 siamese_sampling_frame_range: int,
                 siamese_sampling_method: SiamesePairSamplingMethod,
                 siamese_sampling_frame_range_auto_extend_step: int,
                 siamese_sampling_frame_range_auto_extend_max_retry_count: int,
                 siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found: bool,
                 negative_sample_weight: float,
                 negative_sample_generation_methods: Sequence[SiamesePairNegativeSamplingMethod],
                 negative_sample_generation_method_weights: Optional[np.ndarray],
                 aux_frame_sampler: Optional[AuxFrameSampling] = None):
        self.datasets = datasets

        dataset_weights = np.array(dataset_weights, dtype=np.float64)
        dataset_weights /= dataset_weights.sum()
        self.dataset_weights = dataset_weights

        self.sequence_picker = sequence_picker
        self.siamese_sampling_frame_range = siamese_sampling_frame_range
        self.siamese_sampling_method = siamese_sampling_method
        self.siamese_sampling_frame_range_auto_extend_step = siamese_sampling_frame_range_auto_extend_step
        self.siamese_sampling_frame_range_auto_extend_max_retry_count = siamese_sampling_frame_range_auto_extend_max_retry_count
        self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found = siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found
        self.negative_sample_weight = negative_sample_weight
        self.negative_sample_generation_methods = negative_sample_generation_methods
        self.negative_sample_generation_method_weights = negative_sample_generation_method_weights

        if len(negative_sample_generation_methods) > 0:
            distractor_picker_required = False
            for weight, method in zip(negative_sample_generation_method_weights, negative_sample_generation_methods):
                if weight > 0 and method == SiamesePairNegativeSamplingMethod.distractor:
                    distractor_picker_required = True
                    break
            if distractor_picker_required:
                self.distractor_pickers = tuple(DistractorGenerator(dataset) for dataset in self.datasets)
        self.aux_frame_sampler = aux_frame_sampler

    def __call__(self, index: Optional[int], rng_engine: np.random.Generator) -> SiameseTrainingPairSamplingResult:
        if index is not None:
            assert self.sequence_picker is not None, 'Sequence picker is required for indexed sampling'
            dataset_index, sequence_index = self.sequence_picker[index]
            dataset = self.datasets[dataset_index]
            sequence = dataset[sequence_index]
        else:
            dataset_index = rng_engine.choice(np.arange(len(self.datasets)), p=self.dataset_weights)
            dataset = self.datasets[dataset_index]
            sequence_index = rng_engine.integers(0, len(dataset))
            sequence = dataset[sequence_index]

        if self.negative_sample_weight > 0:
            is_positive = rng_engine.random() > self.negative_sample_weight
        else:
            is_positive = True

        track = _get_random_track(sequence, rng_engine)
        if is_positive:
            frame_indices = get_random_positive_siamese_training_pair_from_track(
                track, self.siamese_sampling_frame_range, self.siamese_sampling_method, rng_engine,
                self.siamese_sampling_frame_range_auto_extend_step,
                self.siamese_sampling_frame_range_auto_extend_max_retry_count,
                self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found
            )
            if len(frame_indices) == 1:
                frame_indices = (frame_indices[0], frame_indices[0])

            training_pair = SiameseTrainingPairSamplingResult(
                SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_indices[0]),
                SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_indices[1]),
                True)
        else:
            method = rng_engine.choice(self.negative_sample_generation_methods, p=self.negative_sample_generation_method_weights)
            track = _get_random_track(sequence, rng_engine)
            frame = _get_random_frame_from_track(track, True, rng_engine)
            x_dataset_index = rng_engine.choice(np.arange(len(self.datasets)), p=self.dataset_weights)
            x_dataset = self.datasets[x_dataset_index]
            if method in (SiamesePairNegativeSamplingMethod.random, SiamesePairNegativeSamplingMethod.random_semantic_object):
                x_sequence_index = rng_engine.integers(0, len(x_dataset))
                x_sequence = x_dataset[x_sequence_index]
                x_track = _get_random_track(x_sequence, rng_engine)
                x_frame = _get_random_frame_from_track(x_track, method == SiamesePairNegativeSamplingMethod.random_semantic_object, rng_engine)
            elif method == SiamesePairNegativeSamplingMethod.distractor:
                distractor_picker = self.distractor_pickers[x_dataset_index]
                x_sequence_index, x_track_index = distractor_picker(track.get_category_id(), rng_engine)
                x_track = x_dataset[x_sequence_index].get_track_by_index(x_track_index)
                x_frame = _get_random_frame_from_track(x_track, True, rng_engine)
            else:
                raise NotImplementedError(method)

            training_pair = SiameseTrainingPairSamplingResult(
                SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame.get_frame_index()),
                SamplingResult_Element(x_dataset_index, x_sequence_index, x_track.get_object_id(), x_frame.get_frame_index()),
                False)
        if self.aux_frame_sampler is not None:
            training_pair = self.aux_frame_sampler(training_pair, rng_engine)
        return training_pair
