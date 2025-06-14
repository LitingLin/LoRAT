from typing import Sequence, Optional
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler

from ._types import SamplingResult_Element, SiameseTrainingPairSamplingResult, SiamesePairSamplingMethod, SiamesePairNegativeSamplingMethod
from ._algos import _do_siamfc_pair_sampling, _get_random_track, _get_random_frame_from_track


class SiamFCTrainingPairSampler:
    def __init__(self, datasets: Sequence[TrackingDataset],
                 dataset_weights: np.ndarray,
                 sequence_picker: Optional[RandomAccessiblePerSequenceSampler],
                 siamese_sampling_frame_range: int,
                 siamese_sampling_frame_range_adjust_according_to_sequence_fps: bool,
                 siamese_sampling_method: SiamesePairSamplingMethod,
                 siamese_sampling_frame_range_auto_extend_step: int,
                 siamese_sampling_frame_range_auto_extend_max_retry_count: int,
                 siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found: bool,
                 negative_sample_weight: float,
                 negative_sample_generation_methods: Sequence[SiamesePairNegativeSamplingMethod],
                 negative_sample_generation_method_weights: Optional[np.ndarray]):
        self.datasets = datasets

        self.dataset_weights = self._normalize_weights(dataset_weights)

        self.sequence_picker = sequence_picker
        self.siamese_sampling_frame_range = siamese_sampling_frame_range
        self.siamese_sampling_frame_range_adjust_according_to_sequence_fps = siamese_sampling_frame_range_adjust_according_to_sequence_fps
        self.siamese_sampling_method = siamese_sampling_method
        self.siamese_sampling_frame_range_auto_extend_step = siamese_sampling_frame_range_auto_extend_step
        self.siamese_sampling_frame_range_auto_extend_max_retry_count = siamese_sampling_frame_range_auto_extend_max_retry_count
        self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found = siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found
        self.negative_sample_weight = negative_sample_weight
        self.negative_sample_generation_methods = negative_sample_generation_methods
        self.negative_sample_generation_method_weights = negative_sample_generation_method_weights

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1."""
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        return weights

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

        sequence_fps = sequence.get_fps()
        reference_sequence_fps = 30

        track = _get_random_track(sequence, rng_engine)
        if is_positive:
            siamese_sampling_frame_range = self.siamese_sampling_frame_range
            if self.siamese_sampling_frame_range_adjust_according_to_sequence_fps and sequence_fps is not None:
                siamese_sampling_frame_range = int(siamese_sampling_frame_range * sequence_fps / reference_sequence_fps)
                if siamese_sampling_frame_range < 1:
                    siamese_sampling_frame_range = 1

            frame_indices = _do_siamfc_pair_sampling(
                len(track),
                siamese_sampling_frame_range,
                track.get_all_object_existence_flag(),
                None,
                self.siamese_sampling_method, rng_engine,
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
            method = rng_engine.choice(self.negative_sample_generation_methods,
                                       p=self.negative_sample_generation_method_weights)
            track = _get_random_track(sequence, rng_engine)
            frame = _get_random_frame_from_track(track, True, rng_engine)
            x_dataset_index = rng_engine.choice(np.arange(len(self.datasets)), p=self.dataset_weights)
            x_dataset = self.datasets[x_dataset_index]
            if method in (
            SiamesePairNegativeSamplingMethod.random, SiamesePairNegativeSamplingMethod.random_semantic_object):
                x_sequence_index = rng_engine.integers(0, len(x_dataset))
                x_sequence = x_dataset[x_sequence_index]
                x_track = _get_random_track(x_sequence, rng_engine)
                x_frame = _get_random_frame_from_track(x_track,
                                                       method == SiamesePairNegativeSamplingMethod.random_semantic_object,
                                                       rng_engine)
            else:
                raise NotImplementedError(method)

            training_pair = SiameseTrainingPairSamplingResult(
                SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame.get_frame_index()),
                SamplingResult_Element(x_dataset_index, x_sequence_index, x_track.get_object_id(),
                                       x_frame.get_frame_index()),
                False)
        return training_pair
