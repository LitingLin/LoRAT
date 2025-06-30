from typing import Sequence, Optional
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler

from ._algos import get_random_positive_siamese_training_pair_from_track, _get_random_track, _get_random_frame_from_track
from ._types import SiamesePairSamplingMethod, SiamesePairNegativeSamplingMethod, SiameseTrainingPairMultiSamplingResult, SamplingResult_Element, SiameseTrainingPairSamplingResult
from ._distractor import DistractorGenerator


class SPMTrack_TrainingTupletSampler:
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
                 num_template_frames: int,
                 num_search_frames: int,
                 max_sample_interval: int):
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
        self.max_sample_interval = max_sample_interval
        self.num_template_frames = num_template_frames
        self.num_search_frames = num_search_frames

        if len(negative_sample_generation_methods) > 0:
            distractor_picker_required = False
            for weight, method in zip(negative_sample_generation_method_weights, negative_sample_generation_methods):
                if weight > 0 and method == SiamesePairNegativeSamplingMethod.distractor:
                    distractor_picker_required = True
                    break
            if distractor_picker_required:
                self.distractor_pickers = tuple(DistractorGenerator(dataset) for dataset in self.datasets)

    def __call__(self, index: Optional[int], rng_engine: np.random.Generator) -> SiameseTrainingPairMultiSamplingResult:
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
        #import pdb; pdb.set_trace()
        track = _get_random_track(sequence, rng_engine) # 以SOTTrack为单位获取帧
        if is_positive:
            template_indices, search_indices = get_random_positive_siamese_training_pair_from_track(
                track, self.siamese_sampling_frame_range, self.siamese_sampling_method, rng_engine,
                self.siamese_sampling_frame_range_auto_extend_step,
                self.siamese_sampling_frame_range_auto_extend_max_retry_count,
                self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found,
                self.num_template_frames, self.num_search_frames, self.max_sample_interval
            )

            training_pair = SiameseTrainingPairMultiSamplingResult(
                [SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_indices) for frame_indices in template_indices],
                [SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_indices) for frame_indices in search_indices],
                True)
        else:
            raise NotImplementedError("SiamFCTrainingPairSampler negative sampling is not implemented!")
        return training_pair