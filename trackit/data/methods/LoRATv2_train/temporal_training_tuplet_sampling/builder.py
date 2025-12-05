import numpy as np
from typing import Sequence
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler
from trackit.data.source import TrackingDataset


def build_temporal_sampler(sampler_config: dict, config: dict,
                           datasets: Sequence[TrackingDataset], datasets_sampling_weight: np.ndarray,
                           sequence_picker: RandomAccessiblePerSequenceSampler):
    if sampler_config['type'] == 'adapt_from_siamese_sampler':
        from ...siamese_tracker_train.siamese_training_pair_sampling.builder import build_SiamFC_training_pair_sampler
        from .adapt_from_siamese_sampler.sampler import TemporalTrackerTrainingSamplesSampler_AdaptFromSiameseSampler
        from .adapt_from_siamese_sampler.adaptors.builder import build_temporal_sampler_adaptor
        siamfc_training_pair_sampler = build_SiamFC_training_pair_sampler(datasets, datasets_sampling_weight,
                                                                          sequence_picker,
                                                                          sampler_config['siamese_training_pair_sampling'])
        adaptor = build_temporal_sampler_adaptor(sampler_config['adaptor'], datasets)
        return TemporalTrackerTrainingSamplesSampler_AdaptFromSiameseSampler(siamfc_training_pair_sampler, adaptor)
    elif sampler_config['type'] == 'plain':
        from .plain import PlainTemporalTrainingSamplesSampler
        common_config = config['common']
        num_templates = len(common_config['template_sizes'])
        num_search_regions = len(common_config['search_region_sizes'])

        sampling_frame_range = sampler_config['max_gaps']
        sampling_frame_range_adjust_according_to_sequence_fps = sampler_config.get(
            'max_gaps_adjust_according_to_sequence_fps', False)
        if 'auto_extend' in sampler_config:
            sampling_frame_range_auto_extend_step = sampler_config['auto_extend']['step']
            sampling_frame_range_auto_extend_max_retry_count = sampler_config['auto_extend'][
                'max_retry_count']
        else:
            sampling_frame_range_auto_extend_step = 5
            sampling_frame_range_auto_extend_max_retry_count = 10
        sampling_disable_frame_range_constraint_if_search_frame_not_found = sampler_config.get(
            'disable_constraint_if_not_found', False)
        return PlainTemporalTrainingSamplesSampler(num_templates, num_search_regions, datasets, datasets_sampling_weight,
                                                   sequence_picker,
                                                   sampling_frame_range,
                                                   sampling_frame_range_adjust_according_to_sequence_fps,
                                                   sampling_frame_range_auto_extend_step,
                                                   sampling_frame_range_auto_extend_max_retry_count,
                                                   sampling_disable_frame_range_constraint_if_search_frame_not_found)
    else:
        raise ValueError(f"Unknown temporal sampler type: {sampler_config['type']}")
