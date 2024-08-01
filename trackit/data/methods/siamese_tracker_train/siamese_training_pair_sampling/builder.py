import numpy as np
from typing import Sequence
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler
from trackit.data.source import TrackingDataset
from trackit.miscellanies.pretty_format import pretty_format
from .worker import SiamFCTrainingPairSampler
from ._types import SiamesePairSamplingMethod, SiamesePairNegativeSamplingMethod


def build_SiamFC_training_pair_sampler(datasets: Sequence[TrackingDataset], datasets_sampling_weight: np.ndarray,
                                       sequence_picker: RandomAccessiblePerSequenceSampler,
                                       siamese_sampling_config: dict):
    print('Siamese training pair sampling:\n' + pretty_format(siamese_sampling_config))
    # positive training pair sampling
    positive_sample_config = siamese_sampling_config['positive_sample']
    siamese_sampling_method = SiamesePairSamplingMethod[positive_sample_config['sample_mode']]
    siamese_sampling_frame_range = positive_sample_config['max_gaps']
    if 'auto_extend' in positive_sample_config:
        siamese_sampling_frame_range_auto_extend_step = positive_sample_config['auto_extend']['step']
        siamese_sampling_frame_range_auto_extend_max_retry_count = positive_sample_config['auto_extend']['max_retry_count']
    else:
        siamese_sampling_frame_range_auto_extend_step = 0
        siamese_sampling_frame_range_auto_extend_max_retry_count = 0
    siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found = positive_sample_config.get('disable_constraint_if_not_found', False)
    positive_sample_weight = positive_sample_config.get('weight', 1.0)

    # negative training pair sampling
    negative_sample_weight = 0
    negative_sample_methods = []
    negative_sample_methods_weight = None
    if 'negative_sample' in siamese_sampling_config:
        negative_sample_weight = siamese_sampling_config['negative_sample']['weight']

        negative_sample_methods_weight = []

        for negative_sample_generation_rule in siamese_sampling_config['methods']:
            negative_sample_methods.append(SiamesePairNegativeSamplingMethod[negative_sample_generation_rule['type']])
            negative_sample_methods_weight.append(negative_sample_generation_rule['weight'])

        negative_sample_methods_weight = np.array(negative_sample_methods_weight, dtype=np.float64)
        negative_sample_methods_weight /= negative_sample_methods_weight.sum()

    negative_sample_weight = negative_sample_weight / (positive_sample_weight + negative_sample_weight)

    return SiamFCTrainingPairSampler(datasets, datasets_sampling_weight, sequence_picker,
                                     siamese_sampling_frame_range=siamese_sampling_frame_range,
                                     siamese_sampling_method=siamese_sampling_method,
                                     siamese_sampling_frame_range_auto_extend_step=siamese_sampling_frame_range_auto_extend_step,
                                     siamese_sampling_frame_range_auto_extend_max_retry_count=siamese_sampling_frame_range_auto_extend_max_retry_count,
                                     siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found=siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found,
                                     negative_sample_weight=negative_sample_weight,
                                     negative_sample_generation_methods=negative_sample_methods,
                                     negative_sample_generation_method_weights=negative_sample_methods_weight)
