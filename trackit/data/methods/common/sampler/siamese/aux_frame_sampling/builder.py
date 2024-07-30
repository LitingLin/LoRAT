from typing import Sequence
from trackit.data.source import TrackingDataset


def build_aux_frame_sampler(aux_frame_sampling_config: dict, datasets: Sequence[TrackingDataset]):
    if aux_frame_sampling_config['type'] == 'plain':
        from .plain import PlainAuxFrameSampling
        from .._types import SiamesePairSamplingMethod
        aux_frame_sampler = PlainAuxFrameSampling(datasets, aux_frame_sampling_config['max_gaps'],
                                                  SiamesePairSamplingMethod[aux_frame_sampling_config['type']])
    elif aux_frame_sampling_config['type'] == 'tubelet_search_region':
        from .tubelet import TubeletAuxFrameSampling
        aux_frame_sampler = TubeletAuxFrameSampling(datasets, aux_frame_sampling_config['size'],
                                                    aux_frame_sampling_config.get('interval', 0),
                                                    aux_frame_sampling_config['causal'])
    else:
        raise NotImplementedError(aux_frame_sampling_config['type'])

    return aux_frame_sampler
