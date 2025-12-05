import torch

from trackit.miscellanies.printing import pretty_format
from . import TrackerOutputPostProcess


def build_post_process(post_process_config: dict, common_config: dict, device: torch.device) -> TrackerOutputPostProcess:
    print('Tracker output post processing:\n' + pretty_format(post_process_config, indent_level=1))
    post_process_type = post_process_config['type']
    if post_process_type == 'box_with_score_map':
        from .box_with_score_map import PostProcessing_BoxWithScoreMap
        response_map_size = common_config['response_map_size']
        search_region_size = common_config['search_region_size']
        window_penalty_ratio = post_process_config['window_penalty']
        return PostProcessing_BoxWithScoreMap(device, response_map_size, search_region_size, window_penalty_ratio)
    elif post_process_type == 'LoRATv2':
        from .box_with_score_map import PostProcessing_BoxWithScoreMap
        search_region_size = common_config['search_regions'][-1]['size']
        response_map_size = search_region_size[0] // common_config['model_stride'], search_region_size[1] // common_config['model_stride']
        window_penalty_ratio = post_process_config['window_penalty']
        return PostProcessing_BoxWithScoreMap(device, response_map_size, search_region_size, window_penalty_ratio)
    else:
        raise NotImplementedError("Unknown post process type: {}".format(post_process_type))
