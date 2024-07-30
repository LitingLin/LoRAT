import torch

from trackit.miscellanies.pretty_format import pretty_format
from . import TrackerOutputPostProcess


def build_post_process(post_process_config: dict, common_config: dict, device: torch.device) -> TrackerOutputPostProcess:
    print('Tracker output post processing:\n' + pretty_format(post_process_config))
    post_process_type = post_process_config['type']
    if post_process_type == 'box_with_score_map':
        from .box_with_score_map import PostProcessing_BoxWithScoreMap
        response_map_size = common_config['response_map_size']
        search_region_size = common_config['search_region_size']
        window_penalty_ratio = post_process_config['window_penalty']
        return PostProcessing_BoxWithScoreMap(device, response_map_size, search_region_size, window_penalty_ratio, post_process_config.get('classification_score_do_penalty', False))
    else:
        raise NotImplementedError("Unknown post process type: {}".format(post_process_type))
