import torch
import torch.nn as nn


def build_criterion(criterion_config: dict, device: torch.device, dtype: torch.dtype) -> nn.Module:
    if criterion_config['type'] == 'box_with_score_map':
        from .methods.box_with_score_map.builder import build_box_with_score_map_criteria
        return build_box_with_score_map_criteria(criterion_config, device)
    elif criterion_config['type'] == 'SPMTrack':
        from .methods.SPMTrack.builder import build_SPMTrack_criteria
        return build_SPMTrack_criteria(criterion_config)
    else:
        raise ValueError("unknown criterion type")
