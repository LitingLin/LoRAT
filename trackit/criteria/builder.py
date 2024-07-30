import torch.nn as nn
from trackit.core.runtime.build_context import BuildContext


def build_criterion(criterion_config: dict, build_context: BuildContext, num_total_iterations: int) -> nn.Module:
    if criterion_config['type'] == 'box_with_score_map':
        from .methods.box_with_score_map.builder import build_box_with_score_map_criteria
        return build_box_with_score_map_criteria(criterion_config)
    else:
        raise ValueError("unknown criterion type")
