import torch.nn as nn
from typing import Dict, Tuple


def apply_zero_1d_param_weight_decay_rule_(rule: dict, module_parameters: Dict[str, nn.Parameter],
                                           optimizer_param_dict: list, decay_parameter_names: Tuple[str, ...]):
    one_dim_params = []
    for module_parameter_name in list(module_parameters.keys()):
        if module_parameter_name not in decay_parameter_names:
            one_dim_params.append(module_parameters.pop(module_parameter_name))

    if len(one_dim_params) > 0:
        optimizer_param_dict.append({'params': tuple(one_dim_params), 'weight_decay': 0.})
