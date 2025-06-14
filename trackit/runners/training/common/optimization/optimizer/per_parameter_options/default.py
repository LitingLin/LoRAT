import torch.nn as nn
from typing import Dict, Optional
from ._common import get_common_per_parameter_optimizer_options, filter_out_params_by_rule_


def apply_default_per_parameter_optimizer_options_(rule: dict,
                                                   base_lr: float, base_weight_decay: Optional[float],
                                                   zero_1d_params_weight_decay: bool,
                                                   module_parameters: Dict[str, nn.Parameter],
                                                   optimizer_param_dict: list):
    named_params = filter_out_params_by_rule_(rule, module_parameters)
    assert len(named_params) > 0, f"rule must be effective\nrule:{rule}"
    if 'ignore' in rule and rule['ignore']:
        return

    this_params_optimizer_dict = {'params': tuple(named_params.values())}
    this_params_optimizer_dict.update(get_common_per_parameter_optimizer_options(rule, base_lr, base_weight_decay))
    if zero_1d_params_weight_decay:
        bias_norm_params = tuple(param for param in named_params.values() if param.ndim <= 1)
        if len(bias_norm_params) > 0:
            this_params_optimizer_dict['params'] = tuple(param for param in named_params.values() if param.ndim > 1)
            bias_norm_params_optimizer_dict = {key: value for key, value in this_params_optimizer_dict.items() if key not in ('params', 'weight_decay')}
            bias_norm_params_optimizer_dict['weight_decay'] = 0.
            bias_norm_params_optimizer_dict['params'] = bias_norm_params
            optimizer_param_dict.append(bias_norm_params_optimizer_dict)
    if len(this_params_optimizer_dict['params']) > 0:
        optimizer_param_dict.append(this_params_optimizer_dict)
