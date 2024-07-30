import torch.nn as nn
from typing import Dict, Optional
from ._common import get_common_per_parameter_optimizer_options, filter_out_params_by_rule_


def apply_default_per_parameter_optimizer_options_(rule: dict, base_lr: float, base_weight_decay: Optional[float], module_parameters: Dict[str, nn.Parameter], optimizer_param_dict: list):
    named_params = filter_out_params_by_rule_(rule, module_parameters)
    assert len(named_params) > 0, f"rule must be effective\nrule:{rule}"
    if 'ignore' in rule and rule['ignore']:
        return

    this_params_optimizer_dict = {'params': tuple(named_params.values())}
    this_params_optimizer_dict.update(get_common_per_parameter_optimizer_options(rule, base_lr, base_weight_decay))
    optimizer_param_dict.append(this_params_optimizer_dict)
