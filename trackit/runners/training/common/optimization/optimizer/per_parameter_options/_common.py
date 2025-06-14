from typing import Optional, Dict
import re
import torch.nn as nn


def get_common_per_parameter_optimizer_options(rule: dict, base_lr: float, base_weight_decay: Optional[float]):
    optimizer_options = {}
    if 'lr_mult' in rule:
        optimizer_options['lr'] = base_lr * rule['lr_mult']
    if 'lr' in rule:
        optimizer_options['lr'] = rule['lr']
    if 'weight_decay_mult' in rule:
        assert base_weight_decay is not None, "weight_decay must be set if weight_decay_mult is used"
        optimizer_options['weight_decay'] = base_weight_decay * rule['weight_decay_mult']
    if 'weight_decay' in rule:
        optimizer_options['weight_decay'] = rule['weight_decay']
    return optimizer_options


class _Filter:
    def __init__(self, rule: dict):
        if 'name_prefix' in rule:
            self.name_filter = lambda name: name.startswith(rule['name_prefix'])
        elif 'name_regex' in rule:
            regex_matcher = re.compile(rule['name_regex'])
            self.name_filter = lambda name: regex_matcher.search(name) is not None
        elif 'name' in rule:
            parameter_name = rule['name']
            self.name_filter = lambda name: name == parameter_name
        else:
            self.name_filter = None

        self.ndim_filter = ()
        if 'ndim' in rule:
            if isinstance(rule['ndim'], int):
                self.ndim_filter = (rule['ndim'],)
            else:
                self.ndim_filter = rule['ndim']

    def __call__(self, name: str, param: nn.Parameter):
        if self.name_filter is not None:
            if not self.name_filter(name):
                return False

        if len(self.ndim_filter) > 0:
            if param.ndim not in self.ndim_filter:
                return False

        return True


def filter_out_params_by_rule_(rule: dict, named_module_parameters: Dict[str, nn.Parameter]) -> Dict[str, nn.Parameter]:
    named_params = {}
    parameter_fileter = _Filter(rule)
    for module_parameter_name in list(named_module_parameters.keys()):
        if parameter_fileter(module_parameter_name, named_module_parameters[module_parameter_name]):
            named_params[module_parameter_name] = named_module_parameters.pop(module_parameter_name)
    return named_params
