import torch.nn as nn
from typing import Dict, Optional, Tuple
from ._common import filter_out_params_by_rule_


def _get_lr_decay_rate(num_layers: int, lr_decay_rate: float, layer_id: int):
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _update_optimizer_params_dict(named_params: Dict[str, nn.Parameter], lr: float, weight_decay: Optional[float],
                                  zero_1d_params_weight_decay: bool, optimizer_param_dict: list,
                                  decay_parameter_names: Tuple[str, ...]):
    if zero_1d_params_weight_decay:
        bias_norm_params = tuple(param for name, param in named_params.items() if name not in decay_parameter_names)
        if len(bias_norm_params) > 0:
            optimizer_param_dict.insert(0, {'params': bias_norm_params, 'lr': lr, 'weight_decay': 0.})
            named_params = {name: param for name, param in named_params.items() if name in decay_parameter_names}
    this_params_optimizer_dict = {'params': tuple(named_params.values()), 'lr': lr}
    if weight_decay is not None:
        this_params_optimizer_dict['weight_decay'] = weight_decay
    optimizer_param_dict.insert(0, this_params_optimizer_dict)


def apply_vit_lr_layer_decay_(rule: dict, base_lr: float, base_weight_decay: Optional[float],
                              module_parameters: Dict[str, nn.Parameter],
                              optimizer_param_dict: list,
                              decay_parameter_names: Tuple[str, ...]):
    blocks_param_path = rule['blocks']
    num_layers = rule.get('num_layers', None)
    if num_layers is None:
        num_layers = max([int(name[len(blocks_param_path) + 1:].split('.')[0]) for name in module_parameters.keys() if name.startswith(blocks_param_path)]) + 1
        print(f"vit_lr_layer_decay: num_layers is not specified, automatically set to {num_layers}")
    lr_decay_rate = rule['lr_decay_rate']
    zero_1d_params_weight_decay = rule['zero_1d_params_weight_decay']

    if 'first_layer' in rule:
        first_layer_param_paths = rule['first_layer']
        if 'first_layer_lr_mult' in rule:
            first_layer_lr = rule['first_layer_lr_mult'] * base_lr
        else:
            first_layer_lr = _get_lr_decay_rate(num_layers, lr_decay_rate, 0) * base_lr
        first_layer_params = {}
        for first_layer_param_path in first_layer_param_paths:
            first_layer_params.update(filter_out_params_by_rule_(first_layer_param_path, module_parameters))
        assert len(first_layer_params) > 0, f"rule must be effective\nrule:{rule}"
        _update_optimizer_params_dict(first_layer_params, first_layer_lr, base_weight_decay, zero_1d_params_weight_decay,
                                      optimizer_param_dict, decay_parameter_names)

    block_modules = {}

    for module_parameter_name in list(module_parameters.keys()):
        if module_parameter_name.startswith(blocks_param_path):
            block_id = int(module_parameter_name[len(blocks_param_path) + 1:].split('.')[0])
            if block_id not in block_modules:
                block_modules[block_id] = {}
            block_modules[block_id][module_parameter_name] = module_parameters.pop(module_parameter_name)
    assert len(block_modules) > 0, f"rule must be effective\nrule:{rule}"
    for block_id, block_named_params in block_modules.items():
        block_lr = _get_lr_decay_rate(num_layers, lr_decay_rate, block_id + 1) * base_lr
        _update_optimizer_params_dict(block_named_params, block_lr, base_weight_decay, zero_1d_params_weight_decay,
                                      optimizer_param_dict, decay_parameter_names)
