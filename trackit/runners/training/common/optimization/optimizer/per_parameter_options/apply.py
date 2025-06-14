import torch.nn as nn
from typing import Optional, Dict, List, Any
from .default import apply_default_per_parameter_optimizer_options_
from .vit_lr_layer_decay import apply_vit_lr_layer_decay_
from ._decay_param_names import get_decay_parameter_names
from .zero_1d_param_weight_decay import apply_zero_1d_param_weight_decay_rule_


def _apply_per_parameter_adjustment_rules_(base_lr: float, base_weight_decay: Optional[float], per_parameter_rules: dict,
                                           remaining_module_parameters: Dict[str, nn.Parameter], optimizer_param_dict: list,
                                           model: nn.Module):
    decay_parameter_names = get_decay_parameter_names(model)
    for per_parameter_rule in per_parameter_rules:
        if 'type' not in per_parameter_rule or ('type' in per_parameter_rule and per_parameter_rule['type'] == 'default'):
            apply_default_per_parameter_optimizer_options_(per_parameter_rule, base_lr, base_weight_decay, per_parameter_rule.get('zero_1d_params_weight_decay', False), remaining_module_parameters, optimizer_param_dict)
        elif per_parameter_rule['type'] == 'vit_lr_layer_decay':
            apply_vit_lr_layer_decay_(per_parameter_rule, base_lr, base_weight_decay, remaining_module_parameters, optimizer_param_dict, decay_parameter_names)
        elif per_parameter_rule['type'] == 'zero_1d_param_weight_decay':
            apply_zero_1d_param_weight_decay_rule_(per_parameter_rule, remaining_module_parameters, optimizer_param_dict, decay_parameter_names)
        else:
            raise NotImplementedError(f"per_parameter_rule type {per_parameter_rule['type']} not implemented")


def parse_optimizer_per_params_config(model: nn.Module, criterion: Optional[nn.Module], optimizer_config: dict):
    model_named_parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    criterion_named_parameters = {name: param for name, param in criterion.named_parameters() if param.requires_grad} if criterion is not None else None

    if len(model_named_parameters) == 0 and (criterion_named_parameters is None or len(criterion_named_parameters) == 0):
        raise RuntimeError('No parameters to optimize.')

    base_lr = optimizer_config['lr']
    base_weight_decay = None
    if 'weight_decay' in optimizer_config:
        base_weight_decay = optimizer_config['weight_decay']
    optimizer_param_dict = []

    if 'per_parameter' in optimizer_config:
        _apply_per_parameter_adjustment_rules_(base_lr, base_weight_decay, optimizer_config['per_parameter'], model_named_parameters, optimizer_param_dict, model)
    if 'criterion' in optimizer_config and 'per_parameter' in optimizer_config['criterion']:
        _apply_per_parameter_adjustment_rules_(base_lr, base_weight_decay, optimizer_config['criterion']['per_parameter'], criterion_named_parameters, optimizer_param_dict, criterion)

    if criterion_named_parameters is not None and len(criterion_named_parameters) > 0:
        optimizer_param_dict.insert(0, {'params': list(criterion_named_parameters.values())})
    if len(model_named_parameters) > 0:
        optimizer_param_dict.insert(0, {'params': list(model_named_parameters.values())})

    _check_optimizer_params(model, criterion, optimizer_param_dict)

    return optimizer_param_dict


def _get_optimizer_param_stats(model: nn.Module, criterion: Optional[nn.Module],
                               optimizer_param_dict: List[Dict[str, Any]]):
    param_info = {}
    duplicate_param_names = []
    frozen_param_info = {}
    _get_optimizer_param_stats_helper(model, optimizer_param_dict, param_info, frozen_param_info, duplicate_param_names)
    if criterion is not None:
        _get_optimizer_param_stats_helper(criterion, optimizer_param_dict, param_info, frozen_param_info, duplicate_param_names, name_prefix='(criterion) ')

    return param_info, frozen_param_info, duplicate_param_names


def _get_optimizer_param_stats_helper(model: nn.Module, optimizer_param_dict: List[Dict[str, Any]],
                                      param_info: dict, frozen_param_names: dict, duplicate_param_names: list,
                                      name_prefix: str = ''):
    for name, param in model.named_parameters():
        name = name_prefix + name
        if not param.requires_grad:
            assert name not in frozen_param_names, f'Duplicate frozen parameter name: {name}'
            frozen_param_names[name] = {'size': str(list(param.shape))}
            continue
        found = False
        is_duplicate = False
        for optimizer_param_group in optimizer_param_dict:
            for curr_param in optimizer_param_group['params']:
                if param is curr_param:
                    if not found:
                        curr_param_info = {k: v for k, v in optimizer_param_group.items() if k != 'params'}
                        curr_param_info['size'] = str(list(param.shape))
                        assert name not in param_info, f'Duplicate parameter name: {name}'
                        param_info[name] = curr_param_info
                    if found:
                        is_duplicate = True
                    found = True
        if is_duplicate:
            duplicate_param_names.append(name)


def _check_optimizer_params(model: nn.Module, criterion: Optional[nn.Module],
                            optimizer_param_dict: List[Dict[str, Any]]):
    model_named_parameters_all = dict(model.named_parameters())
    # filter out frozen parameters
    model_named_parameters = {name: param for name, param in model_named_parameters_all.items() if param.requires_grad}
    if len(model_named_parameters) < len(model_named_parameters_all):
        print(f'optimizer: {len(model_named_parameters_all) - len(model_named_parameters)} out of {len(model_named_parameters_all)} parameters are frozen, not optimized.')

    if criterion is not None:
        criterion_named_parameters_all = dict(criterion.named_parameters())
        criterion_named_parameters = {name: param for name, param in criterion_named_parameters_all.items() if param.requires_grad}
        if len(criterion_named_parameters) < len(criterion_named_parameters_all):
            print(f'optimizer: (criterion) {len(criterion_named_parameters_all) - len(criterion_named_parameters)} out of {len(criterion_named_parameters_all)} parameters are frozen, not optimized.')

    param_info, frozen_param_names, duplicate_param_names = _get_optimizer_param_stats(model, criterion, optimizer_param_dict)

    for name, info in param_info.items():
        print(f'optimizer: {name}: ' + ', '.join([f'{k}: {v}' for k, v in info.items()]))

    if len(frozen_param_names) > 0:
        for name, info in frozen_param_names.items():
            print(f'optimizer: (frozen) {name}: ' + ', '.join([f'{k}: {v}' for k, v in info.items()]))

    if len(duplicate_param_names) > 0:
        raise RuntimeError(f'bug check: Duplicate param exists in optimizer param_groups: {duplicate_param_names}')
