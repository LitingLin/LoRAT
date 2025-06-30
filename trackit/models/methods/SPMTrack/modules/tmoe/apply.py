from typing import List

from torch import nn
from . import LinearWithTMoE, LinearWithTMoE_TimmQKV


def find_all_frozen_nn_linear_names(model, inference=False):
    cls = nn.Linear
    tmoe_module_names = set()
    for name, module in model.named_modules():
        if not inference:
            if isinstance(module, cls) and not module.weight.requires_grad:
                tmoe_module_names.add(name)
        else:
            if isinstance(module, cls):
                tmoe_module_names.add(name)

    return list(tmoe_module_names)


def apply_tmoe(model: nn.Module, tmoe_module_names: List[str],
               expert_r: int, expert_alpha: float,
               expert_dropout: float = 0.0, use_rsexpert: bool = False,
               expert_nums: int = 4, init_method: str = 'bert', shared_expert: bool = False, route_compression: bool = False):
    for tmoe_module_name in tmoe_module_names:
        tokens = tmoe_module_name.split('.')
        parent_module_name = '.'.join(tokens[:-1])
        module_name = tokens[-1]
        parent_module = model.get_submodule(parent_module_name)

        if module_name == 'qkv':
            layer_cls = LinearWithTMoE_TimmQKV
        else:
            layer_cls = LinearWithTMoE
        setattr(parent_module, module_name, layer_cls(getattr(parent_module, module_name), expert_r, expert_alpha,
                                                      expert_dropout, use_rsexpert, expert_nums=expert_nums,
                                                      init_method=init_method, shared_expert=shared_expert, route_compression=route_compression))
