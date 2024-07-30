from typing import List

from torch import nn
from . import LinearWithLoRA, LinearWithLoRA_TimmQKV
# Modified from https://github.com/artidoro/qlora/blob/main/qlora.py


def find_all_frozen_nn_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and not module.weight.requires_grad:
            lora_module_names.add(name)

    return list(lora_module_names)


def apply_lora(model: nn.Module, lora_module_names: List[str],
               lora_r: int, lora_alpha: float,
               lora_dropout: float = 0.0, use_rslora: bool = False):
    for lora_module_name in lora_module_names:
        tokens = lora_module_name.split('.')
        parent_module_name = '.'.join(tokens[:-1])
        module_name = tokens[-1]
        parent_module = model.get_submodule(parent_module_name)

        if module_name == 'qkv':
            layer_cls = LinearWithLoRA_TimmQKV
        else:
            layer_cls = LinearWithLoRA
        setattr(parent_module, module_name, layer_cls(getattr(parent_module, module_name), lora_r, lora_alpha,
                                                      lora_dropout, use_rslora))
