import torch.nn as nn
from typing import Tuple
from torch.nn.modules.batchnorm import _NormBase

ALL_NORM_LAYERS = (nn.LayerNorm, nn.GroupNorm, _NormBase, nn.LocalResponseNorm)


def get_decay_parameter_names(model: nn.Module) -> Tuple[str, ...]:
    r"""
    Returns a list of names of parameters with weight decay. (weights in non-layernorm layers)
    """
    forbid_module_names = tuple(name for name, module in model.named_modules() if isinstance(module, ALL_NORM_LAYERS))
    decay_parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith(forbid_module_names) and param.squeeze().ndim > 1:
            decay_parameters.append(name)
    return tuple(decay_parameters)
