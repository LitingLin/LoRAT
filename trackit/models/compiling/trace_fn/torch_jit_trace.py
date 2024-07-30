import torch
import torch.nn as nn
from typing import Tuple


def trace_module_with_torch_jit_trace(module: nn.Module, data: Tuple[torch.Tensor, ...], input_names, output_names):
    return torch.jit.trace(module, data)
