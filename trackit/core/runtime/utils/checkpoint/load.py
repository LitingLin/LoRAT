from typing import Callable, Dict, Any
import torch
import torch.nn as nn
import safetensors.torch


def load_model_weight(model: nn.Module, weight_path: str, use_safetensors=True, strict=True):
    if use_safetensors:
        return safetensors.torch.load_model(model, weight_path, strict=strict)
    else:
        return model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=strict)


def load_application_state(state_set_fn: Callable[[Dict[str, Any]], None], weight_path: str):
    state_set_fn(torch.load(weight_path, map_location='cpu'))
