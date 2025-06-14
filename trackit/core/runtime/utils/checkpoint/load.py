import os
import torch
import torch.nn as nn
import safetensors.torch
from trackit.miscellanies.torch.check_version import is_torch_version_greater_or_equal
from . import application_state_load_fn


def load_model_weight(model: nn.Module, weight_path: str, use_safetensors=True, strict=True):
    if use_safetensors:
        return safetensors.torch.load_model(model, weight_path, strict=strict)
    else:
        return model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=strict)


def load_application_state(state_set_fn: application_state_load_fn, weight_path: str):
    if is_torch_version_greater_or_equal((1, 13)):
        application_state = torch.load(weight_path, map_location='cpu', weights_only=False)
    else:
        application_state = torch.load(weight_path, map_location='cpu')
    state_folder_path = os.path.dirname(weight_path)
    state_set_fn(application_state, state_folder_path)
