# modified from https://github.com/microsoft/Swin-Transformer/blob/main/utils.py
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
import torch.optim
import torch.linalg
from typing import Iterable


def ampscaler_get_grad_norm(parameters: Iterable[torch.Tensor], norm_type: float = 2.0) -> torch.Tensor:
    parameters = tuple(p for p in parameters if p.grad is not None)
    norm_type = float(norm_type)
    assert len(parameters) != 0, "No params to compute norm for"
    return torch.linalg.norm(torch.stack(tuple(torch.linalg.norm(p.grad.detach(), norm_type) for p in parameters)), norm_type)


def _get_params_from_optimizer_grouped_params(optimizer: torch.optim.Optimizer):
    return tuple(p for group in optimizer.param_groups for p in group['params'])


class ParameterUpdater_WithAMPSupport:
    def __init__(self, grad_scaler: torch.cuda.amp.GradScaler, max_grad_norm=None):
        self._grad_scaler = grad_scaler
        self._max_grad_norm = max_grad_norm
        self._always_get_grad_norm = True

    def is_grad_scaler_enabled(self):
        return self._grad_scaler.is_enabled()

    def has_grad_norm(self):
        return self._always_get_grad_norm or self._max_grad_norm is not None

    def backward_and_unscale(self, loss, optimizer, create_graph=False, update_grad=True):
        self._grad_scaler.scale(loss).backward(create_graph=create_graph)
        norm = None
        if update_grad:
            self._grad_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            if self._max_grad_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(_get_params_from_optimizer_grouped_params(optimizer), self._max_grad_norm).item()
            elif self._always_get_grad_norm:
                norm = ampscaler_get_grad_norm(_get_params_from_optimizer_grouped_params(optimizer)).item()
        return self._grad_scaler.get_scale(), norm

    def step(self, optimizer, update_grad=True):
        if update_grad:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()

    def state_dict(self):
        return self._grad_scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._grad_scaler.load_state_dict(state_dict)
