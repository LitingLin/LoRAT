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
from typing import Iterable, Optional
from contextlib import nullcontext
from ...torch_compile import CompiledAutogradOptions
from trackit.miscellanies.torch.check_version import is_torch_version_greater_or_equal


@torch.no_grad()
def ampscaler_get_grad_norm(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    grads = tuple(p.grad for p in parameters if p.grad is not None)
    assert len(grads) != 0, "No grads to compute norm for"

    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g, 2.0) for g in grads]), 2.0
    )
    return total_norm


def _get_params_from_optimizer_grouped_params(optimizer: torch.optim.Optimizer):
    return tuple(p for group in optimizer.param_groups for p in group["params"])


class ParameterUpdater_WithAMPSupport:
    def __init__(
        self,
        grad_scaler: torch.cuda.amp.GradScaler,
        max_grad_norm: Optional[int] = None,
        compiled_autograd_options: CompiledAutogradOptions = CompiledAutogradOptions(),
    ):
        self._grad_scaler = grad_scaler
        self._max_grad_norm = max_grad_norm
        self._always_get_grad_norm = True
        self._first_autograd_call = True
        if compiled_autograd_options.enabled:
            if compiled_autograd_options.parameters is not None:
                import torch._dynamo.compiled_autograd
                def _context():
                    if self._first_autograd_call:
                        print(f"compiled autograd is enabled. "
                              f"torch.compile options: {compiled_autograd_options.parameters}, " 
                              f"dynamic={compiled_autograd_options.dynamic}")
                        self._first_autograd_call = False
                    if is_torch_version_greater_or_equal((2, 6)):
                        return torch._dynamo.compiled_autograd._enable(
                            torch.compile(**compiled_autograd_options.parameters), compiled_autograd_options.dynamic
                        )
                    else:
                        return torch._dynamo.compiled_autograd.enable(
                            torch.compile(dynamic=compiled_autograd_options.dynamic, **compiled_autograd_options.parameters)
                        )

                self._autograd_context = _context
            else:

                def _context():
                    if self._first_autograd_call:
                        print("compiled autograd is enabled with default option: fullgraph=True, dynamic=True.")
                        self._first_autograd_call = False
                    if is_torch_version_greater_or_equal((2, 6)):
                        return torch._dynamo.compiled_autograd._enable(torch.compile(fullgraph=True, dynamic=True))
                    else:
                        return torch._dynamo.compiled_autograd.enable(
                            torch.compile(fullgraph=True, dynamic=True)
                        )

                self._autograd_context = _context
        else:
            self._autograd_context = nullcontext

    def is_grad_scaler_enabled(self):
        return self._grad_scaler.is_enabled()

    def has_grad_norm(self):
        return self._always_get_grad_norm or self._max_grad_norm is not None

    def backward_and_unscale(
        self, 
        loss: torch.Tensor, 
        optimizer: torch.optim.Optimizer, 
        create_graph: bool = False, 
        update_grad: bool = True
    ) -> tuple[float, Optional[float]]:
        loss = self._grad_scaler.scale(loss)
        with self._autograd_context():
            loss.backward(create_graph=create_graph)
        norm = None
        if update_grad:
            self._grad_scaler.unscale_(
                optimizer
            )  # unscale the gradients of optimizer's assigned params in-place
            if self._max_grad_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(
                    _get_params_from_optimizer_grouped_params(optimizer),
                    self._max_grad_norm,
                ).item()
            elif self._always_get_grad_norm:
                norm = ampscaler_get_grad_norm(
                    _get_params_from_optimizer_grouped_params(optimizer)
                ).item()
        return self._grad_scaler.get_scale(), norm

    def step(self, optimizer, update_grad=True):
        if update_grad:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()

    def state_dict(self):
        return self._grad_scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._grad_scaler.load_state_dict(state_dict)
