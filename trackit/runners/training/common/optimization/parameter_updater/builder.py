from typing import Optional
from contextlib import nullcontext

import torch
from trackit.miscellanies.torch.amp_autocast import get_torch_amp_autocast_fn_and_grad_scaler
from trackit.miscellanies.torch.config_parsing.auto_mixed_precision import parse_auto_mixed_precision_config_with_machine_compatibility_checking
from .parameter_updater import ParameterUpdater_WithAMPSupport
from ...torch_compile import CompiledAutogradOptions


def build_parameter_updater(mixed_precision_config: dict, max_grad_norm: Optional[float], device: torch.device,
                            compiled_autograd_option: CompiledAutogradOptions):
    amp_dtype = parse_auto_mixed_precision_config_with_machine_compatibility_checking(mixed_precision_config, device.type)
    enabled = amp_dtype is not None
    if enabled:
        assert amp_dtype in (torch.float16, torch.bfloat16), f'Unsupported auto mixed precision dtype {amp_dtype}'

    grad_scaler_config = mixed_precision_config.get('grad_scaler', {})

    autocast_fn, grad_scaler = get_torch_amp_autocast_fn_and_grad_scaler(device.type, enabled, amp_dtype, **grad_scaler_config)
    if enabled and autocast_fn is not nullcontext:
        print(f'Auto mixed precision training is enabled: dtype: {amp_dtype}'
              + (f' grad_scaler config: {grad_scaler_config}' if len(grad_scaler_config) > 0 and grad_scaler.is_enabled() else ''))

    return ParameterUpdater_WithAMPSupport(grad_scaler, max_grad_norm=max_grad_norm, 
                                           compiled_autograd_options=compiled_autograd_option), autocast_fn
