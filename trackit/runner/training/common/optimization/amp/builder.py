from typing import Optional
import torch
from trackit.miscellanies.torch.amp_autocast import get_torch_amp_autocast_fn_and_grad_scaler
from .parameter_updater import ParameterUpdater_WithAMPSupport


def build_auto_mixed_precision_optimization_components(mixed_precision_config: dict, max_grad_norm: Optional[float], device: torch.device):
    enabled = mixed_precision_config['enabled']

    if 'grad_scaler' in mixed_precision_config:
        grad_scaler_config = mixed_precision_config['grad_scaler']
    else:
        grad_scaler_config = {}

    dtype = torch.float16
    if 'dtype' in mixed_precision_config:
        if mixed_precision_config['dtype'] == 'float16':
            dtype = torch.float16
        elif mixed_precision_config['dtype'] == 'bfloat16':
            dtype = torch.bfloat16
        else:
            raise NotImplementedError(f'Unsupported auto mixed precision dtype {mixed_precision_config["dtype"]}')

    autocast_fn, grad_scaler = get_torch_amp_autocast_fn_and_grad_scaler(device.type, enabled, dtype, **grad_scaler_config)
    if enabled and grad_scaler.is_enabled():
        print(f'Auto mixed precision is enabled: dtype: {dtype}'
              + (f' grad_scaler config: {grad_scaler_config}' if len(grad_scaler_config) > 0 else ''))

    return ParameterUpdater_WithAMPSupport(grad_scaler, max_grad_norm=max_grad_norm), autocast_fn
