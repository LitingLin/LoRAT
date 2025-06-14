import torch
from contextlib import nullcontext
from functools import partial


def get_torch_amp_autocast_fn_and_grad_scaler(device_type: str, enabled: bool, dtype: torch.dtype, **grad_scaler_kwargs):
    if enabled:
        if torch.torch_version.__version__ < (2, 3) and device_type != 'cuda':
            print('Auto mixed precision training is disabled. reason: Auto mixed precision training is only supported on CUDA with Pytorch < 2.3.')
            enabled = False
        else:
            if not device_type in ('cuda', 'cpu'):
                print(f'Auto mixed precision training is disabled. reason: torch.amp.grad_scaler is not supported on {device_type}.')
                enabled = False

    grad_scaler_enabled = enabled and dtype == torch.float16
    if torch.torch_version.__version__ >= (2, 4):
        from torch.amp.grad_scaler import GradScaler
        grad_scaler = GradScaler(device_type, enabled=grad_scaler_enabled, **grad_scaler_kwargs)
    else:
        if device_type == 'cpu' and ((2, 3) <= torch.torch_version.__version__ < (2, 4)):
            from torch.cpu.amp.grad_scaler import GradScaler
        else:
            from torch.cuda.amp.grad_scaler import GradScaler
        grad_scaler = GradScaler(enabled=grad_scaler_enabled, **grad_scaler_kwargs)

    return get_torch_amp_autocast_fn(device_type, enabled, dtype), grad_scaler


def get_torch_amp_autocast_fn(device_type: str, enabled: bool, dtype: torch.dtype):
    return partial(torch.amp.autocast, device_type=device_type, enabled=True, dtype=dtype) if enabled else nullcontext
