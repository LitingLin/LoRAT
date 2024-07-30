import torch
from contextlib import nullcontext
from functools import partial


def get_torch_amp_autocast_fn(device_type: str, enabled: bool, dtype: torch.dtype):
    # to be removed in the future, once they are supported
    if device_type == 'mps' and enabled:
        print('Auto mixed precision is disabled. reason: Auto mixed precision is not supported on MPS.', flush=True)
        enabled = False
    if device_type == 'cpu' and dtype == torch.float16 and enabled:
        dtype = torch.bfloat16
        print(f'Warning: CPU does not support float16, use bfloat16 instead', flush=True)
    return partial(torch.amp.autocast, device_type=device_type, enabled=enabled, dtype=dtype) if enabled else nullcontext


def get_torch_amp_autocast_fn_and_grad_scaler(device_type: str, enabled: bool, dtype: torch.dtype, **grad_scaler_kwargs):
    if device_type != 'cuda' and enabled:
        print('Auto mixed precision training is disabled. reason: Auto mixed precision training is only supported on CUDA.', flush=True)
        enabled = False
    from torch.cuda.amp.grad_scaler import GradScaler
    return partial(torch.amp.autocast, device_type=device_type, enabled=enabled, dtype=dtype) if enabled else nullcontext, GradScaler(enabled=enabled, **grad_scaler_kwargs)
