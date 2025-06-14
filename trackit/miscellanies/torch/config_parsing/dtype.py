import torch
from typing import Union


# https://github.com/pytorch/pytorch/issues/74669#issuecomment-2561908896
def get_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)

    return dtype


def get_torch_dtype_with_machine_compatibility_checking(dtype: Union[torch.dtype, str], device_name: str) -> torch.dtype:
    dtype = get_torch_dtype(dtype)
    if device_name == 'cuda':
        if dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print('GPU does not support bfloat16, falling back to float32.')
                return torch.float32
        return dtype
    elif device_name == 'cpu':
        if dtype == torch.float16:
            from trackit.miscellanies.system.machine.cpu_info import get_x86_cpu_features, get_arm_cpu_features, is_arm, is_x86
            if is_arm:
                cpu_features = get_arm_cpu_features()
                if not cpu_features.has_fp16:
                    print('CPU does not support float16 arithmetic instructions, falling back to float32.')
                    return torch.float32
            elif is_x86:
                cpu_features = get_x86_cpu_features()
                if not (cpu_features.has_avx512_fp16 or cpu_features.has_amx_fp16):
                    print('CPU does not support float16 arithmetic instructions, falling back to float32.')
                    return torch.float32
        elif dtype == torch.bfloat16:
            from trackit.miscellanies.system.machine.cpu_info import get_x86_cpu_features, get_arm_cpu_features, is_arm, is_x86
            if is_arm:
                cpu_features = get_arm_cpu_features()
                if not cpu_features.has_bf16:
                    print('CPU does not support bfloat16 arithmetic instructions, falling back to float32.')
                    return torch.float32
            elif is_x86:
                cpu_features = get_x86_cpu_features()
                if not (cpu_features.has_avx512_bf16 or cpu_features.has_amx_bf16):
                    print('CPU does not support bfloat16 arithmetic instructions, falling back to float32.')
                    return torch.float32
        return dtype
    elif device_name == 'mps':
        if dtype == torch.bfloat16:
            if not torch.backends.mps.is_macos_or_newer(14, 0):
                print('bfloat16 is not supported on Metal Performance Shaders with macOS < 14.0, falling back to float32.')
                return torch.float32
        return dtype
    return dtype
