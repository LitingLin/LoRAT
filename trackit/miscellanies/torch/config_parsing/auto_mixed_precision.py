import torch

from typing import Optional
from .dtype import get_torch_dtype

def parse_auto_mixed_precision_config_with_machine_compatibility_checking(amp_config: dict, device_type: str) -> Optional[torch.dtype]:
    if amp_config is None:
        return None
    if not amp_config['enabled']:
        return None

    dtype = get_torch_dtype(amp_config.get('dtype', 'float16'))
    if device_type == 'cuda':
        if dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print('Auto mixed precision is disabled. reason: GPU does not support bfloat16.')
                return None
            return dtype
        return dtype
    elif device_type == 'mps':
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 5):
            print('Auto mixed precision is disabled. reason: Auto mixed precision is not supported on MPS with PyTorch < 2.5.0.')
            return None
        if dtype == torch.bfloat16:
            if pytorch_version < (2, 6):
                print('Auto mixed precision is disabled. reason: Auto mixed precision with bfloat16 is not supported on MPS with PyTorch < 2.6.0.')
                return None
            if not torch.backends.mps.is_macos_or_newer(14, 0):
                print('Auto mixed precision is disabled. reason: bfloat16 is not supported on MPS with macOS < 14.0.')
                return None
        return dtype
    if device_type == 'cpu':
        from trackit.miscellanies.system.machine.cpu_info import get_x86_cpu_features, get_arm_cpu_features, is_arm, is_x86
        if dtype == torch.float16:
            print('Auto mixed precision is disabled. reason: Auto mixed precision is not support on CPU with float16.')
            return None
        if is_arm:
            cpu_features = get_arm_cpu_features()
            if dtype == torch.bfloat16 and not cpu_features.has_bf16:
                print('Auto mixed precision is disabled. reason: Auto mixed precision is slow on CPU without FEAT_BF16 instructions support.')
                return None
            return dtype
        if is_x86:
            cpu_features = get_x86_cpu_features()
            if dtype == torch.float16 and not (cpu_features.has_avx512_fp16 or cpu_features.has_amx_fp16):
                print('Auto mixed precision is disabled. reason: Auto mixed precision is slow on CPU without AVX512_FP16 | AMX_FP16 instruction set.')
                return None
            return dtype
    return dtype