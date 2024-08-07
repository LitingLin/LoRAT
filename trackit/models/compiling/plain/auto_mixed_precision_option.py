import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class AutoMixedPrecisionOption:
    enabled: bool = False
    dtype: torch.dtype = torch.float16

    @staticmethod
    def from_config(config: dict):
        dtype = torch.float16
        if 'dtype' in config:
            dtype = config['dtype']
            if dtype == 'float16':
                dtype = torch.float16
            elif dtype == 'bfloat16':
                dtype = torch.bfloat16
            else:
                raise ValueError('dtype: {} is not supported for auto mixed precision'.format(dtype))
        return AutoMixedPrecisionOption(
            enabled=config['enabled'],
            dtype=dtype
        )
