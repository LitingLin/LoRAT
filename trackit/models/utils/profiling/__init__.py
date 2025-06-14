from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferencePrecision:
    dtype: torch.dtype = torch.float32
    auto_mixed_precision_enabled: bool = False
    auto_mixed_precision_dtype: torch.dtype = torch.float16
