import torch
from torch import nn
from typing import Any, Optional
from trackit.models import ModelManager
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizedModel:
    model: Any
    raw_model: nn.Module
    device: torch.device
    dtype: torch.dtype
    auto_mixed_precision_dtype: Optional[torch.dtype]


class InferenceEngine:
    def __call__(self, model_manager: ModelManager, device: torch.device, dtype: torch.dtype,
                 max_batch_size: int, max_num_input_data_streams: int) -> OptimizedModel:
        raise NotImplementedError()
