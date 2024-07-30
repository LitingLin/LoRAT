import torch
from torch import nn
from typing import Any, Tuple
from trackit.models import ModelManager


class InferenceEngine:
    def __call__(self, model_manager: ModelManager, device: torch.device, max_batch_size: int) -> Tuple[Any, nn.Module]:
        raise NotImplementedError()
