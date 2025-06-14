import math
from typing import Optional
import torch


def _lora_delta(lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float], use_rslora: bool) -> torch.Tensor:
    r = lora_A.size(0)
    if alpha is not None:
        if use_rslora:
            scaling = alpha / math.sqrt(r)
        else:
            scaling = alpha / r
    else:
        scaling = 1.
    return (lora_B @ lora_A) * scaling


def lora_merge(weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float], use_rslora: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _lora_delta(lora_A.to(torch.float32), lora_B.to(torch.float32), alpha, use_rslora)

    return (weight.to(torch.float32) + delta).to(original_dtype)


def lora_unmerge(weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float], use_rslora: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _lora_delta(lora_A.to(torch.float32), lora_B.to(torch.float32), alpha, use_rslora)

    return (weight.to(torch.float32) - delta).to(original_dtype)
