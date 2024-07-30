from dataclasses import dataclass
import torch
from typing import Optional, Mapping


@dataclass(frozen=True)
class CriterionOutput:
    loss: torch.Tensor
    metrics: Optional[Mapping[str, float]] = None
    extra_metrics: Optional[Mapping[str, float]] = None
