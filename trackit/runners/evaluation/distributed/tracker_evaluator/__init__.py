from dataclasses import dataclass

import torch
import torch.nn as nn

from typing import Any, Optional, Mapping
from trackit.data.protocol.eval_input import TrackerEvalData


@dataclass(frozen=True)
class EvaluatorContext:
    epoch: int
    max_batch_size: int
    num_input_data_streams: int
    dtype: torch.dtype
    auto_mixed_precision_dtype: Optional[torch.dtype]
    model: nn.Module


class TrackerEvaluator:
    def start(self, context: EvaluatorContext):
        pass

    def stop(self, context: EvaluatorContext):
        pass

    def run(self, data: Optional[TrackerEvalData], optimized_model: Any, raw_model: nn.Module) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError
