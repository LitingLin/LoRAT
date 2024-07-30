from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProgress_Dataset:
    total_repeat_times: int
    all_evaluated: bool
    this_repeat_all_evaluated: bool


@dataclass(frozen=True)
class EvaluationProgress:
    repeat_index: int
    this_dataset: Optional[EvaluationProgress_Dataset]


class EvaluationTaskTracer:
    def submit(self, dataset_full_name: str, track_name: str) -> EvaluationProgress:
        raise NotImplementedError()
