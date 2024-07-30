from dataclasses import dataclass
from typing import Set, Optional, Tuple


@dataclass(frozen=True)
class DatasetEvaluationTask:
    dataset_name: str
    data_split: Optional[Tuple[str, ...]]
    dataset_full_name: str
    track_names: Set[str]
    repeat_times: int
