from typing import NamedTuple, Optional, Tuple


class SequenceInfo(NamedTuple):
    dataset_name: Optional[str]
    data_split: Optional[Tuple[str, ...]]
    dataset_full_name: Optional[str]
    sequence_name: Optional[str]
    length: Optional[int]
    fps: Optional[float]
