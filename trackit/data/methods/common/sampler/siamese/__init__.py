from typing import Sequence, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingResult_Element:
    dataset_index: int
    sequence_index: int
    track_id: int
    frame_index: int


@dataclass(frozen=True)
class SiameseTrainingPairSamplingResult:
    z: SamplingResult_Element
    x: SamplingResult_Element
    is_positive: bool
    aux_frames: Optional[Sequence[SamplingResult_Element]] = None
