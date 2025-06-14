from enum import Enum, auto
from dataclasses import dataclass


class SiamesePairSamplingMethod(Enum):
    interval = auto()
    causal = auto()
    reverse_causal = auto()


class SiamesePairNegativeSamplingMethod(Enum):
    random = auto()
    random_semantic_object = auto()


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
