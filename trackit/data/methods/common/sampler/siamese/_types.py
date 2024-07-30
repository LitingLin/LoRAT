from enum import Enum, auto


class SiamesePairSamplingMethod(Enum):
    interval = auto()
    causal = auto()
    reverse_causal = auto()


class SiamesePairNegativeSamplingMethod(Enum):
    random = auto()
    random_semantic_object = auto()
    distractor = auto()
