from typing import Sequence
from trackit.data.source import TrackingDataset


def build_temporal_sampler_adaptor(adaptor_config: dict, datasets: Sequence[TrackingDataset]):
    if adaptor_config['type'] == 'passthrough':
        from .passthrough import PassThroughWrapper
        return PassThroughWrapper()
    elif adaptor_config['type'] == 'increased_difficulty':
        from .increased_difficulty import IncreasedDifficultyAuxFrameSampling
        return IncreasedDifficultyAuxFrameSampling(datasets)
    else:
        raise ValueError(f"Unknown adaptor type: {adaptor_config['type']}")
