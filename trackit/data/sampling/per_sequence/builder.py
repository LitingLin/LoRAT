from typing import Sequence, Tuple
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.core.runtime.build_context import BuildContext
from . import RandomAccessiblePerSequenceSampler


def build_per_sequence_sampler(datasets: Sequence[TrackingDataset], sequence_picking_config: dict, build_context: BuildContext) -> Tuple[RandomAccessiblePerSequenceSampler, np.ndarray]:
    if sequence_picking_config['type'] == 'random':
        from .random.builder import build_random_sequence_picker
        return build_random_sequence_picker(datasets, sequence_picking_config, build_context)
    else:
        raise NotImplementedError('Unknown sequence picker type: {}'.format(sequence_picking_config['type']))
