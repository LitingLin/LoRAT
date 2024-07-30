from typing import Sequence, Tuple
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.core.runtime.build_context import BuildContext
from .sampler import RandomSequencePicker
from .. import RandomAccessiblePerSequenceSampler
from ...utils.dataset_sampling_weight import get_dataset_sampling_weight


def build_random_sequence_picker(datasets: Sequence[TrackingDataset], sequence_picking_config: dict, build_context: BuildContext) -> Tuple[RandomAccessiblePerSequenceSampler, np.ndarray]:
    assert sequence_picking_config['type'] == 'random'
    if 'samples_per_epoch' in sequence_picking_config:
        samples_per_epoch = sequence_picking_config['samples_per_epoch']
    else:
        samples_per_epoch = sum(len(dataset) for dataset in datasets)

    seed = sequence_picking_config.get('seed', None)

    dataset_sampling_weights = get_dataset_sampling_weight(datasets, sequence_picking_config)
    sequence_picker = RandomSequencePicker(tuple(len(dataset) for dataset in datasets), dataset_sampling_weights, samples_per_epoch, seed, init=False)

    build_context.services.event.register_on_epoch_begin_event_listener(lambda epoch, is_train: sequence_picker.shuffle())
    build_context.services.checkpoint.register('random_sequence_picker', sequence_picker.get_state, sequence_picker.set_state)
    build_context.variables['num_samples_per_epoch'] = samples_per_epoch

    return sequence_picker, dataset_sampling_weights
