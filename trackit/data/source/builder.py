from typing import Sequence
from . import TrackingDataset


def build_data_source(data_source_config: dict) -> Sequence[TrackingDataset]:
    if data_source_config['type'] == 'dataset':
        from .dataset.builder import build_dataset_as_data_source
        datasets = build_dataset_as_data_source(data_source_config)
    else:
        raise NotImplementedError(f'Unknown data source type: {data_source_config["type"]}')
    return datasets
