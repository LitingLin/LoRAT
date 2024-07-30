import numpy as np
from typing import Sequence
from trackit.data.source import TrackingDataset
from trackit.data.utils.data_source_matcher.builder import build_data_source_matcher


def get_dataset_sampling_weight(datasets: Sequence[TrackingDataset], sequence_picking_config: dict) -> np.ndarray:
    if 'weight' in sequence_picking_config:
        sampling_context = []

        dataset_matchers = []
        for rule in sequence_picking_config['weight']:
            dataset_matchers.append(build_data_source_matcher(rule['match']))
            sampling_context.append(([], rule['value']))

        weights = np.empty(len(datasets), dtype=np.float64)
        weights.fill(float('nan'))

        for index_of_dataset, dataset in enumerate(datasets):
            dataset_name = dataset.get_name()
            dataset_split = dataset.get_data_split()
            matched = False
            for index_of_dataset_matcher, dataset_matcher in enumerate(dataset_matchers):
                if dataset_matcher(dataset_name, dataset_split):
                    matched = True
                    sampling_context[index_of_dataset_matcher][0].append(index_of_dataset)
                    break
            if not matched:
                sampling_context.append(((index_of_dataset,), 1.))

        for dataset_indices, weight in sampling_context:
            sub_dataset_weight = np.empty(len(dataset_indices), dtype=np.float64)
            sub_dataset_weight.fill(float('nan'))
            for i, dataset_index in enumerate(dataset_indices):
                sub_dataset_weight[i] = len(datasets[dataset_index])
            sub_dataset_weight /= sub_dataset_weight.sum()
            sub_dataset_weight *= weight
            for i, dataset_index in enumerate(dataset_indices):
                weights[dataset_index] = sub_dataset_weight[i]
    else:
        weights = np.array([len(dataset) for dataset in datasets], dtype=np.float64)
    weights /= weights.sum()
    return weights
