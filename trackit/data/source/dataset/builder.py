import importlib

from datetime import timedelta
from trackit.miscellanies.torch.distributed.barrier import torch_distributed_zero_first
from trackit.core.runtime.global_constant import get_global_constant
from trackit.datasets.DET.specialization.memory_mapped.dataset import DetectionDataset_MemoryMapped
from trackit.datasets.SOT.specialization.memory_mapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from trackit.datasets.MOT.specialization.memory_mapped.dataset import MultipleObjectTrackingDataset_MemoryMapped

from .SOT import SOTDataset
from .MOT import MOTDataset
from .DET import DETDataset


def _parse_filters(config: list):
    filters = []
    for filter_config in config:
        filter_type = filter_config['type']
        filter_module_path_components = filter_type.split('.')
        module = importlib.import_module('trackit.datasets.common.filter.' + '.'.join(filter_module_path_components))
        filter_class_name_components = (''.join(c.title() for c in component.split('_')) for component in filter_module_path_components)
        filter_class = getattr(module, '_'.join(filter_class_name_components))
        filters.append(filter_class() if 'parameters' not in filter_config else filter_class(**filter_config['parameters']))
    return filters


def build_datasets_from_config(config: dict):
    filters = []
    if 'filters' in config:
        dataset_filter_names = config['filters']
        filters.extend(_parse_filters(dataset_filter_names))

    datasets = []

    constructor_params = {}
    if 'config' in config:
        dataset_building_config = config['config']
        if 'dump_human_readable' in dataset_building_config:
            constructor_params['dump_human_readable'] = dataset_building_config['dump_human_readable']
        if 'cache_meta_data' in dataset_building_config:
            constructor_params['cache_base_format'] = dataset_building_config['cache_base_format']

    for dataset_building_parameter in config['datasets']:
        dataset_name = dataset_building_parameter['name']
        dataset_type = dataset_building_parameter['type']

        if dataset_type == 'SOT':
            from trackit.datasets.SOT.factory import SingleObjectTrackingDatasetFactory
            module = importlib.import_module('trackit.datasets.SOT.datasets.{}'.format(dataset_name))
            factory_class = SingleObjectTrackingDatasetFactory
        elif dataset_type == 'MOT':
            from trackit.datasets.MOT.factory import MultipleObjectTrackingDatasetFactory
            module = importlib.import_module('trackit.datasets.MOT.datasets.{}'.format(dataset_name))
            factory_class = MultipleObjectTrackingDatasetFactory
        elif dataset_type == 'DET':
            from trackit.datasets.DET.factory import DetectionDatasetFactory
            module = importlib.import_module('trackit.datasets.DET.datasets.{}'.format(dataset_name))
            factory_class = DetectionDatasetFactory
        else:
            raise Exception('Unsupported dataset type {}'.format(dataset_type))

        seed_class = getattr(module, '{}_Seed'.format(dataset_name))

        seeds = []
        seed_parameters = {}

        if 'path' in dataset_building_parameter:
            seed_parameters['root_path'] = dataset_building_parameter['path']

        if 'parameters' in dataset_building_parameter:
            seed_parameters.update(dataset_building_parameter['parameters'])

        data_splits = []
        if 'splits' in dataset_building_parameter:
            if isinstance(dataset_building_parameter['splits'], str):
                data_splits.append(dataset_building_parameter['splits'])
            else:
                data_splits.extend(dataset_building_parameter['splits'])

        if len(data_splits) == 0:
            seeds.append(seed_class(**seed_parameters))
        else:
            for data_split in data_splits:
                seeds.append(seed_class(data_split=data_split, **seed_parameters))

        factory = factory_class(seeds)

        if 'filters' in dataset_building_parameter:
            dataset_filters = _parse_filters(dataset_building_parameter['filters'])
            dataset_filters.extend(filters)
        else:
            dataset_filters = filters

        if len(dataset_filters) == 0:
            dataset_filters = None

        dataset = factory.construct(dataset_filters, **constructor_params)

        datasets.extend(dataset)

    return datasets


def _build_dataset(data_source_config: dict):
    assert data_source_config['type'] == 'dataset'
    dataset_configs = data_source_config['parameters']

    with torch_distributed_zero_first(on_local_master=not get_global_constant('on_shared_file_system'), timeout=timedelta(hours=5)):
        return build_datasets_from_config(dataset_configs)


def build_dataset_as_data_source(data_source_config: dict):
    datasets = _build_dataset(data_source_config)
    wrapped_datasets = []
    for dataset in datasets:
        if isinstance(dataset, DetectionDataset_MemoryMapped):
            wrapped_datasets.append(DETDataset(dataset))
        elif isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
            wrapped_datasets.append(SOTDataset(dataset))
        elif isinstance(dataset, MultipleObjectTrackingDataset_MemoryMapped):
            wrapped_datasets.append(MOTDataset(dataset))
        else:
            raise NotImplementedError(f'Unknown dataset type: {type(dataset)}')
    return wrapped_datasets
