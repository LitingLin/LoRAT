import os
import copy
from typing import Sequence

from trackit.datasets.common.cache_path import prepare_dataset_cache_path
from trackit.datasets.common.types.exception import IncompatibleError


__all__ = ['DatasetFactory']

_base_impl_cache_extension = '.pkl'
_base_impl_cache_extension_yaml = '.yaml'
_memory_mapped_impl_cache_extension = '.np'


class _DatasetFactory:
    def __init__(self, seed, base_dataset_class, base_dataset_constructor, filter_func, specialized_dataset_class, specialized_dataset_converter):
        self.seed = seed
        self.base_dataset_class = base_dataset_class
        self.base_dataset_constructor = base_dataset_constructor
        self.filter_func = filter_func
        self.specialized_dataset_class = specialized_dataset_class
        self.specialized_dataset_converter = specialized_dataset_converter

    def get_dataset_name(self):
        if len(self.seed.data_split) == 0:
            return self.seed.name
        else:
            return self.seed.name + '-' + ''.join(self.seed.data_split)

    def construct(self, filters=None, cache_meta_data=False, dump_human_readable=False):
        if filters is not None and len(filters) == 0:
            filters = None

        dataset, cache_file_prefix = _try_load_from_cache(self.seed, self.specialized_dataset_class, _memory_mapped_impl_cache_extension, filters)
        if dataset is not None:
            return dataset
        base_dataset = self.construct_base_interface(filters, cache_meta_data, dump_human_readable)
        dataset = self.specialized_dataset_class(base_dataset.root_path,
                                                 self.specialized_dataset_converter(base_dataset.dataset, cache_file_prefix + _memory_mapped_impl_cache_extension))
        return dataset

    @staticmethod
    def _dump_base_dataset(dataset, path):
        temp_file_path = path + '.tmp'
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        dataset.dump(temp_file_path)
        os.rename(temp_file_path, path)

    @staticmethod
    def _dump_base_dataset_yaml(dataset, path):
        if not os.path.exists(path):
            temp_file_path = path + '.tmp'
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            dataset.dump_yaml(temp_file_path)
            os.rename(temp_file_path, path)

    @staticmethod
    def _dump(dataset, cache_path_prefix, dump, dump_human_readable):
        if dump:
            _DatasetFactory._dump_base_dataset(dataset, cache_path_prefix + _base_impl_cache_extension)
        if dump_human_readable:
            _DatasetFactory._dump_base_dataset_yaml(dataset, cache_path_prefix + _base_impl_cache_extension_yaml)

    def _construct_base_interface_unfiltered(self, make_cache=False, dump_human_readable=False):
        dataset, cache_file_prefix = _try_load_from_cache(self.seed, self.base_dataset_class, _base_impl_cache_extension, None)
        if dataset is not None:
            self._dump(dataset, cache_file_prefix, False, dump_human_readable)
            return dataset

        dataset = self.base_dataset_class(self.seed.root_path)
        with self.base_dataset_constructor(dataset.dataset, dataset.root_path, self.seed.version) as constructor:
            constructor.set_name(self.seed.name)
            constructor.set_split(self.seed.data_split)
            self.seed.construct(constructor)
        self._dump(dataset, cache_file_prefix, make_cache, dump_human_readable)
        return dataset

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False):
        if filters is not None and len(filters) == 0:
            filters = None

        dataset, cache_file_prefix = _try_load_from_cache(self.seed, self.base_dataset_class, _base_impl_cache_extension, filters)
        if dataset is not None:
            return dataset
        dataset = self._construct_base_interface_unfiltered(make_cache, dump_human_readable)
        if filters is None:
            return dataset
        self.filter_func(dataset.dataset, filters)
        self._dump(dataset, cache_file_prefix, make_cache, dump_human_readable)
        return dataset


class DatasetFactory:
    def __init__(self, seeds, base_dataset_class, base_dataset_constructor, filter_func, specialized_dataset_class, specialized_dataset_converter):
        expanded_seeds = []
        for seed in seeds:
            assert seed.data_split is None or isinstance(seed.data_split, Sequence)
            if seed.data_split is None or len(seed.data_split) == 0:
                expanded_seeds.append(seed)
            else:
                for data_split in seed.data_split:
                    new_seed = copy.copy(seed)
                    new_seed.data_split = (data_split,)
                    expanded_seeds.append(new_seed)

        self.factories = [_DatasetFactory(seed, base_dataset_class, base_dataset_constructor, filter_func, specialized_dataset_class, specialized_dataset_converter)
                          for seed in expanded_seeds]

    def construct(self, filters=None, cache_base_format=False, dump_human_readable=False):
        dataset_names = [factory.get_dataset_name() for factory in self.factories]
        print(f'Loading {", ".join(dataset_names)}...', end='')

        datasets = [factory.construct(filters, cache_base_format, dump_human_readable) for factory in self.factories]

        print('Done')
        return datasets

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False):
        return [factory.construct_base_interface(filters, make_cache, dump_human_readable) for factory in self.factories]


def _try_load_from_cache(seed, dataset_class, cache_extension, filters):
    cache_folder_path, cache_file_name = prepare_dataset_cache_path(dataset_class.__name__, seed.name, seed.data_split, filters)
    cache_file_path = os.path.join(cache_folder_path, cache_file_name + cache_extension)
    if os.path.exists(cache_file_path):
        try:
            dataset = dataset_class.load(seed.root_path, cache_file_path)
            if dataset.get_version() == seed.version:
                return dataset, os.path.join(cache_folder_path, cache_file_name)
            del dataset
        except IncompatibleError:
            pass
        os.remove(cache_file_path)
    return None, os.path.join(cache_folder_path, cache_file_name)
