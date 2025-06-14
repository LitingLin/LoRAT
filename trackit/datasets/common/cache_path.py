import os
from typing import Optional, Sequence
from trackit.miscellanies.slugify import slugify
from trackit.datasets.common.unique_id import generate_dataset_unique_id

_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cache'))
_dir_created = set()


def get_dataset_attributes_cache_path():
    global _cache_dir
    return _cache_dir


def set_dataset_attributes_cache_path(path: str):
    global _cache_dir
    _cache_dir = path


def prepare_dataset_cache_path(dataset_type_name: str,
                               dataset_name: str,
                               dataset_splits: Optional[Sequence[str]],
                               dataset_flags: Optional[Sequence[str]],
                               dataset_filters: list):
    cache_path = get_dataset_attributes_cache_path()
    cache_path = os.path.join(cache_path, dataset_type_name)
    if dataset_filters is not None:
        cache_path = os.path.join(cache_path, 'filtered')
    if cache_path not in _dir_created:
        os.makedirs(cache_path, exist_ok=True)
        _dir_created.add(cache_path)
    dataset_unique_id = generate_dataset_unique_id(dataset_name, dataset_splits, dataset_flags, dataset_filters)
    cache_file_name_prefix = slugify(dataset_unique_id)
    return cache_path, cache_file_name_prefix
