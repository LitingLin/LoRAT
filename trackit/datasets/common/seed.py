from typing import Optional, Sequence, Union, Tuple
from trackit.core.runtime.global_constant import get_global_constant


class BaseSeed:
    name: str
    root_path: str
    data_split: Tuple[str, ...]
    extra_flags: Tuple[str, ...]
    version: int

    def __init__(self, name: str, root_path: str,
                 data_split: Optional[Union[str, Sequence[str]]] = None,
                 supported_data_splits: Optional[Tuple[str, ...]] = None,
                 extra_flags: Optional[Union[str, Sequence[str]]] = None,
                 version: int = 0):
        assert root_path is not None and len(root_path) > 0, 'root_path must be a valid path for dataset {}'.format(name)
        self.name = name
        self.root_path = root_path
        if data_split is None:
            data_split = ()
        if isinstance(data_split, str):
            data_split = (data_split,)
        elif isinstance(data_split, Sequence):
            data_split = tuple(data_split)
        else:
            raise ValueError('data_split must be a string or a sequence of strings')
        if supported_data_splits is None or len(supported_data_splits) == 0:
            assert len(data_split) == 0, 'data_split must be empty for dataset {}'.format(name)
        else:
            for ds in data_split:
                assert ds in supported_data_splits, 'data_split {} is not supported for dataset {}'.format(ds, name)
        self._data_split = data_split
        self._supported_data_splits = supported_data_splits
        if extra_flags is None:
            extra_flags = ()
        elif isinstance(extra_flags, str):
            extra_flags = (extra_flags,)
        elif isinstance(extra_flags, Sequence):
            extra_flags = tuple(extra_flags)
        else:
            raise ValueError('extra_flags must be a string or a sequence of strings')
        self.extra_flags = extra_flags
        self.version = version

    @property
    def data_split(self):
        return self._data_split

    @data_split.setter
    def data_split(self, value):
        if value is None:
            value = ()
        if isinstance(value, str):
            value = (value,)
        elif isinstance(value, Sequence):
            value = tuple(value)
        else:
            raise ValueError('data_split must be a string or a sequence of strings')
        if len(self._supported_data_splits) == 0:
            assert len(value) == 0, 'data_split must be empty for dataset {}'.format(self.name)
        else:
            for ds in value:
                assert ds in self._supported_data_splits, 'data_split {} is not supported for dataset {}'.format(ds, self.name)
        self._data_split = value

    @staticmethod
    def get_path_from_config(name: str):
        return get_global_constant(name)

    def construct(self, constructor):
        raise NotImplementedError()
