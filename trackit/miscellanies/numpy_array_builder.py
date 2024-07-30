import numpy as np
from typing import Sequence, Union


class _Iterator:
    def __init__(self, array, size_limit):
        self.array = array
        self.size_limit = size_limit
        self.index = 0

    def __next__(self):
        if self.index == self.size_limit:
            raise StopIteration
        value = self.array[self.index]
        self.index += 1
        return value


class NumpyArrayBuilder:
    def __init__(self, dtype, /, initial_capacity=32, grow_ratio=2, extra_dims=()):
        self.data = np.empty((initial_capacity, *extra_dims), dtype=dtype)
        self.capacity = initial_capacity
        self.grow_ratio = grow_ratio
        self.size = 0

    def reserve(self, size: int):
        assert size > self.size
        new_data = np.empty((size, *self.data.shape[1:]), dtype=self.data.dtype)
        new_data[:self.size] = self.data

        self.data = new_data
        self.capacity = size

    def extend(self, array: Union[Sequence[float], np.ndarray]):
        new_min_size = self.size + len(array)
        if new_min_size > self.capacity:
            self.reserve(new_min_size)
        if isinstance(array, np.ndarray):
            self.data[self.size: new_min_size] = array
        else:
            for index, value in enumerate(array):
                self.data[self.size + index] = value
        self.size = new_min_size

    def append(self, x: Union[float, np.ndarray]):
        if self.size == self.capacity:
            self.reserve(self.capacity * self.grow_ratio)

        self.data[self.size] = x
        self.size += 1

    def _update_selector(self, selector: Union[int, slice, tuple]):
        if isinstance(selector, tuple):
            if isinstance(selector[0], slice):
                if selector[0].stop is None:
                    selector = (slice(selector[0].start, self.size, selector[0].step), *[selector[i] for i in range(1, len(selector))])
                else:
                    assert selector[0].stop < self.size
            else:
                assert selector[0] < self.size
        elif isinstance(selector, slice):
            if selector.stop is None:
                selector = slice(selector.start, self.size, selector.step)
            else:
                assert selector.stop < self.size
        else:
            assert selector < self.size
            assert selector >= -self.size
        return selector

    def __getitem__(self, key: Union[int, slice, tuple]) -> np.ndarray:
        # key = self._update_selector(key)
        return self.data[: self.size].__getitem__(key)

    def __setitem__(self, key: Union[int, slice, tuple], value: Union[float, np.ndarray]):
        # key = self._update_selector(key)
        self.data[: self.size].__setitem__(key, value)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return _Iterator(self.data, self.size)

    def build(self, readonly=True) -> np.ndarray:
        view = self.data[: self.size]
        if readonly:
            view.flags.writeable = False
        return view
