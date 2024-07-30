import torch
from typing import Iterable, Sequence, Union


class TensorCache:
    def __init__(self, max_cache_length: int, dims: Sequence[int], device: torch.device, dtype=torch.float):
        self.shape = (max_cache_length, *dims)
        self.cache = torch.empty(self.shape, dtype=dtype, device=device)

    def put(self, index: int, tensor: torch.Tensor):
        self.cache[index, ...] = tensor

    def put_batch(self, indices: Sequence[int], tensor_list: Union[torch.Tensor, Iterable[torch.Tensor]]):
        assert len(indices) == len(tensor_list)
        if isinstance(tensor_list, torch.Tensor):
            self.cache[indices, ...] = tensor_list
        else:
            for index, tensor in zip(indices, tensor_list):
                self.cache[index, ...] = tensor

    def get_all(self):
        return self.cache

    def get_batch(self, indices: Sequence[int]):
        return self.cache[indices, ...]

    def get(self, index: int):
        return self.cache[index, ...]


class MultiScaleTensorCache:
    def __init__(self, max_num_elements: int, dims_list: Sequence[Sequence[int]], device: torch.device):
        self.shape_list = tuple((max_num_elements, *dims) for dims in dims_list)
        self.cache_list = tuple(torch.empty(shape, dtype=torch.float, device=device) for shape in self.shape_list)

    def put(self, index: int, multi_scale_tensor: Sequence[torch.Tensor]):
        assert len(multi_scale_tensor) == len(self.cache_list)
        for cache, tensor in zip(self.cache_list, multi_scale_tensor):
            cache[index, ...] = tensor

    def put_batch(self, indices: Sequence[int], multi_scale_tensor_list: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]):
        assert len(multi_scale_tensor_list) == len(self.cache_list)
        for cache, tensor_list in zip(self.cache_list, multi_scale_tensor_list):
            assert len(indices) == len(tensor_list)
            if isinstance(tensor_list, torch.Tensor):
                cache[indices, ...] = tensor_list
            else:
                for index, tensor in zip(indices, tensor_list):
                    cache[index, ...] = tensor

    def get_all(self):
        return self.cache_list

    def get_batch(self, indices):
        return tuple(cache[indices, ...] for cache in self.cache_list)

    def get(self, index):
        return tuple(cache[index, ...] for cache in self.cache_list)


class CacheService:
    def __init__(self, max_num_elements, cache):
        self.id_list = [None] * max_num_elements
        self.free_bits = [True] * max_num_elements
        self.cache = cache

    def put(self, id_, item):
        try:
            index = self.id_list.index(id_)
        except ValueError:
            index = None
        if index is not None:
            assert not self.free_bits[index]
        else:
            index = self.free_bits.index(True)
            self.id_list[index] = id_
            self.free_bits[index] = False
        self.cache.put(index, item)

    def put_batch(self, ids, items):
        indices = []
        for id_ in ids:
            index = self.id_list.index(id_) if id_ in self.id_list else None
            if index is not None:
                assert not self.free_bits[index]
            else:
                index = self.free_bits.index(True)
                self.id_list[index] = id_
                self.free_bits[index] = False
            indices.append(index)
        self.cache.put_batch(indices, items)

    def get_all(self):
        return self.cache.get_all()

    def delete(self, id_):
        index = self.id_list.index(id_)
        self.free_bits[index] = True
        self.id_list[index] = None

    def rename(self, old_id, new_id):
        assert new_id not in self.id_list
        index = self.id_list.index(old_id)
        self.id_list[index] = new_id

    def get(self, id_):
        index = self.id_list.index(id_)
        return self.cache.get(index)

    def get_batch(self, ids):
        indices = [self.id_list.index(id_) for id_ in ids]
        return self.cache.get_batch(indices)
