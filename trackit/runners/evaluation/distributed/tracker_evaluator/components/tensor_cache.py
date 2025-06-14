import torch
from typing import Iterable, Sequence, Union, Any, List


class TensorCache:
    def __init__(self, max_cache_length: int, dims: Sequence[int], device: torch.device,
                 dtype: torch.dtype = torch.float):
        assert max_cache_length > 0
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

    def get_batch(self, indices: Sequence[int]):
        return self.cache[indices, ...]

    def get(self, index: int):
        return self.cache[index, ...]

    def size_in_bytes(self):
        return self.cache.element_size() * self.cache.nelement()

    def free(self, _):
        # no op
        pass

    def capacity(self):
        return self.shape[0]


class MultiTensorCache:
    def __init__(self, max_num_elements: int, dims_list: Sequence[Sequence[int]], device: torch.device,
                 dtype: torch.dtype = torch.float):
        assert max_num_elements > 0
        self.shape_list = tuple((max_num_elements, *dims) for dims in dims_list)
        self.cache_list = tuple(torch.empty(shape, dtype=dtype, device=device) for shape in self.shape_list)

    def put(self, index: int, tensors: Sequence[torch.Tensor]):
        assert len(tensors) == len(self.cache_list)
        for cache, tensor in zip(self.cache_list, tensors):
            cache[index, ...] = tensor

    def put_batch(self, indices: Sequence[int],
                  tensors_list: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]):
        assert len(tensors_list) == len(self.cache_list)
        for cache, tensors in zip(self.cache_list, tensors_list):
            assert len(indices) == len(tensors)
            if isinstance(tensors, torch.Tensor):
                cache[indices, ...] = tensors
            else:
                for index, tensor in zip(indices, tensors):
                    cache[index, ...] = tensor

    def get_batch(self, indices: Sequence[int]):
        return tuple(cache[indices, ...] for cache in self.cache_list)

    def get(self, index: int):
        return tuple(cache[index, ...] for cache in self.cache_list)

    def size_in_bytes(self):
        return sum(cache.element_size() * cache.nelement() for cache in self.cache_list)

    def free(self, _):
        # no op
        pass

    def capacity(self):
        return self.shape_list[0][0]


class TensorCache_ZeroCopy:
    def __init__(self, max_cache_length: int, dims: Sequence[int], device: torch.device,
                 dtype: torch.dtype = torch.float):
        occupied_bytes = dtype.itemsize * max_cache_length
        for dim in dims:
            occupied_bytes *= dim
        self.max_occupied_bytes = occupied_bytes
        self.max_cache_length = max_cache_length
        self.cache: List[torch.Tensor | None] = [None for _ in range(max_cache_length)]

    def put(self, index: int, tensor: torch.Tensor):
        self.cache[index] = tensor

    def put_batch(self, indices: Sequence[int], tensor_list: Union[torch.Tensor, Iterable[torch.Tensor]]):
        assert len(indices) == len(tensor_list)
        for index, tensor in zip(indices, tensor_list):
            self.cache[index] = tensor

    def get_batch(self, indices: Sequence[int]):
        return torch.stack([self.cache[index] for index in indices], dim=0)

    def get(self, index: int):
        return self.cache[index]

    def size_in_bytes(self):
        return self.max_occupied_bytes

    def free(self, index: int):
        self.cache[index] = None

    def capacity(self):
        return self.max_cache_length


class MultiTensorCache_ZeroCopy:
    def __init__(self, max_num_elements: int, dims_list: Sequence[Sequence[int]],
                 dtype: torch.dtype = torch.float):
        base_bytes = dtype.itemsize * max_num_elements
        occupied_bytes = 0
        for dim_list in dims_list:
            b = base_bytes
            for dim in dim_list:
                b *= dim
            occupied_bytes += b
        self.max_occupied_bytes = occupied_bytes
        self.max_num_elements = max_num_elements
        self.cache_list: List[List[torch.Tensor | None]] = [[None for _ in range(max_num_elements)] for _ in range(len(dims_list))]

    def put(self, index: int, tensors: Sequence[torch.Tensor]):
        assert len(tensors) == len(self.cache_list)
        for cache, tensor in zip(self.cache_list, tensors):
            cache[index] = tensor

    def put_batch(self, indices: Sequence[int],
                  tensors_list: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]):
        assert len(tensors_list) == len(self.cache_list)
        for cache, tensors in zip(self.cache_list, tensors_list):
            assert len(indices) == len(tensors)
            for index, tensor in zip(indices, tensors):
                cache[index] = tensor

    def get_batch(self, indices: Sequence[int]):
        return tuple(torch.stack([cache[index] for index in indices], dim=0) for cache in self.cache_list)

    def get(self, index: int):
        return tuple(cache[index] for cache in self.cache_list)

    def size_in_bytes(self):
        return self.max_occupied_bytes

    def free(self, index: int):
        for cache in self.cache_list:
            cache[index] = None

    def capacity(self):
        return self.max_num_elements


class CacheService:
    def __init__(self, cache: Union[TensorCache, MultiTensorCache, TensorCache_ZeroCopy, MultiTensorCache_ZeroCopy]):
        self.index_allocator = IndexAllocator(cache.capacity())
        self.cache = cache

    def empty(self):
        return self.index_allocator.empty()

    def allocate(self, id_: Any):
        return self.index_allocator.allocate(id_)

    def put(self, id_: Any, item):
        self.cache.put(self.index_allocator.allocate(id_), item)

    def put_batch(self, ids: Sequence[Any], items: Union[torch.Tensor, Sequence[Any]]):
        self.cache.put_batch([self.index_allocator.allocate(id_) for id_ in ids], items)

    def delete(self, id_: Any):
        self.cache.free(self.index_allocator.free(id_))

    def delete_batch(self, ids: Sequence[Any]):
        for id_ in ids:
            self.cache.free(self.index_allocator.free(id_))

    def rename(self, old_id: Any, new_id: Any):
        self.index_allocator.rename(old_id, new_id)

    def get(self, id_: Any):
        return self.cache.get(self.index_allocator.get_index(id_)) # raise on not found

    def get_batch(self, ids: Sequence[Any]):
        return self.cache.get_batch([self.index_allocator.get_index(id_) for id_ in ids])

    def size_in_bytes(self):
        return self.cache.size_in_bytes()

    def has(self, id_):
        return self.index_allocator.has(id_)

class IndexAllocator:
    def __init__(self, max_num_elements):
        self.id_to_index = {}
        self.free_indices = list(range(max_num_elements))

    def empty(self):
        return len(self.id_to_index) == 0

    def has(self, id_):
        return id_ in self.id_to_index

    def allocate(self, id_):
        # Return existing index if id is already allocated
        if id_ in self.id_to_index:
            return self.id_to_index[id_]
        # Allocate a new index if available
        if not self.free_indices:
            raise RuntimeError("No free indices available")
        index = self.free_indices.pop()
        self.id_to_index[id_] = index
        return index

    def free(self, id_):
        # Free the index associated with the id
        if id_ not in self.id_to_index:
            raise ValueError(f"ID {id_} not found")
        index = self.id_to_index.pop(id_)
        self.free_indices.append(index)
        return index

    def rename(self, old_id, new_id):
        # Rename an id, ensuring the new id doesn't exist
        if new_id in self.id_to_index:
            raise ValueError(f"New ID {new_id} already exists")
        if old_id not in self.id_to_index:
            raise ValueError(f"Old ID {old_id} not found")
        index = self.id_to_index.pop(old_id)
        self.id_to_index[new_id] = index

    def get_index(self, id_):
        # Get the index for a single id
        if id_ not in self.id_to_index:
            raise ValueError(f"ID {id_} not found")
        return self.id_to_index[id_]
