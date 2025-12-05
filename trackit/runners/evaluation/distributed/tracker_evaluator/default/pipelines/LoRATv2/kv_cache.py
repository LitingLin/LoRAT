from typing import Sequence

import torch

from trackit.miscellanies.system.machine.utils import sizeof_fmt
from trackit.runners.evaluation.distributed import EvaluatorContext
from ....components.tensor_cache import IndexAllocator
from trackit.models import ModelKVCacheSelfDescriptionMixin


class KVCacheMaintainer:
    def __init__(self, device: torch.device):
        self._device = device

    def start(self, context: EvaluatorContext):
        kv_cache_dtype = context.auto_mixed_precision_dtype if context.auto_mixed_precision_dtype is not None else context.dtype
        max_concurrent_tasks = context.max_batch_size * context.num_input_data_streams
        assert isinstance(context.model, ModelKVCacheSelfDescriptionMixin)
        kv_cache_shapes = context.model.get_kv_cache_shapes(max_concurrent_tasks)
        self.kv_caches = tuple((torch.empty(shape, device=self._device, dtype=kv_cache_dtype), torch.empty(shape, device=self._device, dtype=kv_cache_dtype))
                               for shape in kv_cache_shapes)
        self._id_index_mapper = IndexAllocator(max_concurrent_tasks)
        print('kv cache: initialized KV cache, size: '
              f'{sizeof_fmt(sum(cache[0].element_size() * cache[0].nelement() * 2 for cache in self.kv_caches))}, '
              f'max_concurrent_tasks: {max_concurrent_tasks}, depth: {len(self.kv_caches)}, '
              f'device: {self._device}')

    def stop(self):
        assert self._id_index_mapper.empty()
        del self._id_index_mapper
        del self.kv_caches

    def get_input_params(self, task_ids: Sequence[int]):
        kv_cache_indices = [self._id_index_mapper.get_index(task_id) for task_id in task_ids]

        if len(kv_cache_indices) == 0:
            return {}

        return {'kv_caches': self.kv_caches,
                'kv_cache_batch_idx': torch.tensor(kv_cache_indices, dtype=torch.int32, device=self._device)}

    def allocate(self, task_id: int):
        self._id_index_mapper.allocate(task_id)

    def free(self, task_id: int):
        self._id_index_mapper.free(task_id)
