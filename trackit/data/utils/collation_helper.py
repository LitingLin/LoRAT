import torch
import numpy as np
from typing import Sequence, Union, Tuple, List
import torch.utils.data


# from torch.utils.data._utils.collate
def _collate_tensor_fn(batch: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        new_shape = list(elem.size())
        new_shape.insert(dim, len(batch))
        out = elem.new(storage).resize_(*new_shape)
    return torch.stack(batch, dim, out=out)


def collate_element_as_torch_tensor(data_list: Sequence, element_index_or_key: Union[int, str], dim=0):
    tensors = tuple(data[element_index_or_key] for data in data_list)
    if isinstance(tensors[0], torch.Tensor):
        return _collate_tensor_fn(tensors, dim=dim)
    elif isinstance(tensors[0], np.ndarray):
        tensors = tuple(torch.from_numpy(data) for data in tensors)
        return _collate_tensor_fn(tensors, dim=dim)
    else:
        tensors = tuple(torch.tensor(data) for data in tensors)
        return _collate_tensor_fn(tensors)


def collate_element_as_np_array(data_list: Sequence, element_index_or_key: Union[int, str], dim=0):
    tensors = tuple(data[element_index_or_key] for data in data_list)
    if isinstance(tensors[0], torch.Tensor):
        return torch.stack(tensors, dim=dim).numpy()
    elif isinstance(tensors[0], np.ndarray):
        return np.stack(tensors, axis=dim)
    else:
        return np.stack(tuple(np.array(data) for data in tensors), axis=dim)
