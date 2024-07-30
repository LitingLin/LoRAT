import torch
from collections.abc import Sequence, Mapping, MutableSequence, MutableMapping


plain_object_classes = (str, bytes, bytearray, memoryview)


def get_torch_tensor_list_from_object(data):
    tensor_list = []
    if isinstance(data, torch.Tensor):
        tensor_list.append(data)
    elif isinstance(data, plain_object_classes):
        pass
    elif isinstance(data, Mapping):
        for v in data.values():
            tensor_list.extend(get_torch_tensor_list_from_object(v))
    elif isinstance(data, Sequence):
        for v in data:
            tensor_list.extend(get_torch_tensor_list_from_object(v))
    return tensor_list


def replace_torch_tensor_in_object(data, tensor_list: list):
    if isinstance(data, torch.Tensor):
        return tensor_list.pop(0)
    elif isinstance(data, plain_object_classes):
        return data
    elif isinstance(data, Mapping):
        return {k: replace_torch_tensor_in_object(sample, tensor_list) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(replace_torch_tensor_in_object(sample, tensor_list) for sample in data))
    elif isinstance(data, MutableSequence):
        return [replace_torch_tensor_in_object(sample, tensor_list) for sample in data]
    elif isinstance(data, Sequence):
        return tuple(replace_torch_tensor_in_object(sample, tensor_list) for sample in data)
    else:
        return data


def move_torch_tensors_in_object_to_device(data, device):
    tensor_list = get_torch_tensor_list_from_object(data)
    tensor_list = [tensor.to(device) for tensor in tensor_list]
    return replace_torch_tensor_in_object(data, tensor_list)
