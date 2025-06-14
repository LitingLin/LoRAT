import torch
from collections.abc import Sequence, Mapping, MutableSequence


plain_object_classes = (str, bytes, bytearray, memoryview)


def extract_torch_tensor_list_from_object(object_):
    tensor_list = []
    if isinstance(object_, torch.Tensor):
        tensor_list.append(object_)
    elif isinstance(object_, plain_object_classes):
        pass
    elif isinstance(object_, Mapping):
        for v in object_.values():
            tensor_list.extend(extract_torch_tensor_list_from_object(v))
    elif isinstance(object_, Sequence):
        for v in object_:
            tensor_list.extend(extract_torch_tensor_list_from_object(v))
    return tensor_list


def rebuild_object_with_given_torch_tensor_list(object_, tensor_list: list):
    if isinstance(object_, torch.Tensor):
        return tensor_list.pop(0)
    elif isinstance(object_, plain_object_classes):
        return object_
    elif isinstance(object_, Mapping):
        return {k: rebuild_object_with_given_torch_tensor_list(sample, tensor_list) for k, sample in object_.items()}
    elif isinstance(object_, tuple) and hasattr(object_, '_fields'):  # namedtuple
        return type(object_)(*(rebuild_object_with_given_torch_tensor_list(sample, tensor_list) for sample in object_))
    elif isinstance(object_, MutableSequence):
        return [rebuild_object_with_given_torch_tensor_list(sample, tensor_list) for sample in object_]
    elif isinstance(object_, Sequence):
        return tuple(rebuild_object_with_given_torch_tensor_list(sample, tensor_list) for sample in object_)
    else:
        return object_


def move_nested_torch_tensors_to(object_, *args, **kwargs):
    tensor_list = extract_torch_tensor_list_from_object(object_)
    tensor_list = [tensor.to(*args, **kwargs) for tensor in tensor_list]
    return rebuild_object_with_given_torch_tensor_list(object_, tensor_list)


def make_nested_torch_tensors_contiguous(object_):
    tensor_list = extract_torch_tensor_list_from_object(object_)
    tensor_list = [tensor.contiguous() for tensor in tensor_list]
    return rebuild_object_with_given_torch_tensor_list(object_, tensor_list)


def make_nested_torch_tensors_channels_last(object_):
    tensor_list = extract_torch_tensor_list_from_object(object_)
    tensor_list = [tensor.to(memory_format=torch.channels_last) if tensor.ndim == 4 else tensor for tensor in tensor_list]
    return rebuild_object_with_given_torch_tensor_list(object_, tensor_list)
