import torch
import gc
from typing import Protocol, MutableSequence, Any, Iterable

from trackit.miscellanies.torch.nested_tensor_helper import extract_torch_tensor_list_from_object, rebuild_object_with_given_torch_tensor_list


class TensorFilter(Protocol):
    def select(self, data: Any) -> MutableSequence[torch.Tensor]:
        ...

    def rearrange(self, data: Any, device_tensors: MutableSequence[torch.Tensor]) -> Any:
        ...


class DefaultTensorFilter:
    @staticmethod
    def select(data):
        return extract_torch_tensor_list_from_object(data)

    @staticmethod
    def rearrange(data, device_tensors):
        return rebuild_object_with_given_torch_tensor_list(data, device_tensors)


class TensorFilteringByIndices:
    def __init__(self, indices):
        self.indices = indices

    def select(self, data):
        split_points = []
        device_tensor_list = []
        for index in self.indices:
            datum = data[index]
            if datum is not None:
                device_tensors = extract_torch_tensor_list_from_object(datum)
                split_points.append(len(device_tensors))
                device_tensor_list.extend(device_tensors)
        return device_tensor_list

    def rearrange(self, data, device_tensors: list):
        collated = []
        for index, datum in enumerate(data):
            if index in self.indices and datum is not None:
                datum = rebuild_object_with_given_torch_tensor_list(datum, device_tensors)
            collated.append(datum)
        return collated


class CUDATensorStreamer_WithDataPrefetching:
    def __init__(self, iterator: Iterable, device: torch.device, tensor_filter: TensorFilter):
        self.iterator = iterator
        self.device = device
        self.tensor_filter = tensor_filter
        self.tensor_list = None

    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ] # in pytorch/torch/utils/data/sampler.py
    #   raise TypeError if __len__ is not implemented
    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.iter = iter(self.iterator)
        self.preload()
        assert self.tensor_list is not None, "empty iterator"
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        data = self.data
        tensor_list = self.tensor_list

        if data is StopIteration:
            if hasattr(self, 'stream'):
                del self.stream
                del self.iter
            raise StopIteration

        for tensor in tensor_list:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        data = self.tensor_filter.rearrange(data, tensor_list)
        assert len(tensor_list) == 0
        return data

    def preload(self):
        try:
            self.data = next(self.iter)
        except StopIteration:
            self.data = StopIteration
            self.tensor_list = None
            return

        self.tensor_list = self.tensor_filter.select(self.data)

        with torch.cuda.stream(self.stream):
            for i in range(len(self.tensor_list)):
                self.tensor_list[i] = self.tensor_list[i].to(self.device, non_blocking=True)


class DeviceTensorStreamer:
    def __init__(self, iterator: Iterable, device: torch.device, device_tensor_selector: TensorFilter):
        self.iterator = iterator
        self.device = device
        self.tensor_filter = device_tensor_selector

    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ] # in pytorch/torch/utils/data/sampler.py
    #   raise TypeError if __len__ is not implemented
    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        self.iter = iter(self.iterator)
        return self

    def __next__(self):
        data = next(self.iter)
        tensor_list = self.tensor_filter.select(data)

        for i in range(len(tensor_list)):
            tensor_list[i] = tensor_list[i].to(self.device)

        data = self.tensor_filter.rearrange(data, tensor_list)
        assert len(tensor_list) == 0
        return data


def build_device_tensor_streamer(iterator, device, tensor_filter=DefaultTensorFilter, prefetch=False):
    if 'cuda' == device.type and prefetch:
        return CUDATensorStreamer_WithDataPrefetching(iterator, device, tensor_filter)
    else:
        return DeviceTensorStreamer(iterator, device, tensor_filter)
