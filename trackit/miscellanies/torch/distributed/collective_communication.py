import pickle
import io
import hashlib
import torch
import torch.distributed
from typing import Optional, Union, Iterable
from trackit.miscellanies.torch.distributed import get_world_size, get_backend, get_rank, get_aux_process_group
from trackit.miscellanies.torch.check_version import is_torch_version_greater_or_equal

_HASHLEN = 20
_prefer_aux_process_group = True


def safepickle(obj):
    s = io.BytesIO()
    s.write(b'    ')
    assert s.tell() == 4
    pickle.dump(obj, s, protocol=pickle.HIGHEST_PROTOCOL)
    hash_ = hashlib.sha1(s.getbuffer()[4:]).digest()
    assert len(hash_) == _HASHLEN
    s.write(hash_)
    buffer_length = s.tell()
    s.seek(0)
    if buffer_length > 2 ** 32:
        print(obj)
        print(f"Warning: Pickle is too large! {buffer_length}")
        raise ValueError(f"Pickle is too large! {buffer_length}")
    s.write(int.to_bytes(buffer_length, length=4, byteorder='little', signed=False))
    return s.getbuffer()


def safeunpickle(pstr):
    size = pstr[: 4]
    size = int.from_bytes(size, byteorder='little', signed=False)
    assert size > 0
    data, checksum = pstr[4: size - _HASHLEN], pstr[size - _HASHLEN: size]
    if hashlib.sha1(data).digest() != checksum:
        raise ValueError("Pickle hash does not match!")
    return pickle.loads(data)


def _all_gather(object_: torch.Tensor, cpu_buffer: list[torch.Tensor], device_buffer: Optional[list[torch.Tensor]],
                object_buffer: Optional[torch.Tensor], use_aux_process_group=True):
    process_group = None
    if use_aux_process_group:
        process_group = get_aux_process_group()

    if device_buffer is not None:
        if object_buffer is None:
            object_buffer = object_.to(device_buffer[0].device)
        else:
            object_buffer[:len(object_)].copy_(object_)
        assert device_buffer[get_rank()].shape == object_buffer.shape, f"device_buffer {device_buffer[get_rank()].shape} != object_buffer {object_buffer.shape}, all buffer shape: {[buffer.shape for buffer in device_buffer]}"
        torch.distributed.all_gather(device_buffer, object_buffer, group=process_group)
        for i in range(len(device_buffer)):
            cpu_buffer[i].copy_(device_buffer[i])
    else:
        if object_buffer is None:
            object_buffer = object_
        else:
            object_buffer[:len(object_)].copy_(object_)
        assert cpu_buffer[get_rank()].shape == object_buffer.shape, f"cpu_buffer {cpu_buffer[get_rank()].shape} != object_buffer {object_buffer.shape}, all buffer shape: {[buffer.shape for buffer in cpu_buffer]}"
        torch.distributed.all_gather(cpu_buffer, object_buffer, group=process_group)


def _allocate_buffer(buffer_size: Union[int, Iterable[int]], use_aux_process_group=True):
    if isinstance(buffer_size, int):
        if _use_nccl_backend(use_aux_process_group):
            local_buffer = torch.empty((buffer_size,), dtype=torch.uint8, pin_memory=True)
            device_buffer = torch.empty((buffer_size,), dtype=torch.uint8, device='cuda')
        else:
            local_buffer = torch.empty((buffer_size,), dtype=torch.uint8)
            device_buffer = None
    else:
        if _use_nccl_backend(use_aux_process_group):
            local_buffer = [torch.empty((size,), dtype=torch.uint8, pin_memory=True) for size in buffer_size]
            device_buffer = [torch.empty((size,), dtype=torch.uint8, device='cuda') for size in buffer_size]
        else:
            local_buffer = [torch.empty((size,), dtype=torch.uint8) for size in buffer_size]
            device_buffer = None

    return local_buffer, device_buffer


def _allocate_object_buffer(buffer_size: int, use_aux_process_group: bool):
    return torch.empty((buffer_size,), dtype=torch.uint8,
                                         device='cuda' if _use_nccl_backend(use_aux_process_group) else 'cpu')


class CollectiveCommunication:
    def __init__(self, buffer_size: int = 16 * 1024, use_aux_process_group=_prefer_aux_process_group):
        use_aux_process_group = use_aux_process_group and get_aux_process_group() is not None
        self.local_buffer, self.device_buffer = _allocate_buffer([buffer_size] * get_world_size(), use_aux_process_group)
        self.object_buffer = _allocate_object_buffer(buffer_size, use_aux_process_group)
        self.buffer_size = buffer_size
        self.rank = get_rank()
        self.called_count = 0
        self.out_of_buffer_count = 0
        self.use_aux_process_group = use_aux_process_group

    def all_gather(self, object_, index = None):
        current_object = safepickle((object_, index, None))
        current_object = torch.frombuffer(current_object, dtype=torch.uint8)

        all_gather_phase_1_object = current_object
        current_message_required_size = len(current_object)
        current_message_out_of_buffer = current_message_required_size > self.buffer_size
        if current_message_out_of_buffer:
            all_gather_phase_1_object = safepickle((None, index, current_message_required_size))
            all_gather_phase_1_object = torch.frombuffer(all_gather_phase_1_object, dtype=torch.uint8)
        assert len(all_gather_phase_1_object) <= self.buffer_size
        assert len(all_gather_phase_1_object) > 0

        _all_gather(all_gather_phase_1_object, self.local_buffer, self.device_buffer, self.object_buffer, self.use_aux_process_group)

        gathered_objects = []
        all_rank_indices = []
        all_rank_requiring_buffer_size = []
        for gathered_raw_object in self.local_buffer:
            gathered_object, gathered_index, required_buffer_size = safeunpickle(memoryview(gathered_raw_object.numpy()))
            if gathered_index != index:
                print(f'warning: gathered {gathered_index} vs index {index}')
            gathered_objects.append(gathered_object)
            all_rank_indices.append(gathered_index)
            all_rank_requiring_buffer_size.append(required_buffer_size)
        assert all(rank_index == index for rank_index in all_rank_indices), f"gathered rank indices {all_rank_indices}"

        need_phase_2 = any(rank_requiring_buffer_size is not None for rank_requiring_buffer_size in all_rank_requiring_buffer_size)

        if need_phase_2:
            self.out_of_buffer_count += 1
            gathered_objects.clear()
            allow_uneven_buffer_size = _use_nccl_backend(self.use_aux_process_group) and is_torch_version_greater_or_equal((2, 5))

            all_gather_phase_2_object = current_object
            assert len(all_gather_phase_2_object) > 0
            if allow_uneven_buffer_size:
                phase_2_local_buffer = []
                phase_2_device_buffer = [] if self.device_buffer is not None else None
                for i, rank_requiring_buffer_size in enumerate(all_rank_requiring_buffer_size):
                    if rank_requiring_buffer_size is not None:
                        local_buffer, device_buffer = _allocate_buffer(rank_requiring_buffer_size, self.use_aux_process_group)
                    else:
                        local_buffer, device_buffer = self.local_buffer[i], self.device_buffer[i] if self.device_buffer is not None else None
                    phase_2_local_buffer.append(local_buffer)
                    if device_buffer is not None:
                        phase_2_device_buffer.append(device_buffer)

                if current_message_out_of_buffer:
                    phase_2_object_buffer = None
                else:
                    phase_2_object_buffer = self.object_buffer

            else:
                phase_2_buffer_size = max(rank_requiring_buffer_size for rank_requiring_buffer_size in all_rank_requiring_buffer_size if rank_requiring_buffer_size is not None)

                phase_2_local_buffer, phase_2_device_buffer = _allocate_buffer([phase_2_buffer_size] * get_world_size(),
                                                                         self.use_aux_process_group)
                phase_2_object_buffer = _allocate_object_buffer(phase_2_buffer_size, self.use_aux_process_group)

            _all_gather(all_gather_phase_2_object, phase_2_local_buffer, phase_2_device_buffer, phase_2_object_buffer,
                        self.use_aux_process_group)

            for gathered_phase_2_raw_object in phase_2_local_buffer:
                gathered_object, gathered_index, required_buffer_size = safeunpickle(memoryview(gathered_phase_2_raw_object.numpy()))
                assert required_buffer_size is None
                assert gathered_index == index, f"gathered {gathered_index} vs index {index} phase 2"
                gathered_objects.append(gathered_object)

        self.called_count += 1

        return gathered_objects

    def gather(self, object_, rank: int, index = None):
        current_object = safepickle((object_, index, None))
        current_object = torch.frombuffer(current_object, dtype=torch.uint8)

        all_gather_phase_1_object = current_object
        current_message_required_size = len(current_object)
        current_message_out_of_buffer = current_message_required_size > self.buffer_size
        if current_message_out_of_buffer:
            all_gather_phase_1_object = safepickle((None, index, current_message_required_size))
            all_gather_phase_1_object = torch.frombuffer(all_gather_phase_1_object, dtype=torch.uint8)
        assert len(all_gather_phase_1_object) <= self.buffer_size
        assert len(all_gather_phase_1_object) > 0

        _all_gather(all_gather_phase_1_object, self.local_buffer, self.device_buffer, self.object_buffer, self.use_aux_process_group)

        gathered_objects = []
        all_rank_indices = []
        all_rank_requiring_buffer_size = []
        for gathered_raw_object in self.local_buffer:
            gathered_object, gathered_index, required_buffer_size = safeunpickle(memoryview(gathered_raw_object.numpy()))
            if gathered_index != index:
                print(f'warning: gathered {gathered_index} vs index {index}')
            gathered_objects.append(gathered_object)
            all_rank_indices.append(gathered_index)
            all_rank_requiring_buffer_size.append(required_buffer_size)
        assert all(rank_index == index for rank_index in all_rank_indices), f"gathered rank indices {all_rank_indices}"

        need_phase_2 = any(rank_requiring_buffer_size is not None for rank_requiring_buffer_size in all_rank_requiring_buffer_size)

        if need_phase_2:
            self.out_of_buffer_count += 1
            gathered_objects.clear()
            all_gather_phase_2_object = current_object
            assert len(all_gather_phase_2_object) > 0
            phase_2_buffer_size = max(rank_requiring_buffer_size for rank_requiring_buffer_size in all_rank_requiring_buffer_size if rank_requiring_buffer_size is not None)

            if get_rank() == rank:
                phase_2_local_buffer, phase_2_device_buffer = _allocate_buffer([phase_2_buffer_size] * get_world_size(),
                                                                         self.use_aux_process_group)
            else:
                phase_2_local_buffer, phase_2_device_buffer = None, None
            phase_2_object_buffer = _allocate_object_buffer(phase_2_buffer_size, self.use_aux_process_group)

            device = self.device_buffer[0].device if self.device_buffer is not None else torch.device('cpu')
            _gather(all_gather_phase_2_object, phase_2_local_buffer, phase_2_device_buffer, phase_2_object_buffer,
                    rank, device, self.use_aux_process_group)

            if get_rank() == rank:
                for gathered_phase_2_raw_object in phase_2_local_buffer:
                    gathered_object, gathered_index, required_buffer_size = safeunpickle(memoryview(gathered_phase_2_raw_object.numpy()))
                    assert required_buffer_size is None
                    assert gathered_index == index, f"gathered {gathered_index} vs index {index} phase 2"
                    gathered_objects.append(gathered_object)

        self.called_count += 1

        return gathered_objects if get_rank() == rank else None


def _use_nccl_backend(use_aux_process_group: bool):
    return get_backend() == 'nccl' and not use_aux_process_group


def _gather(object_: torch.Tensor, cpu_buffer: list[torch.Tensor], device_buffer: Optional[list[torch.Tensor]],
            object_buffer: Optional[torch.Tensor], dst_rank: int, device: torch.device,
            use_aux_process_group=True):
    process_group = None
    if use_aux_process_group:
        process_group = get_aux_process_group()

    if device.type != 'cpu':
        if object_buffer is None:
            object_buffer = object_.to(device)
        else:
            object_buffer[:len(object_)].copy_(object_)
        if device_buffer is not None:
            assert device_buffer[get_rank()].shape == object_buffer.shape, f"device_buffer {device_buffer[get_rank()].shape} != object_buffer {object_buffer.shape}, all buffer shape: {[buffer.shape for buffer in device_buffer]}"
            if get_rank() != dst_rank:
                device_buffer = None
        torch.distributed.gather(object_buffer, device_buffer, dst_rank, group=process_group)
        if get_rank() == dst_rank:
            for i in range(len(device_buffer)):
                cpu_buffer[i].copy_(device_buffer[i])
    else:
        if object_buffer is None:
            object_buffer = object_
        else:
            object_buffer[:len(object_)].copy_(object_)
        if cpu_buffer is not None:
            assert cpu_buffer[get_rank()].shape == object_buffer.shape, f"cpu_buffer {cpu_buffer[get_rank()].shape} != object_buffer {object_buffer.shape}, all buffer shape: {[buffer.shape for buffer in cpu_buffer]}"
            if get_rank() != dst_rank:
                cpu_buffer = None
        torch.distributed.gather(object_buffer, cpu_buffer, dst_rank, group=process_group)
