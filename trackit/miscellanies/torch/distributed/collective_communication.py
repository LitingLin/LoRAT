import pickle
import io
import hashlib
import torch
import torch.distributed
from typing import Optional
from trackit.miscellanies.torch.distributed import get_world_size, get_backend, get_rank, get_aux_process_group

_HASHLEN = 20


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


def _allocate_buffer(buffer_size: int, use_aux_process_group=True):
    shape = (get_world_size(), buffer_size)
    if get_backend() == 'nccl' and not use_aux_process_group:
        local_buffer = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        device_buffer = torch.empty(shape, dtype=torch.uint8, device='cuda')
    else:
        local_buffer = torch.empty(shape, dtype=torch.uint8)
        device_buffer = None
    return local_buffer, device_buffer


def _all_gather(object_: torch.Tensor, cpu_buffer: torch.Tensor, device_buffer: Optional[torch.Tensor], buffer_size: int, rank: int, use_aux_process_group=True):
    assert len(object_) <= buffer_size

    cpu_buffer.zero_()
    cpu_buffer[rank, :len(object_)] = object_
    process_group = None
    if use_aux_process_group:
        process_group = get_aux_process_group()
    if device_buffer is not None:
        device_buffer.zero_()
        device_buffer[rank].copy_(cpu_buffer[rank])
        torch.distributed.all_reduce(device_buffer, group=process_group)
        cpu_buffer.copy_(device_buffer)
    else:
        torch.distributed.all_reduce(cpu_buffer, group=process_group)


class CollectiveCommunication:
    def __init__(self, buffer_size: int = 16 * 1024, use_aux_process_group=True):
        use_aux_process_group = use_aux_process_group and get_aux_process_group() is not None
        self.local_buffer, self.device_buffer = _allocate_buffer(buffer_size, use_aux_process_group)
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
        if current_message_required_size > self.buffer_size:
            all_gather_phase_1_object = safepickle((None, index, current_message_required_size))
            all_gather_phase_1_object = torch.frombuffer(all_gather_phase_1_object, dtype=torch.uint8)
        assert len(all_gather_phase_1_object) <= self.buffer_size
        assert len(all_gather_phase_1_object) > 0

        _all_gather(all_gather_phase_1_object, self.local_buffer, self.device_buffer, self.buffer_size, self.rank)

        gathered_objects = []
        all_rank_indices = []
        all_rank_required_buffer_size = []
        for gathered_raw_object in self.local_buffer:
            gathered_object, gathered_index, required_buffer_size = safeunpickle(memoryview(gathered_raw_object.numpy()))
            if gathered_index != index:
                print(f'warning: gathered {gathered_index} vs index {index}')
            gathered_objects.append(gathered_object)
            all_rank_indices.append(gathered_index)
            all_rank_required_buffer_size.append(required_buffer_size)
        assert all(rank_index == index for rank_index in all_rank_indices), f"gathered rank indices {all_rank_indices}"

        phase_2_buffer_size = None
        for rank_required_buffer_size in all_rank_required_buffer_size:
            if rank_required_buffer_size is not None:
                assert isinstance(rank_required_buffer_size, int)
                if phase_2_buffer_size is None:
                    phase_2_buffer_size = rank_required_buffer_size
                else:
                    phase_2_buffer_size = max(phase_2_buffer_size, rank_required_buffer_size)

        if phase_2_buffer_size is not None:
            self.out_of_buffer_count += 1
            gathered_objects.clear()

            phase_2_local_buffer, phase_2_device_buffer = _allocate_buffer(phase_2_buffer_size)
            all_gather_phase_2_object = current_object

            assert len(all_gather_phase_2_object) > 0

            _all_gather(all_gather_phase_2_object, phase_2_local_buffer, phase_2_device_buffer, phase_2_buffer_size, self.rank)

            for gathered_phase_2_raw_object in phase_2_local_buffer:
                gathered_object, gathered_index, required_buffer_size = safeunpickle(memoryview(gathered_phase_2_raw_object.numpy()))
                assert required_buffer_size is None
                assert gathered_index == index, f"gathered {gathered_index} vs index {index} phase 2"
                gathered_objects.append(gathered_object)

        self.called_count += 1

        return gathered_objects
