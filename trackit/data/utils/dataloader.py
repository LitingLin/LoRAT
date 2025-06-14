from functools import partial
from typing import Optional
import torch
import torch.utils.data

from trackit.core.runtime.build_context import BuildContext
from trackit.data.context.worker import WorkerInfoInitializer
from trackit.miscellanies.torch.distributed import is_dist_initialized, get_rank, get_world_size
from trackit.data.utils.device_tensor_streamer import build_device_tensor_streamer, DefaultTensorFilter, TensorFilter


def build_dataloader(dataset: torch.utils.data.Dataset, batch_size: Optional[int], num_workers: int,
                     build_context: BuildContext,
                     do_shuffle: bool = True,
                     device_tensor_selection_filter: Optional[TensorFilter] = DefaultTensorFilter, collate_fn=None):
    device = build_context.device
    pin_memory = build_context.pin_memory and device.type == 'cuda'

    torch_distributed_enabled = is_dist_initialized()

    worker_info_initializer = WorkerInfoInitializer(get_rank(), build_context.seed,
                                                    num_workers if num_workers > 0 else 1, get_world_size(),
                                                    num_workers != 0)
    worker_init_fn = partial(custom_worker_init_fn, additional_fn=worker_info_initializer.worker_init_fn) if num_workers > 0 else None

    if collate_fn is None and batch_size is None:
        collate_fn = no_op_collate_fn

    if isinstance(dataset, torch.utils.data.IterableDataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                  worker_init_fn=worker_init_fn,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    else:
        has_len = True
        try:
            len(dataset)
        except TypeError:
            has_len = False
        if has_len:
            if torch_distributed_enabled:
                sampler = torch.utils.data.DistributedSampler(dataset, shuffle=do_shuffle)
                if do_shuffle:
                    build_context.services.event.register_on_epoch_begin_event_listener(
                        lambda epoch, is_train: sampler.set_epoch(epoch))
            else:
                if do_shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
        else:
            sampler = DataloaderInfiniteSampler()

        drop_last = True
        if batch_size is None:
            drop_last = False

        data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last,
                                                  worker_init_fn=worker_init_fn,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)

    build_context.services.event.register_on_epoch_begin_event_listener(worker_info_initializer.on_epoch_begin)
    build_context.services.event.register_on_epoch_end_event_listener(worker_info_initializer.on_epoch_end)

    if device.type != 'cpu' and device_tensor_selection_filter is not None:
        data_loader = build_device_tensor_streamer(data_loader, device, device_tensor_selection_filter, pin_memory)

    return data_loader


class DataloaderInfiniteSampler:
    def __iter__(self):
        count = 0
        while True:
            yield count
            count += 1


def no_op_collate_fn(data):
    return data


def custom_worker_init_fn(worker_id: int, additional_fn=None):
    import torch
    torch.set_num_threads(1)
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
    if additional_fn is not None:
        additional_fn(worker_id)
