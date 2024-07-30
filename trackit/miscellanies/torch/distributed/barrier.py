import torch.distributed as dist
from typing import Optional
from datetime import timedelta
from contextlib import contextmanager
from . import is_dist_initialized, get_rank, get_backend, get_aux_backend, get_aux_process_group, get_local_rank

_use_monitored_barrier = True


def _get_gloo_backend_process_group():
    if not is_dist_initialized():
        return None
    if get_aux_backend() == 'gloo':
        return get_aux_process_group()
    if get_backend() == 'gloo':
        return dist.group.WORLD
    return None


def _barrier(gloo_process_group: Optional[dist.ProcessGroup] = None, timeout: Optional[timedelta] = None):
    if gloo_process_group is not None and _use_monitored_barrier:
        dist.monitored_barrier(group=gloo_process_group, timeout=timeout)
    else:
        if timeout is not None:
            print(f'Warning: timeout is not supported for {get_backend()} backend')
        dist.barrier()


@contextmanager
def torch_distributed_zero_first(on_local_master: bool = True, timeout: Optional[timedelta] = None):
    rank = get_local_rank() if on_local_master else get_rank()
    gloo_process_group = _get_gloo_backend_process_group()
    if is_dist_initialized() and rank != 0:
        _barrier(gloo_process_group, timeout)
    try:
        yield
    finally:
        if is_dist_initialized() and rank == 0:
            _barrier(gloo_process_group, timeout)


def torch_distributed_barrier(timeout: Optional[timedelta] = None):
    if not is_dist_initialized():
        return
    gloo_process_group = _get_gloo_backend_process_group()
    _barrier(gloo_process_group, timeout)
