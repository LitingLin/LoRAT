import os
import sys

from contextlib import contextmanager
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

_dist_enabled: bool = False

_rank = 0
_world_size: int = 1
_local_rank: int = 0
_local_world_size: int = 1

_group_aux: Optional[dist.ProcessGroup] = None


def is_dist_initialized() -> bool:
    return _dist_enabled


def get_world_size() -> int:
    return _world_size


def get_backend() -> Optional[dist.Backend]:
    if not is_dist_initialized():
        return None
    return dist.get_backend()


def get_aux_backend() -> Optional[dist.Backend]:
    if _group_aux is None:
        return None
    return dist.get_backend(_group_aux)


# may return None
#  None -> use main process group
def get_aux_process_group() -> Optional[dist.ProcessGroup]:
    return _group_aux


def get_local_rank() -> int:
    return _local_rank


def get_local_world_size() -> int:
    return _local_world_size


def get_node_index() -> int:
    return get_rank() // get_local_world_size()


def get_num_nodes() -> int:
    return get_world_size() // get_local_world_size()


def get_rank() -> int:
    return _rank


def is_main_process() -> bool:
    return get_rank() == 0


def is_local_main_process() -> bool:
    return get_local_rank() == 0


def _log_info(message: str, dist_backend: str, rank: int, world_size: int, local_rank: int, local_world_size: int):
    sys.stdout.write(f'| [{dist_backend}] [rank ({rank}/{world_size}) node {rank // local_world_size}.{local_rank}] {message}\n')
    sys.stdout.flush()


def init_torch_distributed(device: str, silent_non_local_master: bool = True, use_aux_process_group: bool = True):
    global _dist_enabled
    global _rank
    global _world_size
    global _local_rank
    global _local_world_size
    if 'RANK' not in os.environ:
        print('Not using distributed mode')
        _dist_enabled = False
        _rank = 0
        _world_size = 1
        _local_rank = 0
        _local_world_size = 1
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

    if device == 'cuda':
        torch.cuda.set_device(local_rank)  # important
        dist_backend = 'nccl'
    else:
        dist_backend = 'gloo'

    _log_info('torch.distributed initializing', dist_backend, rank, world_size, local_rank, local_world_size)
    dist.init_process_group(backend=dist_backend, init_method='env://', world_size=world_size, rank=rank)
    _log_info('torch.distributed initialized', dist_backend, rank, world_size, local_rank, local_world_size)

    # dist.barrier()

    if dist_backend == 'nccl' and use_aux_process_group:
        global _group_aux

        aux_dist_backend = 'gloo'
        _log_info('torch.distributed initializing', aux_dist_backend, rank, world_size, local_rank, local_world_size)
        _group_aux = dist.new_group(backend=aux_dist_backend, timeout=timedelta(days=1))
        assert _group_aux is not None, 'Failed to create auxiliary process group'
        _log_info('torch.distributed initialized', aux_dist_backend, rank, world_size, local_rank, local_world_size)

    if silent_non_local_master:
        if local_rank != 0:
            f = open(os.devnull, 'w')
            sys.stdout = f

    _rank = rank
    _world_size = world_size
    _local_rank = local_rank
    _local_world_size = local_world_size

    _dist_enabled = True


def cleanup_torch_distributed():
    global _dist_enabled
    if _dist_enabled:
        global _group_aux
        dist.barrier()
        if _group_aux is not None:
            dist.destroy_process_group(_group_aux)
            _group_aux = None
        dist.destroy_process_group()
        print('| torch.distributed disabled', flush=True)
        if _local_rank != 0:
            sys.stdout = sys.__stdout__
        _dist_enabled = False


@contextmanager
def torch_distributed_disable_temporarily():
    global _dist_enabled

    dist_enabled = _dist_enabled
    _dist_enabled = False
    try:
        yield
    finally:
        _dist_enabled = dist_enabled
