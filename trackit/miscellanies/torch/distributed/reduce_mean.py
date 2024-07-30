import torch
import torch.distributed
from trackit.miscellanies.torch.distributed import is_dist_initialized, get_world_size, get_aux_process_group


def reduce_mean_(tensor: torch.Tensor):
    if is_dist_initialized():
        group = None
        if tensor.device.type == 'cpu':
            group = get_aux_process_group()
        torch.distributed.all_reduce(tensor, group=group)
        tensor.div_(get_world_size())
