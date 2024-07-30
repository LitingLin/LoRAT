import torch
import torch.distributed
from trackit.miscellanies.torch.distributed import is_dist_initialized, get_world_size, get_rank


class BatchValueStorage_DistributedAware:
    def __init__(self, batch_size, dtype):
        self.storage = torch.zeros(batch_size, dtype=dtype)

    def add(self, val):
        self.storage += val

    def sum(self):
        return torch.sum(self.storage)

    def state_dict(self):
        if is_dist_initialized():
            all_tensors = [torch.zeros(len(self.storage), dtype=self.storage.dtype) for _ in range(get_world_size())]
            torch.distributed.all_gather(all_tensors, self.storage)
            return torch.cat(all_tensors)
        else:
            return self.storage

    def load_state_dict(self, state):
        if is_dist_initialized():
            rank_id = get_rank()
            self.storage.copy_(state[rank_id * len(self.storage): (rank_id + 1) * len(self.storage)])
        else:
            self.storage.copy_(state)
