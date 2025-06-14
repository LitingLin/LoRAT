from typing import List, Optional

import torch.optim.lr_scheduler
from timm.scheduler.scheduler import Scheduler


class TimmLRSchedulerWrapper(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, timm_scheduler: Scheduler, last_t: int = -1) -> None:
        self.optimizer = timm_scheduler.optimizer
        self.timm_scheduler = timm_scheduler
        self.last_epoch = last_t

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def get_lr(self) -> List[float]:
        return self.timm_scheduler._get_lr(self.last_epoch)

    def state_dict(self):
        return self.last_epoch, self.timm_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.last_epoch, state_dict = state_dict
        self.timm_scheduler.load_state_dict(state_dict)

    def step(self, t: Optional[int] = None):
        if t is None:
            self.last_epoch += 1
        else:
            self.last_epoch = t
        if self.timm_scheduler.t_in_epochs:
            self.timm_scheduler.step(self.last_epoch)
        else:
            self.timm_scheduler.step_update(self.last_epoch)
        self._last_lr = [group[self.timm_scheduler.param_group_field] for group in self.optimizer.param_groups]
